import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import argparse
import time
import numpy as np
from tqdm import tqdm
import signal
import sys
import json
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from track.data_track import MOT16Sequences
from track.joint_model_hybrid import SpatialAttentionModule, ChannelAttentionModule
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class MOT16ReIDDataset(Dataset):
    """Dataset wrapper for MOT16 to extract person crops for ReID training"""
    
    def __init__(self, sequences, transform=None):
        """
        Args:
            sequences: List of MOT16Sequence objects
            transform: Optional transform to be applied on a sample
        """
        self.transform = transform
        self.data = []
        self.person_crops = []
        self.person_ids = []
        
        # Collect all frames from all sequences
        for seq in sequences:
            for frame_data in seq:
                img = frame_data['img']  # Already a tensor
                gt_boxes_dict = frame_data['gt']
                
                # Extract person crops and IDs
                for person_id, box in gt_boxes_dict.items():
                    # Convert box to integers for cropping
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Ensure box coordinates are valid
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img.shape[2] - 1, x2)
                    y2 = min(img.shape[1] - 1, y2)
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Extract crop
                    crop = img[:, y1:y2, x1:x2]
                    
                    # Resize crop to a standard size
                    try:
                        crop = F.interpolate(crop.unsqueeze(0), size=(128, 64), mode='bilinear', align_corners=False).squeeze(0)
                        self.person_crops.append(crop)
                        self.person_ids.append(person_id)
                    except Exception as e:
                        print(f"Error resizing crop: {e}, box: {box}")
                        continue
    
    def __len__(self):
        return len(self.person_crops)
    
    def __getitem__(self, idx):
        crop = self.person_crops[idx]
        person_id = self.person_ids[idx]
        
        # Apply transforms if any
        if self.transform is not None:
            crop = self.transform(crop)
        
        return crop, person_id


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining"""
    
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Feature matrix with shape (batch_size, feat_dim)
            targets: Ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        
        # Debug info
        print(f"TripletLoss input: batch size={n}, feature dim={inputs.size(1)}")
        print(f"Targets: {targets}")
        print(f"Unique targets: {torch.unique(targets)}")
        
        # Count samples per ID
        id_counts = {}
        for id in targets.cpu().numpy():
            if id not in id_counts:
                id_counts[id] = 0
            id_counts[id] += 1
        print(f"Samples per ID: {id_counts}")
        
        # Check if we have enough positive samples for each ID
        valid_triplet_possible = False
        for id, count in id_counts.items():
            if count >= 2:  # Need at least 2 samples of the same ID for positive pair
                valid_triplet_possible = True
                break
        
        if not valid_triplet_possible:
            print("WARNING: No valid triplets possible in this batch!")
            # Return a small non-zero loss to ensure training continues
            return torch.tensor(0.1, device=inputs.device, requires_grad=True)
        
        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # For numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        valid_anchors = 0
        
        for i in range(n):
            # Check if this anchor has any positives
            if mask[i].sum() <= 1:  # Only self as positive
                print(f"Anchor {i} (ID {targets[i].item()}) has no positives, skipping")
                continue
                
            # Check if this anchor has any negatives
            if (mask[i] == 0).sum() == 0:  # No negatives
                print(f"Anchor {i} (ID {targets[i].item()}) has no negatives, skipping")
                continue
            
            # Find hardest positive (same ID, furthest distance)
            max_pos_dist = dist[i][mask[i] & (torch.arange(n, device=mask.device) != i)].max()
            dist_ap.append(max_pos_dist.unsqueeze(0))
            
            # Find hardest negative (different ID, closest distance)
            min_neg_dist = dist[i][mask[i] == 0].min()
            dist_an.append(min_neg_dist.unsqueeze(0))
            
            valid_anchors += 1
            
            # Debug info
            print(f"Anchor {i} (ID {targets[i].item()}): pos_dist={max_pos_dist.item():.4f}, neg_dist={min_neg_dist.item():.4f}")
        
        if not dist_ap:  # No valid triplets found
            print("WARNING: No valid triplets found in the batch!")
            # Return a small non-zero loss to ensure training continues
            return torch.tensor(0.1, device=inputs.device, requires_grad=True)
        
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Debug info
        print(f"Valid anchors: {valid_anchors}")
        print(f"Mean positive distance: {dist_ap.mean().item():.4f}")
        print(f"Mean negative distance: {dist_an.mean().item():.4f}")
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # Debug info
        print(f"Computed triplet loss: {loss.item():.6f}")
        
        # Ensure loss is not zero
        if loss.item() == 0:
            print("WARNING: Loss is zero, adding small constant")
            loss = loss + 1e-5
            
        return loss


# Custom ChannelAttentionModule with feature extraction for ReID
class ChannelAttentionReID(nn.Module):
    def __init__(self, in_channels, feature_dim=256, reduction_ratio=16):
        super(ChannelAttentionReID, self).__init__()
        
        # Add a convolutional layer to transform input to the expected number of channels
        self.conv_transform = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention module
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        
        # Feature extraction for ReID
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extraction = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
    
    def forward(self, x):
        # Transform input to expected number of channels
        x = self.conv_transform(x)
        
        # Apply channel attention
        attention_map = self.channel_attention(x)
        x = x * attention_map  # Apply attention map
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Extract ReID features
        features = self.feature_extraction(x)
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        return features


def parse_args():
    parser = argparse.ArgumentParser(description='Train ReID head with channel/spatial attention modules')
    parser.add_argument('--module_type', type=str, default='spatial', choices=['spatial', 'channel'], 
                        help='Type of attention module to train (spatial or channel)')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/reid', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seq_names', type=str, nargs='+', default=['MOT16-02', 'MOT16-04', 'MOT16-05'], 
                        help='Sequence names for training')
    parser.add_argument('--val_seq_names', type=str, nargs='+', default=['MOT16-09'], 
                        help='Sequence names for validation')
    parser.add_argument('--device', type=str, default='', help='cuda or cpu')
    parser.add_argument('--lr_step_size', type=int, default=7, help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for batches')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to use (for quick testing)')
    parser.add_argument('--save_prefix', type=str, default=None, help='Prefix for saved model files')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension for ReID')
    parser.add_argument('--in_channels', type=int, default=256, help='Input channels for attention module')
    parser.add_argument('--margin', type=float, default=0.3, help='Margin for triplet loss')
    parser.add_argument('--reduction_ratio', type=int, default=16, help='Reduction ratio for channel attention')
    parser.add_argument('--pretrained_detector', type=str, default=None, 
                        help='Path to pretrained detector for feature extraction')
    return parser.parse_args()


def extract_roi_features(model, images, boxes, device):
    """Extract ROI features from detector backbone"""
    # Get backbone features
    features = model.backbone(images)
    
    # Apply ROI pooling
    box_features = model.roi_heads.box_roi_pool(features, [boxes], [images.shape[-2:]])
    
    return box_features


# Modified SpatialAttentionModule for training directly on RGB images
class ModifiedSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, feature_dim):
        super(ModifiedSpatialAttentionModule, self).__init__()
        
        # Add a convolutional layer to transform input to the expected number of channels
        self.conv_transform = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention module to generate MC
        self.channel_attention = ChannelAttentionModule(in_channels)
        
        # 3×3 convolution for spatial attention map MS (F3×3×3)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        # CNN feature extractor for refined features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global pooling for vector representation
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final embedding layers for ReID feature extraction
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        # Transform input to expected number of channels
        x = self.conv_transform(x)
        
        # Apply channel attention to get MC
        MC = self.channel_attention(x)
        
        # Compute XC = X ⊗ MC (equation 8)
        XC = x * MC
        
        # Generate spatial attention features - Max_pool and Avg_pool operations
        # Average pooling along channel dimension
        avg_out = torch.mean(XC, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Max pooling along channel dimension
        max_out, _ = torch.max(XC, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Concatenate pooled features along channel dimension [B, 2, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution F3×3×3 and sigmoid to get spatial attention map MS (equation 9)
        MS = self.sigmoid(self.conv(concat))  # [B, 1, H, W]
        
        # Compute Y = XC ⊗ MS (equation 10) - element-wise multiplication
        Y = XC * MS  # Broadcast MS across all channels
        
        # Apply CNN layers for feature extraction
        Y = self.conv_layers(Y)
        
        # Global pooling converts spatial features to a vector
        Y = self.global_avg_pool(Y)
        Y = Y.view(Y.size(0), -1)
        
        # FC layers produce the final embedding
        Y = self.fc_layers(Y)
        
        # L2 normalization makes features suitable for similarity comparison
        Y = F.normalize(Y, p=2, dim=1)
        
        return Y


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch, args):
    """Train the model for one epoch"""
    model.train()
    
    total_loss = 0
    batch_times = []
    successful_batches = 0
    failed_batches = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (crops, person_ids) in pbar:
        try:
            start_time = time.time()
            
            # Skip batches with too few samples for triplet loss
            if len(crops) < 3:
                print(f"  Skipping batch {batch_idx} with only {len(crops)} samples (need at least 3)")
                failed_batches += 1
                continue
                
            # Skip batches with too few unique IDs
            unique_ids = torch.unique(person_ids)
            if len(unique_ids) < 2:
                print(f"  Skipping batch {batch_idx} with only {len(unique_ids)} unique IDs (need at least 2)")
                failed_batches += 1
                continue
            
            # Check if we have enough samples per ID
            id_counts = {}
            for id in person_ids.cpu().numpy():
                if id not in id_counts:
                    id_counts[id] = 0
                id_counts[id] += 1
            
            valid_triplet_possible = False
            for id, count in id_counts.items():
                if count >= 2:  # Need at least 2 samples of the same ID for positive pair
                    valid_triplet_possible = True
                    break
            
            if not valid_triplet_possible:
                print(f"  Skipping batch {batch_idx}: No ID has multiple samples for positive pairs")
                failed_batches += 1
                continue
            
            # Move data to device
            crops = crops.to(device)
            person_ids = person_ids.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            features = model(crops)
            
            # Compute loss
            loss = criterion(features, person_ids)
            
            # Check if loss is valid
            if loss is None or (isinstance(loss, torch.Tensor) and (torch.isnan(loss) or torch.isinf(loss))):
                print(f"  Skipping batch {batch_idx}: Invalid loss value: {loss}")
                failed_batches += 1
                continue
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate batch time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Update statistics
            loss_value = loss.item()
            total_loss += loss_value
            successful_batches += 1
            
            # Update progress bar
            if batch_idx % args.log_interval == 0:
                avg_time = np.mean(batch_times[-10:]) if batch_times else 0
                avg_loss = total_loss / max(successful_batches, 1)
                
                pbar.set_description(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(data_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {avg_time:.3f}s/batch"
                )
        
        except Exception as e:
            print(f"\nError in training batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            failed_batches += 1
            continue
    
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
    else:
        avg_loss = float('inf')
        print("Warning: All training batches failed!")
    
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f} (successful batches: {successful_batches}/{len(data_loader)}, failed: {failed_batches})")
    
    return avg_loss


def validate(model, data_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    successful_batches = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, (crops, person_ids) in enumerate(tqdm(data_loader, desc="Validating")):
            try:
                # Skip batches with too few samples for triplet loss
                if len(crops) < 3:
                    failed_batches += 1
                    continue
                    
                # Skip batches with too few unique IDs
                unique_ids = torch.unique(person_ids)
                if len(unique_ids) < 2:
                    failed_batches += 1
                    continue
                
                # Check if we have enough samples per ID
                id_counts = {}
                for id in person_ids.cpu().numpy():
                    if id not in id_counts:
                        id_counts[id] = 0
                    id_counts[id] += 1
                
                valid_triplet_possible = False
                for id, count in id_counts.items():
                    if count >= 2:  # Need at least 2 samples of the same ID for positive pair
                        valid_triplet_possible = True
                        break
                
                if not valid_triplet_possible:
                    failed_batches += 1
                    continue
                
                # Move data to device
                crops = crops.to(device)
                person_ids = person_ids.to(device)
                
                # Forward pass
                features = model(crops)
                
                # Compute loss
                loss = criterion(features, person_ids)
                
                # Check if loss is valid
                if loss is None or (isinstance(loss, torch.Tensor) and (torch.isnan(loss) or torch.isinf(loss))):
                    failed_batches += 1
                    continue
                
                # Update statistics
                loss_value = loss.item()
                total_loss += loss_value
                successful_batches += 1
                
            except Exception as e:
                print(f"\nError during validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                failed_batches += 1
                continue
    
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
    else:
        avg_loss = float('inf')
        print("Warning: All validation batches failed!")
    
    print(f"Validation Loss: {avg_loss:.4f} (successful batches: {successful_batches}/{len(data_loader)}, failed: {failed_batches})")
    
    return avg_loss


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, best_val_loss, checkpoint_path, is_best=False, save_prefix='reid'):
    """Save model checkpoint with training history"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model if needed
    if is_best:
        best_model_path = os.path.join(os.path.dirname(checkpoint_path), f"{save_prefix}_best.pth")
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to {best_model_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {epoch} with validation loss: {best_val_loss}")
            
        return epoch, best_val_loss
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch")
        return 0, float('inf')


def setup_signal_handler(model, optimizer, scheduler, train_loss, val_loss, best_val_loss, epoch, args):
    """Setup signal handler for graceful interruption"""
    def signal_handler(sig, frame):
        print("\nTraining interrupted! Saving checkpoint before exiting...")
        interrupted_checkpoint_path = os.path.join(args.output_dir, f"{args.save_prefix}_interrupted.pth")
        save_checkpoint(
            epoch, model, optimizer, scheduler, 
            train_loss, val_loss, best_val_loss, 
            interrupted_checkpoint_path, 
            False, 
            args.save_prefix
        )
        
        # Save training history
        history_path = os.path.join(args.output_dir, f"{args.save_prefix}_training_history.json")
        history_data = {
            'interrupted_at_epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f)
        
        print(f"Checkpoint and history saved. Exiting...")
        sys.exit(0)
    
    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    return signal_handler


def visualize_features(model, data_loader, device, num_samples=100, output_dir=None, title_prefix="ReID"):
    """Visualize ReID features using t-SNE"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        model.eval()
        
        # Collect features and labels
        features = []
        labels = []
        
        with torch.no_grad():
            for crops, person_ids in tqdm(data_loader, desc="Extracting features"):
                # Move data to device
                crops = crops.to(device)
                
                # Forward pass
                batch_features = model(crops)
                
                # Store features and labels
                features.append(batch_features.cpu())
                labels.append(person_ids.cpu())
                
                # Limit the number of samples
                if len(features) * crops.size(0) >= num_samples:
                    break
        
        # Concatenate features and labels
        features = torch.cat(features, dim=0)[:num_samples]
        labels = torch.cat(labels, dim=0)[:num_samples]
        
        # Convert to numpy
        features_np = features.numpy()
        labels_np = labels.numpy()
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features_np)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Get unique labels
        unique_labels = np.unique(labels_np)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], c=[colors[i]], label=f'ID {label}')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f't-SNE visualization of {title_prefix} features')
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{title_prefix.lower().replace(' ', '_')}_features_tsne.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        else:
            plt.show()
            
    except ImportError:
        print("scikit-learn not installed. Skipping feature visualization.")
    except Exception as e:
        print(f"Error visualizing features: {e}")


def main():
    args = parse_args()
    
    # Set save prefix based on module type if not provided
    if args.save_prefix is None:
        if args.module_type == 'spatial':
            args.save_prefix = 'spatial_attention'
        else:
            args.save_prefix = 'channel_attention'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading MOT16 sequences...")
    train_sequences = []
    for seq_name in args.seq_names:
        train_sequences.append(MOT16Sequences(seq_name, args.data_dir, load_seg=False)[0])
    
    val_sequences = []
    for seq_name in args.val_seq_names:
        val_sequences.append(MOT16Sequences(seq_name, args.data_dir, load_seg=False)[0])
    
    # Create datasets
    train_dataset = MOT16ReIDDataset(train_sequences)
    val_dataset = MOT16ReIDDataset(val_sequences) if val_sequences else None
    
    # Print dataset statistics before limiting
    print(f"Original training dataset size: {len(train_dataset)}")
    
    # Analyze person IDs in the dataset
    person_ids = [train_dataset[i][1] for i in range(len(train_dataset))]
    unique_ids = set(person_ids)
    id_counts = {}
    for id in person_ids:
        if id not in id_counts:
            id_counts[id] = 0
        id_counts[id] += 1
    
    print(f"Total unique person IDs: {len(unique_ids)}")
    print(f"Person ID counts: {id_counts}")
    
    # Find IDs with enough samples for triplet loss
    valid_ids = [id for id, count in id_counts.items() if count >= 2]
    print(f"IDs with at least 2 samples (valid for triplet loss): {len(valid_ids)}")
    
    if len(valid_ids) < 2:
        print("ERROR: Not enough valid IDs for triplet loss training!")
        print("Need at least 2 IDs with 2+ samples each.")
        print("Consider using more data or different sequences.")
        return
    
    # Ensure we have enough samples per ID for triplet loss
    min_samples_per_id = 2  # Need at least 2 samples of the same ID for positive pair
    
    # Filter dataset to only include IDs with enough samples
    filtered_indices = [i for i, (_, person_id) in enumerate(train_dataset) 
                       if id_counts.get(person_id, 0) >= min_samples_per_id]
    
    if len(filtered_indices) < 3:  # Need at least 3 samples total for triplet
        print("ERROR: Not enough valid samples for triplet loss training!")
        return
    
    print(f"Filtered training samples: {len(filtered_indices)}/{len(train_dataset)}")
    
    # Create filtered dataset
    train_dataset = torch.utils.data.Subset(train_dataset, filtered_indices)
    
    # Limit samples if specified
    if args.max_frames and len(train_dataset) > args.max_frames:
        indices = np.random.choice(len(train_dataset), args.max_frames, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    
    if val_dataset:
        # Do the same filtering for validation dataset
        val_person_ids = [val_dataset[i][1] for i in range(len(val_dataset))]
        val_id_counts = {}
        for id in val_person_ids:
            if id not in val_id_counts:
                val_id_counts[id] = 0
            val_id_counts[id] += 1
        
        val_filtered_indices = [i for i, (_, person_id) in enumerate(val_dataset) 
                             if val_id_counts.get(person_id, 0) >= min_samples_per_id]
        
        if len(val_filtered_indices) >= 3:
            val_dataset = torch.utils.data.Subset(val_dataset, val_filtered_indices)
            print(f"Filtered validation samples: {len(val_filtered_indices)}/{len(val_person_ids)}")
        else:
            print("Not enough valid samples in validation set, using training set for validation")
            val_dataset = train_dataset
        
        if args.max_frames and len(val_dataset) > args.max_frames // 5:
            indices = np.random.choice(len(val_dataset), args.max_frames // 5, replace=False)
            val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    print(f"Final training dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Final validation dataset size: {len(val_dataset)}")
    
    # Ensure batch size is appropriate
    if args.batch_size > len(train_dataset):
        old_batch_size = args.batch_size
        args.batch_size = max(3, len(train_dataset) // 2)  # Ensure at least 3 samples per batch
        print(f"WARNING: Batch size ({old_batch_size}) is larger than dataset size ({len(train_dataset)})")
        print(f"Reducing batch size to {args.batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True  # Ensure we have complete batches for triplet loss
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True  # Ensure we have complete batches for triplet loss
    ) if val_dataset else None
    
    # Create model based on module type
    if args.module_type == 'spatial':
        print("Creating ModifiedSpatialAttentionModule...")
        model = ModifiedSpatialAttentionModule(
            in_channels=args.in_channels,
            feature_dim=args.feature_dim
        )
        visualization_title = "Spatial Attention ReID"
    else:  # channel
        print("Creating ChannelAttentionReID module...")
        model = ChannelAttentionReID(
            in_channels=args.in_channels,
            feature_dim=args.feature_dim,
            reduction_ratio=args.reduction_ratio
        )
        visualization_title = "Channel Attention ReID"
    
    model.to(device)
    
    # Print model architecture for debugging
    print(f"\nModel Architecture:")
    print(model)
    print("\n")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")
    
    # Create loss function
    criterion = TripletLoss(margin=args.margin)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Define checkpoint paths
    interrupted_checkpoint_path = os.path.join(args.output_dir, f"{args.save_prefix}_interrupted.pth")
    
    # Initialize training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Try to load checkpoint if one exists
    start_epoch = 0
    if args.resume:
        checkpoint_path = args.resume
        if os.path.exists(checkpoint_path):
            start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
            start_epoch += 1  # Start from next epoch
    elif os.path.exists(interrupted_checkpoint_path):
        try:
            start_epoch, best_val_loss = load_checkpoint(interrupted_checkpoint_path, model, optimizer, scheduler, device)
            start_epoch += 1  # Start from next epoch
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            start_epoch = 0
    
    # Setup signal handlers for graceful interruption
    setup_signal_handler(model, optimizer, scheduler, train_losses, val_losses, best_val_loss, start_epoch, args)
    
    print(f"Starting training from epoch {start_epoch+1}")
    
    # Check if we have any samples in the first batch
    if len(train_loader) > 0:
        try:
            first_batch = next(iter(train_loader))
            print(f"First batch shape: crops={first_batch[0].shape}, ids={first_batch[1].shape}")
            print(f"Number of unique IDs in first batch: {len(torch.unique(first_batch[1]))}")
            
            # Check if batch has enough unique IDs
            if len(torch.unique(first_batch[1])) < 2:
                print("WARNING: First batch has fewer than 2 unique IDs, triplet loss may not work properly")
                print("Consider increasing batch size or using more diverse data")
        except Exception as e:
            print(f"Error checking first batch: {e}")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device) if val_loader else train_loss
        val_losses.append(val_loss)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"{args.save_prefix}_epoch_{epoch+1}.pth")
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"New best model with validation loss {best_val_loss:.4f}!")
        
        save_checkpoint(
            epoch, model, optimizer, scheduler, 
            train_loss, val_loss, best_val_loss, 
            checkpoint_path, is_best, args.save_prefix
        )
        
        # Save training history
        history_path = os.path.join(args.output_dir, f"{args.save_prefix}_training_history.json")
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': list(range(start_epoch, epoch + 1)),
                'best_val_loss': best_val_loss,
                'current_epoch': epoch
            }, f)
        
        print(f"Epoch {epoch+1} complete. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Visualize features
    if val_loader:
        print("Visualizing ReID features...")
        visualize_features(
            model, val_loader, device, num_samples=100, 
            output_dir=os.path.join(args.output_dir, "visualizations"),
            title_prefix=visualization_title
        )
    
    # Plot training history
    try:
        plt.figure(figsize=(10, 5))
        epochs = list(range(start_epoch, start_epoch + len(train_losses)))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title(f'{visualization_title} Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(args.output_dir, f"{args.save_prefix}_training_plot.png")
        plt.savefig(plot_path)
        print(f"Training history plot saved to {plot_path}")
    except Exception as plot_error:
        print(f"Error creating training plot: {plot_error}")
    
    # Save final model in a format compatible with the hybrid model
    final_model_path = os.path.join(args.output_dir, f"{args.save_prefix}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("\nTraining Summary:")
    print(f"Module Type: {args.module_type}")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Visualization saved to: {os.path.join(args.output_dir, 'visualizations')}")
    
    return model


if __name__ == "__main__":
    main() 