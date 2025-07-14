import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
from torchvision.ops import roi_align

from track.joint_model_hybrid import HybridDetectionTrackingModel
from track.data_track import MOT16Sequences
from track.utils import (
    create_edge_index, calculate_iou, match_detections_with_gt, 
    tensor_to_python, collate_fn
)

class MOT16AssociationDataset(Dataset):
    """Dataset wrapper for MOT16 to extract association data"""
    
    def __init__(self, sequences, max_frames=None, temporal_window=2):
        """
        Args:
            sequences: List of MOT16Sequence objects
            max_frames: Maximum number of frames to use (for quick testing)
            temporal_window: Number of consecutive frames to use for each sample
        """
        self.temporal_window = temporal_window
        self.data = []
        
        # Collect all frames from all sequences
        for seq in sequences:
            seq_data = []
            # Limit frames if specified
            num_frames = len(seq)
            if max_frames:
                num_frames = min(len(seq), max_frames)
            
            # Add frames to sequence data
            for i in range(num_frames):
                seq_data.append(seq[i])
            
            # Create temporal windows
            for i in range(len(seq_data) - temporal_window + 1):
                window = seq_data[i:i+temporal_window]
                self.data.append(window)
        
        print(f"Created dataset with {len(self.data)} temporal windows")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns a temporal window of frames for association training
        """
        return self.data[idx]


def extract_features_and_boxes(model, frames, device, use_random_features=False):
    """
    Extract features and bounding boxes from frames using the model
    
    Args:
        model: HybridDetectionTrackingModel
        frames: List of frames from the dataset
        device: Device to use for computation
        use_random_features: If True, use random features instead of extracting from model
        
    Returns:
        all_features: List of feature tensors for each frame
        all_boxes: List of bounding box tensors for each frame
        all_gt_ids: List of ground truth IDs for each frame
    """
    all_features = []
    all_boxes = []
    all_gt_ids = []
    
    for frame in frames:
        try:
            # Get image tensor
            img = frame['img'].to(device)
            print(f"Processing image with shape: {img.shape}")
            
            # Extract ground truth boxes and IDs
            gt_boxes_dict = frame['gt']
            gt_boxes = []
            gt_ids = []
            
            # Convert ground truth dict to lists
            for gt_id, box in gt_boxes_dict.items():
                gt_boxes.append(box)
                gt_ids.append(gt_id)
            
            print(f"Found {len(gt_boxes)} ground truth boxes")
            
            # Convert to tensors
            if gt_boxes:
                gt_boxes = torch.tensor(gt_boxes, device=device)
                
                if use_random_features:
                    # Use random features for testing
                    print("Using random features as requested...")
                    features = torch.randn(len(gt_boxes), model.feature_dim, device=device)
                    features = F.normalize(features, p=2, dim=1)
                else:
                    # Use the detector to extract features
                    with torch.no_grad():
                        try:
                            # Process image through detector
                            print("Running detector...")
                            detections = model.detector([img])
                            
                            # Check if extract_roi_features method exists
                            if not hasattr(model.detector, 'extract_roi_features'):
                                print("Warning: detector doesn't have extract_roi_features method. Adding it...")
                                # Add method if it doesn't exist
                                def extract_roi_features(self, img, boxes):
                                    """Extract ROI features from image using boxes"""
                                    # Get the backbone features
                                    features = self.detector.backbone(img.unsqueeze(0))
                                    
                                    # Use ROI align to extract features for each box
                                    if isinstance(features, torch.Tensor):
                                        rois = roi_align(features, [boxes], output_size=(7, 7))
                                    else:
                                        # For FPN, use the last feature map
                                        rois = roi_align(features[list(features.keys())[-1]], [boxes], output_size=(7, 7))
                                    
                                    # Flatten the features
                                    roi_features = torch.flatten(rois, start_dim=1)
                                    return roi_features
                                
                                # Add the method to the model
                                import types
                                model.detector.extract_roi_features = types.MethodType(extract_roi_features, model.detector)
                            
                            # Check if extract_reid_features method exists
                            if not hasattr(model.detector, 'extract_reid_features'):
                                print("Warning: detector doesn't have extract_reid_features method. Adding it...")
                                # Add method if it doesn't exist
                                def extract_reid_features(self, roi_features):
                                    """Extract ReID features from ROI features"""
                                    features = self.feature_projection(roi_features)
                                    reid_features = self.reid_head(features)
                                    return reid_features
                                
                                # Add the method to the model
                                import types
                                model.detector.extract_reid_features = types.MethodType(extract_reid_features, model.detector)
                            
                            # Extract ReID features for ground truth boxes
                            print("Extracting ROI features...")
                            roi_features = model.detector.extract_roi_features(img, gt_boxes)
                            print(f"ROI features shape: {roi_features.shape}")
                            
                            print("Extracting ReID features...")
                            reid_features = model.detector.extract_reid_features(roi_features)
                            print(f"ReID features shape: {reid_features.shape}")
                            
                            # Apply spatial attention if available
                            if hasattr(model, 'enhanced_reid'):
                                print("Applying spatial attention...")
                                enhanced_features = model.enhanced_reid(roi_features, reid_features)
                                features = F.normalize(enhanced_features, p=2, dim=1)
                            else:
                                features = F.normalize(reid_features, p=2, dim=1)
                            
                            print(f"Final features shape: {features.shape}")
                        except Exception as e:
                            print(f"Error in feature extraction: {e}")
                            import traceback
                            traceback.print_exc()
                            # Use random features as fallback
                            print("Using random features as fallback...")
                            features = torch.randn(len(gt_boxes), model.feature_dim, device=device)
                            features = F.normalize(features, p=2, dim=1)
            else:
                gt_boxes = torch.zeros((0, 4), device=device)
                features = torch.zeros((0, model.feature_dim), device=device)
            
            all_features.append(features)
            all_boxes.append(gt_boxes)
            all_gt_ids.append(gt_ids)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            # Add empty tensors for this frame
            all_features.append(torch.zeros((0, model.feature_dim), device=device))
            all_boxes.append(torch.zeros((0, 4), device=device))
            all_gt_ids.append([])
    
    return all_features, all_boxes, all_gt_ids


def create_association_pairs(prev_features, curr_features, prev_boxes, curr_boxes, 
                            prev_ids, curr_ids, model, device):
    """
    Create training pairs for association learning
    
    Args:
        prev_features: Features from previous frame
        curr_features: Features from current frame
        prev_boxes: Boxes from previous frame
        curr_boxes: Boxes from current frame
        prev_ids: IDs from previous frame
        curr_ids: IDs from current frame
        model: HybridDetectionTrackingModel
        device: Device to use for computation
        
    Returns:
        pair_features: Tensor of pair features for association
        pair_labels: Tensor of binary labels (1 for same ID, 0 for different ID)
    """
    pair_features = []
    pair_labels = []
    
    # Skip if either frame has no objects
    if prev_features.shape[0] == 0 or curr_features.shape[0] == 0:
        return torch.zeros((0, 512), device=device), torch.zeros(0, device=device)
    
    # Project features if needed
    if prev_features.size(1) != 512:
        prev_features_proj = model.assoc_projection(prev_features)
    else:
        prev_features_proj = prev_features
        
    if curr_features.size(1) != 512:
        curr_features_proj = model.assoc_projection(curr_features)
    else:
        curr_features_proj = curr_features
    
    # Create pairs of features
    for i in range(len(prev_ids)):
        for j in range(len(curr_ids)):
            # Concatenate features
            pair_feature = torch.cat([prev_features_proj[i], curr_features_proj[j]], dim=0)
            
            # Project concatenated features if needed
            if pair_feature.size(0) == 1024:
                pair_feature = model.pair_feature_projection(pair_feature.unsqueeze(0)).squeeze(0)
            
            # Determine label based on ID match
            label = 1.0 if prev_ids[i] == curr_ids[j] else 0.0
            
            # If no ID info, use IoU as a proxy
            if prev_ids[i] == -1 or curr_ids[j] == -1:
                iou = calculate_iou(prev_boxes[i].cpu().numpy(), curr_boxes[j].cpu().numpy())
                if iou > 0.5:
                    label = 0.8  # High IoU suggests same object
                elif iou > 0.3:
                    label = 0.5  # Medium IoU
                else:
                    label = 0.0  # Low IoU suggests different objects
            
            pair_features.append(pair_feature)
            pair_labels.append(torch.tensor(label, device=device))
    
    if pair_features:
        pair_features = torch.stack(pair_features)
        pair_labels = torch.stack(pair_labels)
    else:
        pair_features = torch.zeros((0, 512), device=device)
        pair_labels = torch.zeros(0, device=device)
    
    return pair_features, pair_labels


def parse_args():
    parser = argparse.ArgumentParser(description='Train Association Module for Tracking')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/association', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seq_names', type=str, nargs='+', default=['MOT16-02', 'MOT16-04', 'MOT16-05'], 
                        help='Sequence names for training')
    parser.add_argument('--val_seq_names', type=str, nargs='+', default=['MOT16-09'], 
                        help='Sequence names for validation')
    parser.add_argument('--device', type=str, default='', help='cuda or cpu')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for batches')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to use (for quick testing)')
    parser.add_argument('--save_prefix', type=str, default='association', help='Prefix for saved model files')
    parser.add_argument('--hybrid_model_path', type=str, default='./checkpoints/hybrid/hybrid_best.pth', 
                        help='Path to pre-trained hybrid model')
    parser.add_argument('--temporal_window', type=int, default=2, help='Number of consecutive frames to use for each sample')
    parser.add_argument('--use_random_features', action='store_true', help='Use random features for testing')
    return parser.parse_args()


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    """Train the model for one epoch"""
    model.train()
    
    total_loss = 0
    total_pairs = 0
    batch_times = []
    successful_batches = 0
    failed_batches = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, frames in pbar:
        try:
            start_time = time.time()
            
            # Extract features and boxes from frames
            all_features, all_boxes, all_gt_ids = extract_features_and_boxes(model, frames, device, args.use_random_features)
            
            # Process pairs of consecutive frames
            for i in range(len(all_features) - 1):
                prev_features = all_features[i]
                curr_features = all_features[i+1]
                prev_boxes = all_boxes[i]
                curr_boxes = all_boxes[i+1]
                prev_ids = all_gt_ids[i]
                curr_ids = all_gt_ids[i+1]
                
                # Skip if either frame has no objects
                if prev_features.shape[0] == 0 or curr_features.shape[0] == 0:
                    continue
                
                # Create association pairs
                pair_features, pair_labels = create_association_pairs(
                    prev_features, curr_features, prev_boxes, curr_boxes, 
                    prev_ids, curr_ids, model, device
                )
                
                # Skip if no valid pairs
                if pair_features.shape[0] == 0:
                    continue
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass through association head
                association_scores = model.association_head(pair_features).squeeze()
                
                # Compute loss
                loss = nn.BCELoss()(association_scores, pair_labels)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                loss_value = loss.item()
                total_loss += loss_value
                total_pairs += pair_features.shape[0]
                successful_batches += 1
            
            # Calculate batch time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            if batch_idx % args.log_interval == 0:
                avg_time = np.mean(batch_times[-10:]) if batch_times else 0
                avg_loss = total_loss / max(successful_batches, 1)
                
                pbar.set_description(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(data_loader)} | "
                    f"Loss: {avg_loss:.4f} | Pairs: {total_pairs} | "
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
    
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f} "
          f"(successful batches: {successful_batches}/{len(data_loader)}, failed: {failed_batches})")
    
    return avg_loss


def validate(model, data_loader, device, use_random_features=False):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_pairs = 0
    successful_batches = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(tqdm(data_loader, desc="Validating")):
            try:
                # Extract features and boxes from frames
                all_features, all_boxes, all_gt_ids = extract_features_and_boxes(model, frames, device, use_random_features)
                
                # Process pairs of consecutive frames
                for i in range(len(all_features) - 1):
                    prev_features = all_features[i]
                    curr_features = all_features[i+1]
                    prev_boxes = all_boxes[i]
                    curr_boxes = all_boxes[i+1]
                    prev_ids = all_gt_ids[i]
                    curr_ids = all_gt_ids[i+1]
                    
                    # Skip if either frame has no objects
                    if prev_features.shape[0] == 0 or curr_features.shape[0] == 0:
                        continue
                    
                    # Create association pairs
                    pair_features, pair_labels = create_association_pairs(
                        prev_features, curr_features, prev_boxes, curr_boxes, 
                        prev_ids, curr_ids, model, device
                    )
                    
                    # Skip if no valid pairs
                    if pair_features.shape[0] == 0:
                        continue
                    
                    # Forward pass through association head
                    association_scores = model.association_head(pair_features).squeeze()
                    
                    # Compute loss
                    loss = nn.BCELoss()(association_scores, pair_labels)
                    
                    # Update statistics
                    loss_value = loss.item()
                    total_loss += loss_value
                    total_pairs += pair_features.shape[0]
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
    
    print(f"Validation Loss: {avg_loss:.4f} "
          f"(successful batches: {successful_batches}/{len(data_loader)}, failed: {failed_batches})")
    
    return avg_loss


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, best_val_loss, checkpoint_path, is_best=False, save_prefix='association'):
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
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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


def freeze_all_except_association(model):
    """Freeze all parameters except association head"""
    for name, param in model.named_parameters():
        if 'association_head' not in name and 'pair_feature_projection' not in name and 'assoc_projection' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load hybrid model
    print("Loading hybrid model...")
    model = HybridDetectionTrackingModel(
        pretrained=True,
        num_classes=91,  # COCO dataset classes
        feature_dim=512,
        reid_dim=256,
        hidden_dim=256,
        num_gnn_layers=2
    )
    
    # Load pre-trained weights if available
    if args.hybrid_model_path and os.path.exists(args.hybrid_model_path):
        print(f"Loading pre-trained weights from {args.hybrid_model_path}")
        checkpoint = torch.load(args.hybrid_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    
    # Freeze all layers except association head
    print("Freezing all layers except association head...")
    freeze_all_except_association(model)
    
    # Create optimizer - only optimize trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
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
    
    # Load datasets
    print("Loading MOT16 sequences...")
    train_sequences = []
    for seq_name in args.seq_names:
        train_sequences.append(MOT16Sequences(seq_name, args.data_dir, load_seg=True)[0])
    
    val_sequences = []
    for seq_name in args.val_seq_names:
        val_sequences.append(MOT16Sequences(seq_name, args.data_dir, load_seg=True)[0])
    
    # Create datasets
    train_dataset = MOT16AssociationDataset(train_sequences, max_frames=args.max_frames, temporal_window=args.temporal_window)
    val_dataset = MOT16AssociationDataset(val_sequences, max_frames=args.max_frames // 5 if args.max_frames else None, temporal_window=args.temporal_window)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Process frames individually for temporal processing
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0]  # Each batch is a single temporal window
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0]  # Each batch is a single temporal window
    )
    
    # Setup signal handlers for graceful interruption
    setup_signal_handler(model, optimizer, scheduler, train_losses, val_losses, best_val_loss, start_epoch, args)
    
    print(f"Starting training from epoch {start_epoch+1}")
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device, args.use_random_features)
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
    
    # Plot training history
    try:
        plt.figure(figsize=(10, 5))
        epochs = list(range(start_epoch, start_epoch + len(train_losses)))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Association Module Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(args.output_dir, f"{args.save_prefix}_training_plot.png")
        plt.savefig(plot_path)
        print(f"Training history plot saved to {plot_path}")
    except Exception as plot_error:
        print(f"Error creating training plot: {plot_error}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.save_prefix}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    print("\nTraining Summary:")
    print(f"Association Module Training")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")
    
    return model


if __name__ == "__main__":
    main() 