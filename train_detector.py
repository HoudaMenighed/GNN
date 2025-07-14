import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import time
import numpy as np
from tqdm import tqdm
import signal
import sys
import json
from torch.optim.lr_scheduler import StepLR
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from track.data_track import MOT16Sequences


class MOT16DetectionDataset(Dataset):
    """Dataset wrapper for MOT16 to format it for object detection training"""
    
    def __init__(self, sequences, transform=None):
        """
        Args:
            sequences: List of MOT16Sequence objects
            transform: Optional transform to be applied on a sample
        """
        self.transform = transform
        self.data = []
        
        # Collect all frames from all sequences
        for seq in sequences:
            for frame_data in seq:
                self.data.append(frame_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frame_data = self.data[idx]
        
        # Get image
        img = frame_data['img']  # Already a tensor from MOT16Sequence
        
        # Get ground truth boxes and create target dict
        gt_boxes_dict = frame_data['gt']
        
        # Convert ground truth dict to lists
        boxes = []
        labels = []  # In MOT16 all objects are persons (class 1)
        
        for gt_id, box in gt_boxes_dict.items():
            boxes.append(box)
            labels.append(1)  # Person class
        
        # If no boxes, create empty tensors
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            # Convert list to numpy array first, then to tensor (more efficient)
            boxes = np.array(boxes, dtype=np.float32)
            boxes = torch.from_numpy(boxes)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        # Create target dict in format expected by Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms if any
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Faster R-CNN on MOT16 dataset')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seq_names', type=str, nargs='+', default=['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13'], 
                        help='Sequence names for training')
    parser.add_argument('--val_seq_names', type=str, nargs='+', default=[], 
                        help='Sequence names for validation')
    parser.add_argument('--device', type=str, default='', help='cuda or cpu')
    parser.add_argument('--lr_step_size', type=int, default=3, help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for batches')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to use (for quick testing)')
    parser.add_argument('--save_prefix', type=str, default='fasterrcnn', help='Prefix for saved model files')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (including background)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone layers')
    return parser.parse_args()


def collate_fn(batch):
    """
    Custom collate function for detection data
    """
    return tuple(zip(*batch))


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    """Train the model for one epoch"""
    model.train()
    
    total_loss = 0
    batch_times = []
    successful_batches = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, (images, targets) in pbar:
        try:
            start_time = time.time()
            
            # Debug info
            if batch_idx % args.log_interval == 0:
                print(f"\nTraining batch {batch_idx}:")
                for i, target in enumerate(targets):
                    print(f"  Image {i}: {len(target['boxes'])} boxes, shape: {images[i].shape}")
            
            # Skip batches with no boxes (can cause training issues)
            empty_targets = sum(1 for t in targets if len(t['boxes']) == 0)
            if empty_targets > 0:
                print(f"  Skipping batch {batch_idx} with {empty_targets} empty targets")
                continue
            
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate losses
            if isinstance(loss_dict, list):
                # Handle case where loss_dict is a list
                losses = sum(sum(d.values()) for d in loss_dict if isinstance(d, dict))
            else:
                # Traditional dictionary case
                losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            # Calculate batch time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Update statistics
            total_loss += losses.item()
            successful_batches += 1
            
            # Update progress bar
            if batch_idx % args.log_interval == 0:
                avg_time = np.mean(batch_times[-10:]) if batch_times else 0
                avg_loss = total_loss / max(successful_batches, 1)
                
                # Print losses from loss_dict
                if isinstance(loss_dict, dict):
                    loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                else:
                    loss_str = "Loss details not available"
                
                pbar.set_description(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(data_loader)} | "
                    f"Loss: {avg_loss:.4f} | {loss_str} | "
                    f"Time: {avg_time:.3f}s/batch"
                )
        
        except Exception as e:
            print(f"\nError in training batch {batch_idx}: {e}")
            print("Details:")
            for i, (image, target) in enumerate(zip(images, targets)):
                try:
                    print(f"  Image {i} shape: {image.shape}")
                    print(f"  Target {i} boxes: {target['boxes'].shape if 'boxes' in target else 'No boxes'}")
                except Exception as e2:
                    print(f"  Error getting details: {e2}")
            continue
    
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
    else:
        avg_loss = float('inf')
        print("Warning: All training batches failed!")
    
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f} (successful batches: {successful_batches}/{len(data_loader)})")
    
    return avg_loss


def validate(model, data_loader, device):
    """Validate the model using inference mode instead of loss computation"""
    model.eval()
    
    # Use mean average precision as validation metric
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Validating")):
            try:
                print(f"\nValidation batch {batch_idx}:")
                
                # Move images to device (but keep targets on CPU for evaluation)
                images = [img.to(device) for img in images]
                
                # Run inference
                predictions = model(images)
                
                # Store predictions and targets for evaluation
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
                # Print some information about the predictions
                for i, pred in enumerate(predictions):
                    num_boxes = len(pred['boxes'])
                    print(f"  Image {i}: {num_boxes} predicted boxes")
                    
                    # Print first few predictions
                    if num_boxes > 0:
                        scores = pred['scores'].cpu().numpy()
                        boxes = pred['boxes'].cpu().numpy()
                        for j in range(min(3, num_boxes)):
                            x1, y1, x2, y2 = boxes[j]
                            print(f"    Box {j}: score={scores[j]:.4f}, coords=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                
            except Exception as e:
                print(f"\nError during validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Simple evaluation: average number of detections with score > 0.5
    total_detections = 0
    total_high_conf_detections = 0
    
    for pred in all_predictions:
        if len(pred['boxes']) > 0:
            total_detections += len(pred['boxes'])
            high_conf = (pred['scores'] > 0.5).sum().item()
            total_high_conf_detections += high_conf
    
    # Use a simple metric: average number of high confidence detections per image
    if len(all_predictions) > 0:
        avg_detections = total_detections / len(all_predictions)
        avg_high_conf = total_high_conf_detections / len(all_predictions)
        print(f"Average detections per image: {avg_detections:.2f}")
        print(f"Average high confidence detections per image: {avg_high_conf:.2f}")
        
        # Return negative avg_high_conf as "loss" (lower is better)
        val_loss = -avg_high_conf
    else:
        print("No valid predictions during validation")
        val_loss = float('inf')
    
    print(f"Validation metric: {val_loss:.4f}")
    
    return val_loss


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, best_val_loss, checkpoint_path, is_best=False, save_prefix='fasterrcnn'):
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


def visualize_predictions(model, dataset, device, num_images=5, output_dir=None):
    """Visualize model predictions on sample images"""
    model.eval()
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Sample random images
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    
    for idx in indices:
        # Get image and target
        img, target = dataset[idx]
        
        # Move to device
        img_tensor = img.to(device)
        
        # Get prediction
        with torch.no_grad():
            prediction = model([img_tensor])[0]
        
        # Convert image tensor to numpy for visualization
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Normalize image for display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot ground truth
        ax[0].imshow(img_np)
        ax[0].set_title('Ground Truth')
        for box in target['boxes'].cpu().numpy():
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
            ax[0].add_patch(rect)
        
        # Plot predictions
        ax[1].imshow(img_np)
        ax[1].set_title('Predictions')
        
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        
        # Filter by score threshold
        mask = scores > 0.5
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
            ax[1].text(x1, y1, f"{score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        # Save or show
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"pred_{idx}.png"))
            plt.close()
        else:
            plt.show()


def main():
    args = parse_args()
    
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
    train_dataset = MOT16DetectionDataset(train_sequences)
    val_dataset = MOT16DetectionDataset(val_sequences) if val_sequences else None
    
    # Limit frames if specified
    if args.max_frames:
        train_indices = np.random.choice(len(train_dataset), min(args.max_frames, len(train_dataset)), replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        if val_dataset:
            val_frames = max(1, int(args.max_frames * 0.2))
            val_indices = np.random.choice(len(val_dataset), min(val_frames, len(val_dataset)), replace=False)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    print(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    ) if val_dataset else None
    
    # Create model
    print("Creating Faster R-CNN model...")
    model = fasterrcnn_resnet50_fpn(pretrained=args.pretrained)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args.num_classes)
    
    # Freeze backbone layers if specified
    if args.freeze_backbone:
        print("Freezing backbone layers...")
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
    
    model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")
    
    # Create optimizer
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005
    )
    
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
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device) if val_loader else train_loss
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
    
    # Visualize predictions on validation set
    if val_dataset:
        print("Visualizing predictions...")
        visualize_predictions(model, val_dataset, device, num_images=5, 
                             output_dir=os.path.join(args.output_dir, "visualizations"))
    
    # Plot training history
    try:
        plt.figure(figsize=(10, 5))
        epochs = list(range(start_epoch, start_epoch + len(train_losses)))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
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


if __name__ == "__main__":
    main() 