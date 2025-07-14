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

from track.joint_model_hybrid import GraphNN, GCNConv
from track.data_track import MOT16Sequences
from track.utils import create_edge_index, create_temporal_edge_index, create_comprehensive_edge_index

class MOT16GraphDataset(Dataset):
    """Dataset wrapper for MOT16 to extract graph data for GNN training"""
    
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
        Returns a temporal window of frames for GNN training
        
        Each sample contains:
        - List of frames in the window
        - Each frame has 'img', 'gt' (ground truth boxes and IDs)
        """
        return self.data[idx]


def extract_features_from_frames(frames, device):
    """
    Extract features and bounding boxes from frames
    
    Args:
        frames: List of frames from the dataset
        device: Device to use for computation
        
    Returns:
        all_features: List of feature tensors for each frame
        all_boxes: List of bounding box tensors for each frame
        all_gt_ids: List of ground truth IDs for each frame
    """
    all_features = []
    all_boxes = []
    all_gt_ids = []
    
    for frame in frames:
        # Extract ground truth boxes and IDs
        gt_boxes_dict = frame['gt']
        gt_boxes = []
        gt_ids = []
        
        # Convert ground truth dict to lists
        for gt_id, box in gt_boxes_dict.items():
            gt_boxes.append(box)
            gt_ids.append(gt_id)
        
        # Convert to tensors
        if gt_boxes:
            gt_boxes = torch.tensor(gt_boxes, device=device)
            # Generate random features for training (in real scenario, these would come from a ReID network)
            # Using 128 dimensions to match the GraphNN feature_projection input
            features = torch.randn(len(gt_boxes), 128, device=device)
            features = F.normalize(features, p=2, dim=1)  # Normalize features
        else:
            gt_boxes = torch.zeros((0, 4), device=device)
            features = torch.zeros((0, 128), device=device)
        
        all_features.append(features)
        all_boxes.append(gt_boxes)
        all_gt_ids.append(gt_ids)
    
    return all_features, all_boxes, all_gt_ids


def create_training_graph(prev_features, curr_features, prev_boxes, curr_boxes, device):
    """
    Create a training graph from features and boxes
    
    Args:
        prev_features: Features from previous frame
        curr_features: Features from current frame
        prev_boxes: Boxes from previous frame
        curr_boxes: Boxes from current frame
        device: Device to use for computation
        
    Returns:
        combined_features: Combined features from both frames
        edge_index: Edge indices for the graph
        edge_targets: Target values for each edge (1.0 for temporal edges, 0.0 otherwise)
    """
    # Create comprehensive edge index
    edge_index = create_comprehensive_edge_index(
        prev_features=prev_features,
        curr_features=curr_features,
        prev_boxes=prev_boxes,
        curr_boxes=curr_boxes,
        similarity_threshold=0.5,
        spatial_threshold=0.5,
        device=device
    )
    
    # If no edges, return empty tensors
    if edge_index.shape[1] == 0:
        combined_features = torch.cat([prev_features, curr_features], dim=0)
        return combined_features, edge_index, torch.zeros(0, device=device)
    
    # Create edge targets (1.0 for temporal edges, 0.0 otherwise)
    num_prev = prev_features.shape[0]
    edge_targets = torch.zeros(edge_index.shape[1], device=device)
    
    # Identify temporal edges (connections between frames)
    src, dst = edge_index
    is_temporal_edge = (src < num_prev) & (dst >= num_prev)
    edge_targets[is_temporal_edge] = 1.0
    
    # Combine features
    combined_features = torch.cat([prev_features, curr_features], dim=0)
    
    return combined_features, edge_index, edge_targets


def parse_args():
    parser = argparse.ArgumentParser(description='Train GNN for tracking')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/gnn', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seq_names', type=str, nargs='+', default=['MOT16-02', 'MOT16-04', 'MOT16-05'], 
                        help='Sequence names for training')
    parser.add_argument('--val_seq_names', type=str, nargs='+', default=['MOT16-09'], 
                        help='Sequence names for validation')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--device', type=str, default='', help='cuda or cpu')
    parser.add_argument('--lr_step_size', type=int, default=7, help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for batches')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to use (for quick testing)')
    parser.add_argument('--save_prefix', type=str, default='gnn', help='Prefix for saved model files')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension for GNN')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for GNN')
    parser.add_argument('--temporal_window', type=int, default=2, help='Number of consecutive frames to use for each sample')
    return parser.parse_args()


def train_one_epoch(model, data_loader, optimizer, device, epoch, args):
    """Train the model for one epoch"""
    model.train()
    
    total_loss = 0
    total_edge_loss = 0
    total_edges = 0
    batch_times = []
    successful_batches = 0
    failed_batches = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, frames in pbar:
        try:
            start_time = time.time()
            
            # Extract features and boxes from frames
            all_features, all_boxes, all_gt_ids = extract_features_from_frames(frames, device)
            
            # Process pairs of consecutive frames
            for i in range(len(all_features) - 1):
                prev_features = all_features[i]
                curr_features = all_features[i+1]
                prev_boxes = all_boxes[i]
                curr_boxes = all_boxes[i+1]
                
                # Skip if either frame has no objects
                if prev_features.shape[0] == 0 or curr_features.shape[0] == 0:
                    continue
                
                # Create training graph
                combined_features, edge_index, edge_targets = create_training_graph(
                    prev_features, curr_features, prev_boxes, curr_boxes, device
                )
                
                # Skip if no edges
                if edge_index.shape[1] == 0:
                    continue
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                node_embeddings, edge_weights, _ = model(combined_features, edge_index)
                
                # Compute loss
                edge_loss = nn.BCELoss()(edge_weights, edge_targets)
                
                # Backward and optimize
                edge_loss.backward()
                optimizer.step()
                
                # Update statistics
                loss_value = edge_loss.item()
                total_loss += loss_value
                total_edge_loss += loss_value
                total_edges += edge_index.shape[1]
                successful_batches += 1
            
            # Calculate batch time
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            if batch_idx % args.log_interval == 0:
                avg_time = np.mean(batch_times[-10:]) if batch_times else 0
                avg_loss = total_loss / max(successful_batches, 1)
                avg_edge_loss = total_edge_loss / max(total_edges, 1)
                
                pbar.set_description(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(data_loader)} | "
                    f"Loss: {avg_loss:.4f} | Edge Loss: {avg_edge_loss:.4f} | "
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
        avg_edge_loss = total_edge_loss / max(total_edges, 1)
    else:
        avg_loss = float('inf')
        avg_edge_loss = float('inf')
        print("Warning: All training batches failed!")
    
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}, Edge Loss: {avg_edge_loss:.4f} "
          f"(successful batches: {successful_batches}/{len(data_loader)}, failed: {failed_batches})")
    
    return avg_loss


def validate(model, data_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_edge_loss = 0
    total_edges = 0
    successful_batches = 0
    failed_batches = 0
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(tqdm(data_loader, desc="Validating")):
            try:
                # Extract features and boxes from frames
                all_features, all_boxes, all_gt_ids = extract_features_from_frames(frames, device)
                
                # Process pairs of consecutive frames
                for i in range(len(all_features) - 1):
                    prev_features = all_features[i]
                    curr_features = all_features[i+1]
                    prev_boxes = all_boxes[i]
                    curr_boxes = all_boxes[i+1]
                    
                    # Skip if either frame has no objects
                    if prev_features.shape[0] == 0 or curr_features.shape[0] == 0:
                        continue
                    
                    # Create training graph
                    combined_features, edge_index, edge_targets = create_training_graph(
                        prev_features, curr_features, prev_boxes, curr_boxes, device
                    )
                    
                    # Skip if no edges
                    if edge_index.shape[1] == 0:
                        continue
                    
                    # Forward pass
                    node_embeddings, edge_weights, _ = model(combined_features, edge_index)
                    
                    # Compute loss
                    edge_loss = nn.BCELoss()(edge_weights, edge_targets)
                    
                    # Update statistics
                    loss_value = edge_loss.item()
                    total_loss += loss_value
                    total_edge_loss += loss_value
                    total_edges += edge_index.shape[1]
                    successful_batches += 1
                
            except Exception as e:
                print(f"\nError during validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                failed_batches += 1
                continue
    
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
        avg_edge_loss = total_edge_loss / max(total_edges, 1)
    else:
        avg_loss = float('inf')
        avg_edge_loss = float('inf')
        print("Warning: All validation batches failed!")
    
    print(f"Validation Loss: {avg_loss:.4f}, Edge Loss: {avg_edge_loss:.4f} "
          f"(successful batches: {successful_batches}/{len(data_loader)}, failed: {failed_batches})")
    
    return avg_loss


def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, val_loss, best_val_loss, checkpoint_path, is_best=False, save_prefix='gnn'):
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
    train_dataset = MOT16GraphDataset(train_sequences, max_frames=args.max_frames, temporal_window=args.temporal_window)
    val_dataset = MOT16GraphDataset(val_sequences, max_frames=args.max_frames // 5 if args.max_frames else None, temporal_window=args.temporal_window)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0]  # Each batch is a single temporal window
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0]  # Each batch is a single temporal window
    )
    
    # Create model
    print("Creating GNN model...")
    model = GraphNN(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_gnn_layers
    )
    model.to(device)
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
    print("\n")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})")
    
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
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
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
        plt.title('GNN Training and Validation Loss')
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
    print(f"GNN Layers: {args.num_gnn_layers}")
    print(f"Feature Dimension: {args.feature_dim}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Total Epochs: {args.num_epochs}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_model_path}")
    
    return model


if __name__ == "__main__":
    main() 