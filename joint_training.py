#!/usr/bin/env python3
"""
Joint Training Script for Multiple Object Tracking

This script implements the joint training algorithm for Detection, ReID, GNN, and Association modules
as described in the algorithm. It serves as an entry point for the joint training process.
"""

import os
import sys
import argparse
import torch

# Configure Python's stdout to be unbuffered for real-time logging
import functools
print = functools.partial(print, flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Joint Training of Detection, ReID, GNN, and Association Modules')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/joint', help='Output directory for models')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
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
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to use')
    parser.add_argument('--save_prefix', type=str, default='joint', help='Prefix for saved model files')
    parser.add_argument('--temporal_window', type=int, default=3, help='Number of consecutive frames to use')
    
    # Loss weights for the joint training algorithm
    parser.add_argument('--detector_weight', type=float, default=1.0, help='Weight for detector loss')
    parser.add_argument('--reid_weight', type=float, default=0.5, help='Weight for ReID loss')
    parser.add_argument('--gnn_weight', type=float, default=0.3, help='Weight for GNN loss')
    parser.add_argument('--association_weight', type=float, default=0.7, help='Weight for association loss')
    
    # Components to train
    parser.add_argument('--train_detector', action='store_true', help='Train the detector component')
    parser.add_argument('--train_reid', action='store_true', help='Train the ReID component')
    parser.add_argument('--train_gnn', action='store_true', help='Train the GNN component')
    parser.add_argument('--train_association', action='store_true', help='Train the association component')
    
    # Paths to pretrained models
    parser.add_argument('--detector_path', type=str, default='./checkpoints/detector/fasterrcnn_best.pth', 
                        help='Path to pretrained detector model')
    parser.add_argument('--reid_path', type=str, default='./checkpoints/reid/spatial_attention_best.pth',
                        help='Path to pretrained ReID model')
    parser.add_argument('--gnn_path', type=str, default='./checkpoints/gnn/gnn_best.pth',
                        help='Path to pretrained GNN model')
    parser.add_argument('--association_path', type=str, default='./checkpoints/association/association_best.pth',
                        help='Path to pretrained association model')
    
    # Other parameters
    parser.add_argument('--reid_type', type=str, default='spatial', choices=['spatial', 'channel'],
                        help='Type of ReID module to use')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension for ReID')
    parser.add_argument('--in_channels', type=int, default=256, help='Input channels for ReID module')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for GNN')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving checkpoints')
    parser.add_argument('--plot_losses', action='store_true', help='Plot losses after training')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames to use for training')
    
    return parser.parse_args()

def main():
    """Main function to run joint training"""
    # Parse arguments
    args = parse_args()
    
    try:
        # Import the main function from joint_model.py
        import sys
        import os
        
        # Ensure the current directory is in the path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import the main function from joint_model.py
        from joint_model import main as model_main
        
        # Override the args with our parsed arguments
        sys.argv = [sys.argv[0]]  # Keep only the script name
        
        # Call the main function from joint_model.py with our parsed arguments
        model_main(args)
        
    except ImportError as e:
        print(f"Error: Could not import from joint_model.py: {e}")
        print("Please make sure joint_model.py is in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 