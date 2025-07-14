import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
from tqdm import tqdm
import sys
from PIL import Image
import json
from torch.utils.data import Dataset

# Configure Python's stdout to be unbuffered for real-time logging
import functools
print = functools.partial(print, flush=True)

# Configure tqdm to flush output
class FlushingTqdm(tqdm):
    def display(self, *args, **kwargs):
        super().display(*args, **kwargs)
        sys.stdout.flush()

# Replace tqdm with our flushing version
tqdm = FlushingTqdm

# Import the individual training modules
sys.path.append('.')
from train_detector import parse_args as parse_detector_args
from train_reid_head import parse_args as parse_reid_args
from train_gnn import parse_args as parse_gnn_args
from train_association import parse_args as parse_association_args

# Import specific model components
from train_detector import MOT16DetectionDataset
from train_reid_head import ModifiedSpatialAttentionModule, ChannelAttentionReID
from train_gnn import GraphNN
from train_association import create_association_pairs

from track.data_track import MOT16Sequences
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate joint model on MOT16 sequences')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--detector_path', type=str, default='./checkpoints/detector/fasterrcnn_best.pth', 
                        help='Path to trained detector model')
    parser.add_argument('--reid_path', type=str, default='./checkpoints/reid/spatial_attention_best.pth',
                        help='Path to trained ReID model')
    parser.add_argument('--gnn_path', type=str, default='./checkpoints/gnn/gnn_best.pth',
                        help='Path to trained GNN model')
    parser.add_argument('--association_path', type=str, default='./checkpoints/association/association_best.pth',
                        help='Path to trained association model')
    parser.add_argument('--output_dir', type=str, default='./joint_output', help='Directory to save output visualizations')
    parser.add_argument('--seq_name', type=str, default='MOT16-04', help='MOT16 sequence name to test on')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to process')
    parser.add_argument('--device', type=str, default='', help='cuda or cpu')
    parser.add_argument('--reid_type', type=str, default='spatial', choices=['spatial', 'channel'],
                        help='Type of ReID module to use')
    parser.add_argument('--feature_dim', type=int, default=256, help='Feature dimension for ReID')
    parser.add_argument('--in_channels', type=int, default=256, help='Input channels for ReID module')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for GNN')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Detection confidence threshold')
    # Add parameters for multiple epochs
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to run')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--save_interval', type=int, default=1, help='Interval for saving checkpoints')
    parser.add_argument('--plot_losses', action='store_true', help='Plot losses after training')
    # Add loss weight parameters
    parser.add_argument('--detector_weight', type=float, default=1.0, help='Weight for detector loss')
    parser.add_argument('--reid_weight', type=float, default=0.5, help='Weight for ReID loss')
    parser.add_argument('--gnn_weight', type=float, default=0.3, help='Weight for GNN loss')
    parser.add_argument('--association_weight', type=float, default=0.7, help='Weight for association loss')
    # Add parameters for which components to train
    parser.add_argument('--train_detector', action='store_true', help='Train the detector component')
    parser.add_argument('--train_reid', action='store_true', help='Train the ReID component')
    parser.add_argument('--train_gnn', action='store_true', help='Train the GNN component')
    parser.add_argument('--train_association', action='store_true', help='Train the association component')
    return parser.parse_args() 


def load_detector(checkpoint_path, device):
    """Load the trained detector model"""
    print(f"Loading detector from {checkpoint_path}")
    
    # Create model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    
    # Replace the classifier with a new one for person detection (2 classes: background and person)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Add a method to extract ROI features
    def extract_roi_features(self, img, boxes):
        """Extract ROI features from image using boxes"""
        # Get the backbone features
        features = self.backbone(img.unsqueeze(0))
        
        # Use the feature maps from the FPN
        if isinstance(features, torch.Tensor):
            feature_maps = features
        else:
            # For FPN, use the last feature map (P5)
            feature_maps = features["3"]  # P5 feature map
        
        # Use ROI pooling to extract features for each box
        roi_pool = self.roi_heads.box_roi_pool
        rois = roi_pool(features, [boxes], [(img.shape[-2], img.shape[-1])])
        
        return rois
    
    # Add method to model
    import types
    model.extract_roi_features = types.MethodType(extract_roi_features, model)
    
    return model


def load_reid_module(checkpoint_path, reid_type, in_channels, feature_dim, device):
    """Load the trained ReID module"""
    print(f"Loading {reid_type} ReID module from {checkpoint_path}")
    
    # Create the appropriate ReID model
    if reid_type == 'spatial':
        model = ModifiedSpatialAttentionModule(in_channels=in_channels, feature_dim=feature_dim)
    else:  # channel
        model = ChannelAttentionReID(in_channels=in_channels, feature_dim=feature_dim)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def load_gnn(checkpoint_path, feature_dim, hidden_dim, num_layers, device):
    """Load the trained GNN module"""
    print(f"Loading GNN from {checkpoint_path}")
    
    # Create GNN model
    model = GraphNN(feature_dim=feature_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def load_association_module(checkpoint_path, device):
    """Load the trained association module"""
    print(f"Loading association module from {checkpoint_path}")
    
    # Create a simple MLP for association
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid()
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    try:
        # Try to extract just the association head from the checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Filter keys that belong to the association head
            association_state_dict = {k.replace('association_head.', ''): v for k, v in state_dict.items() 
                                     if k.startswith('association_head.')}
            
            if association_state_dict:
                # If we found association head keys, load them
                print("Loading association head weights from checkpoint")
                model.load_state_dict(association_state_dict)
            else:
                # If no association head keys, try to load directly (might be a standalone module)
                print("Attempting to load association module weights directly")
                model.load_state_dict(state_dict)
        else:
            # Try to load the checkpoint directly
            print("Loading association module weights directly")
            model.load_state_dict(checkpoint)
            
        print("Successfully loaded association module weights")
    except Exception as e:
        print(f"Warning: Could not load association module weights: {e}")
        print("Using randomly initialized association module for training")
    
    model.to(device)
    model.eval()
    
    return model 


def process_frame(frame, detector, reid_module, device, args):
    """Process a single frame with detector and ReID module"""
    # Convert frame to tensor
    img_tensor = frame['img'].to(device)
    
    # Run detection
    with torch.no_grad():
        detections = detector([img_tensor])[0]
    
    # Filter detections by confidence
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    
    # Keep only person detections (label 1) with high confidence
    mask = (scores > args.confidence_threshold) & (labels == 1)
    boxes = boxes[mask]
    scores = scores[mask]
    
    # Extract features for detected boxes
    features = None
    if len(boxes) > 0:
        with torch.no_grad():
            try:
                # Extract ROI features using the detector's backbone
                print(f"Extracting features for {len(boxes)} detections")
                roi_features = detector.extract_roi_features(img_tensor, boxes)
                
                # Process ROI features through the ReID module
                # First, we need to adapt the features to match the ReID module's input format
                # For spatial attention, we need to convert the ROI features to RGB-like format
                if args.reid_type == 'spatial':
                    # Resize ROI features to match the expected input of the ReID module
                    # The ModifiedSpatialAttentionModule expects input with 3 channels
                    batch_size = roi_features.size(0)
                    # If the ROI features have more than 3 channels, take the first 3
                    if roi_features.size(1) > 3:
                        roi_features_resized = roi_features[:, :3, :, :]
                    # If fewer than 3 channels, repeat the channels to get 3
                    else:
                        roi_features_resized = roi_features.repeat(1, 3 // roi_features.size(1) + 1, 1, 1)[:, :3, :, :]
                    
                    # Resize to expected dimensions (usually 64x128 for ReID)
                    roi_features_resized = torch.nn.functional.interpolate(
                        roi_features_resized, 
                        size=(128, 64), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Pass through ReID module
                    features = reid_module(roi_features_resized)
                else:  # channel attention
                    # For channel attention, we need to adapt the features similarly
                    batch_size = roi_features.size(0)
                    # If the ROI features have more than 3 channels, take the first 3
                    if roi_features.size(1) > 3:
                        roi_features_resized = roi_features[:, :3, :, :]
                    # If fewer than 3 channels, repeat the channels to get 3
                    else:
                        roi_features_resized = roi_features.repeat(1, 3 // roi_features.size(1) + 1, 1, 1)[:, :3, :, :]
                    
                    # Resize to expected dimensions
                    roi_features_resized = torch.nn.functional.interpolate(
                        roi_features_resized, 
                        size=(128, 64), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Pass through ReID module
                    features = reid_module(roi_features_resized)
                
                # Ensure features are normalized
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                
                print(f"Successfully extracted features with shape: {features.shape}")
                
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                print("Falling back to random features")
                # Fall back to random features
                features = torch.randn(len(boxes), args.feature_dim, device=device)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
    
    return {
        'boxes': boxes.cpu().numpy() if boxes is not None else np.array([]),
        'scores': scores.cpu().numpy() if scores is not None else np.array([]),
        'features': features.cpu().numpy() if features is not None else None
    }


def create_edge_index_from_features(prev_features, curr_features, device):
    """Create edge index for GNN based on feature similarity"""
    if prev_features is None or curr_features is None or len(prev_features) == 0 or len(curr_features) == 0:
        return None
    
    # Convert to tensors
    prev_features = torch.tensor(prev_features, device=device)
    curr_features = torch.tensor(curr_features, device=device)
    
    # Normalize features
    prev_features = torch.nn.functional.normalize(prev_features, p=2, dim=1)
    curr_features = torch.nn.functional.normalize(curr_features, p=2, dim=1)
    
    # Compute similarity matrix
    similarity = torch.mm(prev_features, curr_features.t())
    
    # Create edge index for edges with similarity above threshold
    threshold = 0.5
    edges = []
    num_prev = prev_features.shape[0]
    
    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            if similarity[i, j] > threshold:
                # Add edge from prev to curr
                edges.append([i, j + num_prev])
    
    if not edges:
        return None
    
    edge_index = torch.tensor(edges, device=device).t().contiguous()
    return edge_index 


def visualize_detections(frame, detections, output_path):
    """Visualize detections on frame and save to output_path"""
    # Convert tensor to numpy
    img_np = frame['img'].permute(1, 2, 0).cpu().numpy()
    
    # Normalize image for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw boxes
    for i, (box, score) in enumerate(zip(detections['boxes'], detections['scores'])):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save image
    cv2.imwrite(output_path, img_bgr)
    
    return img_bgr


def visualize_tracking(prev_frame, curr_frame, prev_detections, curr_detections, 
                      associations, output_path):
    """Visualize tracking associations between frames"""
    # Convert tensors to numpy
    prev_img = prev_frame['img'].permute(1, 2, 0).cpu().numpy()
    curr_img = curr_frame['img'].permute(1, 2, 0).cpu().numpy()
    
    # Normalize images for display
    prev_img = (prev_img - prev_img.min()) / (prev_img.max() - prev_img.min())
    prev_img = (prev_img * 255).astype(np.uint8)
    curr_img = (curr_img - curr_img.min()) / (curr_img.max() - curr_img.min())
    curr_img = (curr_img * 255).astype(np.uint8)
    
    # Convert to BGR for OpenCV
    prev_bgr = cv2.cvtColor(prev_img, cv2.COLOR_RGB2BGR)
    curr_bgr = cv2.cvtColor(curr_img, cv2.COLOR_RGB2BGR)
    
    # Create a combined image
    h, w = prev_img.shape[:2]
    combined_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined_img[:, :w] = prev_bgr
    combined_img[:, w:] = curr_bgr
    
    # Draw boxes on previous frame
    for i, box in enumerate(prev_detections['boxes']):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(combined_img, f"{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw boxes on current frame
    for i, box in enumerate(curr_detections['boxes']):
        x1, y1, x2, y2 = map(int, box)
        # Offset x coordinates for the combined image
        x1 += w
        x2 += w
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(combined_img, f"{i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw association lines
    if associations is not None:
        for prev_idx, curr_idx, score in associations:
            # Get box centers
            prev_box = prev_detections['boxes'][prev_idx]
            curr_box = curr_detections['boxes'][curr_idx]
            
            prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
            curr_center = ((curr_box[0] + curr_box[2]) // 2 + w, (curr_box[1] + curr_box[3]) // 2)
            
            # Draw line with color based on score (green for high score, red for low)
            color = (0, int(255 * score), int(255 * (1 - score)))
            cv2.line(combined_img, prev_center, curr_center, color, 2)
    
    # Save image
    cv2.imwrite(output_path, combined_img)
    
    return combined_img 


def plot_losses(train_losses, val_losses=None, output_dir='./'):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    # Plot validation loss if available
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    
    return plot_path


def save_checkpoint(epoch, models, optimizers, train_loss, val_loss, best_val_loss, checkpoint_path, is_best=False):
    """Save model checkpoint with training history"""
    checkpoint = {
        'epoch': epoch,
        'detector_state_dict': models['detector'].state_dict(),
        'reid_state_dict': models['reid'].state_dict(),
        'gnn_state_dict': models['gnn'].state_dict(),
        'association_state_dict': models['association'].state_dict(),
        'detector_optimizer_state_dict': optimizers['detector'].state_dict() if 'detector' in optimizers else None,
        'reid_optimizer_state_dict': optimizers['reid'].state_dict() if 'reid' in optimizers else None,
        'gnn_optimizer_state_dict': optimizers['gnn'].state_dict() if 'gnn' in optimizers else None,
        'association_optimizer_state_dict': optimizers['association'].state_dict() if 'association' in optimizers else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    }
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model if needed
    if is_best:
        best_model_path = os.path.join(os.path.dirname(checkpoint_path), "joint_best.pth")
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved to {best_model_path}")


def load_checkpoint(checkpoint_path, models, optimizers, device):
    """Load model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model states
        if 'detector_state_dict' in checkpoint and 'detector' in models:
            models['detector'].load_state_dict(checkpoint['detector_state_dict'])
        if 'reid_state_dict' in checkpoint and 'reid' in models:
            models['reid'].load_state_dict(checkpoint['reid_state_dict'])
        if 'gnn_state_dict' in checkpoint and 'gnn' in models:
            models['gnn'].load_state_dict(checkpoint['gnn_state_dict'])
        if 'association_state_dict' in checkpoint and 'association' in models:
            models['association'].load_state_dict(checkpoint['association_state_dict'])
        
        # Load optimizer states
        if 'detector_optimizer_state_dict' in checkpoint and 'detector' in optimizers:
            optimizers['detector'].load_state_dict(checkpoint['detector_optimizer_state_dict'])
        if 'reid_optimizer_state_dict' in checkpoint and 'reid' in optimizers:
            optimizers['reid'].load_state_dict(checkpoint['reid_optimizer_state_dict'])
        if 'gnn_optimizer_state_dict' in checkpoint and 'gnn' in optimizers:
            optimizers['gnn'].load_state_dict(checkpoint['gnn_optimizer_state_dict'])
        if 'association_optimizer_state_dict' in checkpoint and 'association' in optimizers:
            optimizers['association'].load_state_dict(checkpoint['association_optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from epoch {epoch} with validation loss: {best_val_loss}")
            
        return epoch, best_val_loss
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch")
        return 0, float('inf') 


class MOT16JointDataset(Dataset):
    """Dataset wrapper for MOT16 to extract temporal windows for joint training"""
    
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
        Returns a temporal window of frames for joint training
        """
        return self.data[idx] 


def train_epoch(models, data_loader, optimizers, device, args):
    """Train all models for one epoch following the joint training algorithm"""
    # Set models to training mode
    for model_name, model in models.items():
        if model_name in optimizers:  # Only train models with optimizers
            model.train()
        else:
            model.eval()
    
    # Loss weights as specified in the algorithm
    detector_weight = args.detector_weight
    reid_weight = args.reid_weight
    gnn_weight = args.gnn_weight
    association_weight = args.association_weight
    
    total_loss = 0
    total_detector_loss = 0
    total_reid_loss = 0
    total_gnn_loss = 0
    total_association_loss = 0
    successful_batches = 0
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, frames in pbar:
        try:
            # Process each temporal window (batch)
            batch_loss = 0
            
            # Extract features and boxes for all frames in the window
            all_features = []
            all_boxes = []
            all_gt_ids = []
            all_images = []
            all_targets = []
            
            for frame in frames:
                # Store original image and targets for detection loss
                all_images.append(frame['img'].to(device))
                
                # Extract ground truth boxes and labels for detection loss
                gt_boxes_dict = frame['gt']
                targets = {
                    'boxes': [],
                    'labels': []
                }
                
                for gt_id, gt_box in gt_boxes_dict.items():
                    if isinstance(gt_box, dict) and 'box' in gt_box:
                        # Handle case where gt_box is a dictionary with 'box' key
                        box = torch.tensor(gt_box['box'], device=device)
                        targets['boxes'].append(box)
                        targets['labels'].append(torch.tensor(1, device=device))  # Person class
                    elif isinstance(gt_box, (list, tuple)) and len(gt_box) == 4:
                        # Handle case where gt_box is directly a box
                        box = torch.tensor(gt_box, device=device)
                        targets['boxes'].append(box)
                        targets['labels'].append(torch.tensor(1, device=device))  # Person class
                
                # Convert lists to tensors
                if targets['boxes']:
                    targets['boxes'] = torch.stack(targets['boxes'])
                    targets['labels'] = torch.stack(targets['labels'])
                else:
                    targets['boxes'] = torch.zeros((0, 4), device=device)
                    targets['labels'] = torch.zeros(0, dtype=torch.int64, device=device)
                
                all_targets.append(targets)
                
                # Process frame with detector and ReID
                detections = process_frame(frame, models['detector'], models['reid'], device, args)
                
                # Store features and boxes
                all_features.append(detections['features'])
                all_boxes.append(detections['boxes'])
                
                # Extract ground truth IDs from frame
                gt_ids = list(gt_boxes_dict.keys())
                all_gt_ids.append(gt_ids)
            
            # Step 1: Compute detection loss (Ldet)
            detector_loss = 0
            if 'detector' in optimizers:
                optimizers['detector'].zero_grad()
                
                # Forward pass through detector for first frame
                outputs = models['detector']([all_images[0]])
                
                # Compute detection loss
                detector_loss_dict = {}
                for k in outputs[0].keys():
                    if k in all_targets[0]:
                        detector_loss_dict[k] = outputs[0][k]
                
                # Use the built-in loss computation from Faster R-CNN
                if hasattr(models['detector'], 'compute_loss'):
                    detector_loss_dict, _ = models['detector'].compute_loss(detector_loss_dict, [all_targets[0]])
                    detector_loss = sum(loss for loss in detector_loss_dict.values())
                else:
                    # Fallback to simple loss calculation
                    pred_boxes = outputs[0]['boxes']
                    pred_scores = outputs[0]['scores']
                    pred_labels = outputs[0]['labels']
                    
                    # Simple loss based on box regression and classification
                    if len(pred_boxes) > 0 and len(all_targets[0]['boxes']) > 0:
                        # Box regression loss (L1)
                        box_loss = torch.nn.functional.smooth_l1_loss(
                            pred_boxes, all_targets[0]['boxes']
                        )
                        
                        # Classification loss (CE)
                        cls_loss = torch.nn.functional.cross_entropy(
                            pred_scores.unsqueeze(1), 
                            (all_targets[0]['labels'] > 0).float()
                        )
                        
                        detector_loss = box_loss + cls_loss
                
                # Apply detector loss weight
                weighted_detector_loss = detector_weight * detector_loss
                
                # Backward pass for detector
                if detector_loss > 0:
                    weighted_detector_loss.backward(retain_graph=True)
                    optimizers['detector'].step()
                
                total_detector_loss += detector_loss.item()
            
            # Step 2: Compute ReID loss (Lreid)
            reid_loss = 0
            if 'reid' in optimizers:
                optimizers['reid'].zero_grad()
                
                # Implement ReID loss computation using triplet loss
                # We need at least two consecutive frames with detections to compute ReID loss
                if len(all_features) >= 2 and all_features[0] is not None and all_features[1] is not None:
                    # Get features from first two frames
                    anchor_features = all_features[0]
                    compare_features = all_features[1]
                    
                    # Get ground truth IDs for these frames
                    anchor_ids = all_gt_ids[0]
                    compare_ids = all_gt_ids[1]
                    
                    # Skip if either frame has no detections or IDs
                    if len(anchor_features) > 0 and len(compare_features) > 0 and len(anchor_ids) > 0 and len(compare_ids) > 0:
                        # Create triplets for training
                        triplet_loss = 0
                        num_triplets = 0
                        
                        # Convert features to tensors
                        anchor_features_tensor = torch.tensor(anchor_features, device=device)
                        compare_features_tensor = torch.tensor(compare_features, device=device)
                        
                        # Margin for triplet loss
                        margin = 0.3
                        
                        # For each anchor in the first frame
                        for i, anchor_id in enumerate(anchor_ids):
                            if i >= len(anchor_features):
                                continue
                                
                            # Find positive examples (same ID) in the second frame
                            positives = [j for j, compare_id in enumerate(compare_ids) if compare_id == anchor_id and j < len(compare_features)]
                            
                            # Find negative examples (different ID) in the second frame
                            negatives = [j for j, compare_id in enumerate(compare_ids) if compare_id != anchor_id and j < len(compare_features)]
                            
                            # Skip if we don't have both positives and negatives
                            if not positives or not negatives:
                                continue
                            
                            # Use the first positive
                            positive_idx = positives[0]
                            
                            # Use the hardest negative (closest in feature space)
                            anchor_feat = anchor_features_tensor[i]
                            negative_dists = []
                            
                            for neg_idx in negatives:
                                neg_feat = compare_features_tensor[neg_idx]
                                dist = torch.norm(anchor_feat - neg_feat)
                                negative_dists.append((dist.item(), neg_idx))
                            
                            # Sort by distance (ascending) and take the closest (hardest) negative
                            negative_dists.sort()
                            negative_idx = negative_dists[0][1]
                            
                            # Get features for the triplet
                            anchor = anchor_features_tensor[i]
                            positive = compare_features_tensor[positive_idx]
                            negative = compare_features_tensor[negative_idx]
                            
                            # Compute triplet loss
                            pos_dist = torch.norm(anchor - positive)
                            neg_dist = torch.norm(anchor - negative)
                            
                            # Triplet loss with margin
                            loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
                            triplet_loss += loss
                            num_triplets += 1
                        
                        # Average triplet loss
                        if num_triplets > 0:
                            reid_loss = triplet_loss / num_triplets
                
                # Apply reid loss weight
                weighted_reid_loss = reid_weight * reid_loss
                
                # Backward pass for ReID
                if reid_loss > 0:
                    weighted_reid_loss.backward(retain_graph=True)
                    optimizers['reid'].step()
                
                total_reid_loss += reid_loss.item() if isinstance(reid_loss, torch.Tensor) else reid_loss
            
            # Process pairs of consecutive frames for GNN and association
            for i in range(len(all_features) - 1):
                prev_features = all_features[i]
                curr_features = all_features[i+1]
                prev_boxes = all_boxes[i]
                curr_boxes = all_boxes[i+1]
                prev_ids = all_gt_ids[i]
                curr_ids = all_gt_ids[i+1]
                
                # Skip if either frame has no detections
                if (prev_features is None or len(prev_features) == 0 or
                    curr_features is None or len(curr_features) == 0):
                    continue
                
                # Create edge index for GNN
                edge_index = create_edge_index_from_features(
                    prev_features, 
                    curr_features, 
                    device
                )
                
                # Skip if no edges
                if edge_index is None:
                    continue
                
                # Prepare features for GNN
                prev_features_tensor = torch.tensor(prev_features, device=device)
                curr_features_tensor = torch.tensor(curr_features, device=device)
                combined_features = torch.cat([prev_features_tensor, curr_features_tensor], dim=0)
                
                # Step 3: Train GNN and compute GNN loss (Lgnn)
                gnn_loss = 0
                if 'gnn' in optimizers:
                    optimizers['gnn'].zero_grad()
                    
                    # Forward pass through GNN
                    node_embeddings, edge_weights, _ = models['gnn'](combined_features, edge_index)
                    
                    # Create target edge weights based on ground truth IDs
                    src, dst = edge_index
                    edge_targets = torch.zeros(edge_index.shape[1], device=device)
                    
                    num_prev = len(prev_boxes)
                    for j in range(edge_index.shape[1]):
                        if src[j] < num_prev and dst[j] >= num_prev:
                            # This is an edge from prev to curr
                            prev_idx = src[j].item()
                            curr_idx = dst[j].item() - num_prev
                            
                            # Check if IDs match
                            if prev_idx < len(prev_ids) and curr_idx < len(curr_ids):
                                if prev_ids[prev_idx] == curr_ids[curr_idx]:
                                    edge_targets[j] = 1.0
                    
                    # Compute GNN loss
                    gnn_loss = torch.nn.BCELoss()(edge_weights, edge_targets)
                    
                    # Apply GNN loss weight
                    weighted_gnn_loss = gnn_weight * gnn_loss
                    
                    # Backward pass for GNN
                    weighted_gnn_loss.backward(retain_graph=True)
                    optimizers['gnn'].step()
                    
                    total_gnn_loss += gnn_loss.item()
                
                # Step 4: Train association module and compute association loss (Lassoc)
                association_loss = 0
                if 'association' in optimizers:
                    # Create pairs for association training
                    pair_features = []
                    pair_labels = []
                    
                    for j in range(len(prev_features)):
                        for k in range(len(curr_features)):
                            # Create pair feature
                            prev_feat = torch.tensor(prev_features[j], device=device)
                            curr_feat = torch.tensor(curr_features[k], device=device)
                            pair_feature = torch.cat([prev_feat, curr_feat], dim=0)
                            
                            # Determine label based on ID match
                            if j < len(prev_ids) and k < len(curr_ids):
                                label = 1.0 if prev_ids[j] == curr_ids[k] else 0.0
                            else:
                                # If no ID info, use IoU as a proxy
                                iou = calculate_iou(prev_boxes[j], curr_boxes[k])
                                label = 1.0 if iou > 0.5 else 0.0
                            
                            pair_features.append(pair_feature)
                            pair_labels.append(torch.tensor(label, device=device))
                    
                    if pair_features:
                        optimizers['association'].zero_grad()
                        
                        # Stack features and labels
                        pair_features = torch.stack(pair_features)
                        pair_labels = torch.stack(pair_labels)
                        
                        # Forward pass through association module
                        association_scores = models['association'](pair_features).squeeze()
                        
                        # Compute association loss
                        association_loss = torch.nn.BCELoss()(association_scores, pair_labels)
                        
                        # Apply association loss weight
                        weighted_association_loss = association_weight * association_loss
                        
                        # Backward pass for association
                        weighted_association_loss.backward()
                        optimizers['association'].step()
                        
                        total_association_loss += association_loss.item()
                
                # Combine losses for this pair of frames
                batch_loss += (detector_weight * detector_loss + 
                               reid_weight * reid_loss + 
                               gnn_weight * gnn_loss + 
                               association_weight * association_loss)
            
            # Update statistics
            if batch_loss > 0:
                total_loss += batch_loss
                successful_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / max(successful_batches, 1)
            avg_det_loss = total_detector_loss / max(successful_batches, 1)
            avg_reid_loss = total_reid_loss / max(successful_batches, 1)
            avg_gnn_loss = total_gnn_loss / max(successful_batches, 1)
            avg_assoc_loss = total_association_loss / max(successful_batches, 1)
            
            pbar.set_description(
                f"Batch {batch_idx}/{len(data_loader)} | "
                f"Loss: {avg_loss:.4f} | Det: {avg_det_loss:.4f} | "
                f"ReID: {avg_reid_loss:.4f} | GNN: {avg_gnn_loss:.4f} | "
                f"Assoc: {avg_assoc_loss:.4f}"
            )
        
        except Exception as e:
            print(f"\nError in training batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate average losses
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
        avg_detector_loss = total_detector_loss / successful_batches if total_detector_loss > 0 else 0
        avg_reid_loss = total_reid_loss / successful_batches if total_reid_loss > 0 else 0
        avg_gnn_loss = total_gnn_loss / successful_batches if total_gnn_loss > 0 else 0
        avg_assoc_loss = total_association_loss / successful_batches if total_association_loss > 0 else 0
    else:
        avg_loss = float('inf')
        avg_detector_loss = 0
        avg_reid_loss = 0
        avg_gnn_loss = 0
        avg_assoc_loss = 0
        print("Warning: No successful batches in this epoch!")
    
    losses = {
        'total': avg_loss,
        'detector': avg_detector_loss,
        'reid': avg_reid_loss,
        'gnn': avg_gnn_loss,
        'association': avg_assoc_loss
    }
    
    return losses


def validate(models, data_loader, device, args):
    """Validate models on validation data using the joint training approach"""
    # Set models to evaluation mode
    for model_name, model in models.items():
        model.eval()
    
    # Loss weights as specified in the algorithm
    detector_weight = args.detector_weight
    reid_weight = args.reid_weight
    gnn_weight = args.gnn_weight
    association_weight = args.association_weight
    
    total_loss = 0
    total_detector_loss = 0
    total_reid_loss = 0
    total_gnn_loss = 0
    total_association_loss = 0
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(tqdm(data_loader, desc="Validating")):
            try:
                # Process each temporal window
                batch_loss = 0
                
                # Extract features and boxes for all frames in the window
                all_features = []
                all_boxes = []
                all_gt_ids = []
                all_images = []
                all_targets = []
                
                for frame in frames:
                    # Store original image and targets for detection loss
                    all_images.append(frame['img'].to(device))
                    
                    # Extract ground truth boxes and labels for detection loss
                    gt_boxes_dict = frame['gt']
                    targets = {
                        'boxes': [],
                        'labels': []
                    }
                    
                    for gt_id, gt_box in gt_boxes_dict.items():
                        if isinstance(gt_box, dict) and 'box' in gt_box:
                            # Handle case where gt_box is a dictionary with 'box' key
                            box = torch.tensor(gt_box['box'], device=device)
                            targets['boxes'].append(box)
                            targets['labels'].append(torch.tensor(1, device=device))  # Person class
                        elif isinstance(gt_box, (list, tuple)) and len(gt_box) == 4:
                            # Handle case where gt_box is directly a box
                            box = torch.tensor(gt_box, device=device)
                            targets['boxes'].append(box)
                            targets['labels'].append(torch.tensor(1, device=device))  # Person class
                    
                    # Convert lists to tensors
                    if targets['boxes']:
                        targets['boxes'] = torch.stack(targets['boxes'])
                        targets['labels'] = torch.stack(targets['labels'])
                    else:
                        targets['boxes'] = torch.zeros((0, 4), device=device)
                        targets['labels'] = torch.zeros(0, dtype=torch.int64, device=device)
                    
                    all_targets.append(targets)
                    
                    # Process frame with detector and ReID
                    detections = process_frame(frame, models['detector'], models['reid'], device, args)
                    
                    # Store features and boxes
                    all_features.append(detections['features'])
                    all_boxes.append(detections['boxes'])
                    
                    # Extract ground truth IDs from frame
                    gt_ids = list(gt_boxes_dict.keys())
                    all_gt_ids.append(gt_ids)
                
                # Step 1: Compute detection loss (Ldet)
                detector_loss = 0
                if len(all_images) > 0 and len(all_targets) > 0:
                    # Forward pass through detector for first frame
                    outputs = models['detector']([all_images[0]])
                    
                    # Compute detection loss
                    detector_loss_dict = {}
                    for k in outputs[0].keys():
                        if k in all_targets[0]:
                            detector_loss_dict[k] = outputs[0][k]
                    
                    # Use the built-in loss computation from Faster R-CNN
                    if hasattr(models['detector'], 'compute_loss'):
                        detector_loss_dict, _ = models['detector'].compute_loss(detector_loss_dict, [all_targets[0]])
                        detector_loss = sum(loss for loss in detector_loss_dict.values())
                    else:
                        # Fallback to simple loss calculation
                        pred_boxes = outputs[0]['boxes']
                        pred_scores = outputs[0]['scores']
                        pred_labels = outputs[0]['labels']
                        
                        # Simple loss based on box regression and classification
                        if len(pred_boxes) > 0 and len(all_targets[0]['boxes']) > 0:
                            # Box regression loss (L1)
                            box_loss = torch.nn.functional.smooth_l1_loss(
                                pred_boxes, all_targets[0]['boxes']
                            )
                            
                            # Classification loss (CE)
                            cls_loss = torch.nn.functional.cross_entropy(
                                pred_scores.unsqueeze(1), 
                                (all_targets[0]['labels'] > 0).float()
                            )
                            
                            detector_loss = box_loss + cls_loss
                    
                    total_detector_loss += detector_loss.item()
                
                # Step 2: Compute ReID loss (Lreid)
                reid_loss = 0
                # Implement ReID loss computation using triplet loss
                # We need at least two consecutive frames with detections to compute ReID loss
                if len(all_features) >= 2 and all_features[0] is not None and all_features[1] is not None:
                    # Get features from first two frames
                    anchor_features = all_features[0]
                    compare_features = all_features[1]
                    
                    # Get ground truth IDs for these frames
                    anchor_ids = all_gt_ids[0]
                    compare_ids = all_gt_ids[1]
                    
                    # Skip if either frame has no detections or IDs
                    if len(anchor_features) > 0 and len(compare_features) > 0 and len(anchor_ids) > 0 and len(compare_ids) > 0:
                        # Create triplets for training
                        triplet_loss = 0
                        num_triplets = 0
                        
                        # Convert features to tensors
                        anchor_features_tensor = torch.tensor(anchor_features, device=device)
                        compare_features_tensor = torch.tensor(compare_features, device=device)
                        
                        # Margin for triplet loss
                        margin = 0.3
                        
                        # For each anchor in the first frame
                        for i, anchor_id in enumerate(anchor_ids):
                            if i >= len(anchor_features):
                                continue
                                
                            # Find positive examples (same ID) in the second frame
                            positives = [j for j, compare_id in enumerate(compare_ids) if compare_id == anchor_id and j < len(compare_features)]
                            
                            # Find negative examples (different ID) in the second frame
                            negatives = [j for j, compare_id in enumerate(compare_ids) if compare_id != anchor_id and j < len(compare_features)]
                            
                            # Skip if we don't have both positives and negatives
                            if not positives or not negatives:
                                continue
                            
                            # Use the first positive
                            positive_idx = positives[0]
                            
                            # Use the hardest negative (closest in feature space)
                            anchor_feat = anchor_features_tensor[i]
                            negative_dists = []
                            
                            for neg_idx in negatives:
                                neg_feat = compare_features_tensor[neg_idx]
                                dist = torch.norm(anchor_feat - neg_feat)
                                negative_dists.append((dist.item(), neg_idx))
                            
                            # Sort by distance (ascending) and take the closest (hardest) negative
                            negative_dists.sort()
                            negative_idx = negative_dists[0][1]
                            
                            # Get features for the triplet
                            anchor = anchor_features_tensor[i]
                            positive = compare_features_tensor[positive_idx]
                            negative = compare_features_tensor[negative_idx]
                            
                            # Compute triplet loss
                            pos_dist = torch.norm(anchor - positive)
                            neg_dist = torch.norm(anchor - negative)
                            
                            # Triplet loss with margin
                            loss = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
                            triplet_loss += loss
                            num_triplets += 1
                        
                        # Average triplet loss
                        if num_triplets > 0:
                            reid_loss = triplet_loss / num_triplets
                
                total_reid_loss += reid_loss.item() if isinstance(reid_loss, torch.Tensor) else reid_loss
                
                # Process pairs of consecutive frames for GNN and association
                for i in range(len(all_features) - 1):
                    prev_features = all_features[i]
                    curr_features = all_features[i+1]
                    prev_boxes = all_boxes[i]
                    curr_boxes = all_boxes[i+1]
                    prev_ids = all_gt_ids[i]
                    curr_ids = all_gt_ids[i+1]
                    
                    # Skip if either frame has no detections
                    if (prev_features is None or len(prev_features) == 0 or
                        curr_features is None or len(curr_features) == 0):
                        continue
                    
                    # Create edge index for GNN
                    edge_index = create_edge_index_from_features(
                        prev_features, 
                        curr_features, 
                        device
                    )
                    
                    # Skip if no edges
                    if edge_index is None:
                        continue
                    
                    # Prepare features for GNN
                    prev_features_tensor = torch.tensor(prev_features, device=device)
                    curr_features_tensor = torch.tensor(curr_features, device=device)
                    combined_features = torch.cat([prev_features_tensor, curr_features_tensor], dim=0)
                    
                    # Step 3: Evaluate GNN and compute GNN loss (Lgnn)
                    gnn_loss = 0
                    # Forward pass through GNN
                    node_embeddings, edge_weights, _ = models['gnn'](combined_features, edge_index)
                    
                    # Create target edge weights based on ground truth IDs
                    src, dst = edge_index
                    edge_targets = torch.zeros(edge_index.shape[1], device=device)
                    
                    num_prev = len(prev_boxes)
                    for j in range(edge_index.shape[1]):
                        if src[j] < num_prev and dst[j] >= num_prev:
                            # This is an edge from prev to curr
                            prev_idx = src[j].item()
                            curr_idx = dst[j].item() - num_prev
                            
                            # Check if IDs match
                            if prev_idx < len(prev_ids) and curr_idx < len(curr_ids):
                                if prev_ids[prev_idx] == curr_ids[curr_idx]:
                                    edge_targets[j] = 1.0
                    
                    # Compute GNN loss
                    gnn_loss = torch.nn.BCELoss()(edge_weights, edge_targets)
                    total_gnn_loss += gnn_loss.item()
                    
                    # Step 4: Evaluate association module and compute association loss (Lassoc)
                    association_loss = 0
                    # Create pairs for association evaluation
                    pair_features = []
                    pair_labels = []
                    
                    for j in range(len(prev_features)):
                        for k in range(len(curr_features)):
                            # Create pair feature
                            prev_feat = torch.tensor(prev_features[j], device=device)
                            curr_feat = torch.tensor(curr_features[k], device=device)
                            pair_feature = torch.cat([prev_feat, curr_feat], dim=0)
                            
                            # Determine label based on ID match
                            if j < len(prev_ids) and k < len(curr_ids):
                                label = 1.0 if prev_ids[j] == curr_ids[k] else 0.0
                            else:
                                # If no ID info, use IoU as a proxy
                                iou = calculate_iou(prev_boxes[j], curr_boxes[k])
                                label = 1.0 if iou > 0.5 else 0.0
                            
                            pair_features.append(pair_feature)
                            pair_labels.append(torch.tensor(label, device=device))
                    
                    if pair_features:
                        # Stack features and labels
                        pair_features = torch.stack(pair_features)
                        pair_labels = torch.stack(pair_labels)
                        
                        # Forward pass through association module
                        association_scores = models['association'](pair_features).squeeze()
                        
                        # Compute association loss
                        association_loss = torch.nn.BCELoss()(association_scores, pair_labels)
                        total_association_loss += association_loss.item()
                    
                    # Combine losses for this pair of frames with weights
                    batch_loss += (detector_weight * detector_loss + 
                                  reid_weight * reid_loss + 
                                  gnn_weight * gnn_loss + 
                                  association_weight * association_loss)
                
                # Update statistics
                if batch_loss > 0:
                    total_loss += batch_loss
                    successful_batches += 1
            
            except Exception as e:
                print(f"\nError during validation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate average losses
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
        avg_detector_loss = total_detector_loss / successful_batches if total_detector_loss > 0 else 0
        avg_reid_loss = total_reid_loss / successful_batches if total_reid_loss > 0 else 0
        avg_gnn_loss = total_gnn_loss / successful_batches if total_gnn_loss > 0 else 0
        avg_assoc_loss = total_association_loss / successful_batches if total_association_loss > 0 else 0
    else:
        avg_loss = float('inf')
        avg_detector_loss = 0
        avg_reid_loss = 0
        avg_gnn_loss = 0
        avg_assoc_loss = 0
        print("Warning: No successful validation batches!")
    
    losses = {
        'total': avg_loss,
        'detector': avg_detector_loss,
        'reid': avg_reid_loss,
        'gnn': avg_gnn_loss,
        'association': avg_assoc_loss
    }
    
    return losses


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes intersect
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # Calculate area of intersection
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate IoU
    iou = area_i / (area1 + area2 - area_i)
    
    return iou 


def process_sequence(args, models=None):
    """Process a sequence with the trained models"""
    # Set device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models if not provided
    if models is None:
        models = {}
        models['detector'] = load_detector(args.detector_path, device)
        models['reid'] = load_reid_module(args.reid_path, args.reid_type, args.in_channels, args.feature_dim, device)
        models['gnn'] = load_gnn(args.gnn_path, args.feature_dim, args.hidden_dim, args.num_gnn_layers, device)
        models['association'] = load_association_module(args.association_path, device)
    
    # Load sequence
    print(f"Loading sequence {args.seq_name}")
    sequence = MOT16Sequences(args.seq_name, args.data_dir, load_seg=False)[0]
    
    # Limit frames if specified
    num_frames = min(args.num_frames, len(sequence)) if args.num_frames else len(sequence)
    print(f"Processing {num_frames} frames")
    
    # For multi-epoch runs, process the final epoch only
    epoch_output_dir = os.path.join(args.output_dir, f"epoch_{args.num_epochs}")
    os.makedirs(epoch_output_dir, exist_ok=True)
    
    # Process frames
    prev_frame = None
    prev_detections = None
    
    for frame_idx in tqdm(range(num_frames)):
        # Get frame
        frame = sequence[frame_idx]
        
        # Process frame
        detections = process_frame(frame, models['detector'], models['reid'], device, args)
        
        # Visualize detections
        detection_path = os.path.join(epoch_output_dir, f"detection_{frame_idx:06d}.jpg")
        visualize_detections(frame, detections, detection_path)
        
        # Process tracking if we have a previous frame
        if prev_frame is not None and prev_detections is not None:
            # Skip if either frame has no detections
            if (prev_detections['features'] is None or len(prev_detections['features']) == 0 or
                detections['features'] is None or len(detections['features']) == 0):
                print(f"Skipping tracking for frame {frame_idx} due to missing detections")
            else:
                # Create edge index for GNN
                edge_index = create_edge_index_from_features(
                    prev_detections['features'], 
                    detections['features'], 
                    device
                )
                
                # Run GNN if we have edges
                associations = None
                if edge_index is not None:
                    # Prepare features for GNN
                    prev_features = torch.tensor(prev_detections['features'], device=device)
                    curr_features = torch.tensor(detections['features'], device=device)
                    combined_features = torch.cat([prev_features, curr_features], dim=0)
                    
                    # Run GNN
                    with torch.no_grad():
                        node_embeddings, edge_weights, _ = models['gnn'](combined_features, edge_index)
                    
                    # Get edge information
                    src, dst = edge_index
                    num_prev = len(prev_detections['boxes'])
                    
                    # Create associations based on edge weights
                    associations = []
                    for i in range(edge_index.shape[1]):
                        # Only consider edges from prev to curr
                        if src[i] < num_prev and dst[i] >= num_prev:
                            prev_idx = src[i].item()
                            curr_idx = dst[i].item() - num_prev
                            score = edge_weights[i].item()
                            
                            # Use association module for final decision
                            if score > 0.3:  # Filter low GNN scores
                                # Create pair feature
                                prev_feat = prev_features[prev_idx]
                                curr_feat = curr_features[curr_idx]
                                pair_feature = torch.cat([prev_feat, curr_feat], dim=0)
                                
                                # Get association score
                                with torch.no_grad():
                                    assoc_score = models['association'](pair_feature.unsqueeze(0)).item()
                                
                                # Add to associations if score is high enough
                                if assoc_score > 0.5:
                                    associations.append((prev_idx, curr_idx, assoc_score))
                
                # Visualize tracking
                tracking_path = os.path.join(epoch_output_dir, f"tracking_{frame_idx:06d}.jpg")
                visualize_tracking(prev_frame, frame, prev_detections, detections, associations, tracking_path)
        
        # Update previous frame and detections
        prev_frame = frame
        prev_detections = detections
    
    print(f"Processed {num_frames} frames. Results saved to {epoch_output_dir}")
    return epoch_output_dir 


def main(args=None):
    """Main function for training and evaluating the joint model"""
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Set device
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    models = {}
    models['detector'] = load_detector(args.detector_path, device)
    models['reid'] = load_reid_module(args.reid_path, args.reid_type, args.in_channels, args.feature_dim, device)
    models['gnn'] = load_gnn(args.gnn_path, args.feature_dim, args.hidden_dim, args.num_gnn_layers, device)
    models['association'] = load_association_module(args.association_path, device)
    
    # Create optimizers for trainable models based on command line arguments
    optimizers = {}
    
    # Only create optimizers for components we want to train
    if args.train_detector:
        print("Setting up optimizer for detector")
        optimizers['detector'] = torch.optim.Adam(models['detector'].parameters(), lr=args.lr)
    
    if args.train_reid:
        print("Setting up optimizer for ReID module")
        optimizers['reid'] = torch.optim.Adam(models['reid'].parameters(), lr=args.lr)
    
    if args.train_gnn or not args.train_detector and not args.train_reid and not args.train_association:
        # Default to training GNN if no specific component is selected
        print("Setting up optimizer for GNN")
        optimizers['gnn'] = torch.optim.Adam(models['gnn'].parameters(), lr=args.lr)
    
    if args.train_association or not args.train_detector and not args.train_reid and not args.train_gnn:
        # Default to training association if no specific component is selected
        print("Setting up optimizer for association module")
        optimizers['association'] = torch.optim.Adam(models['association'].parameters(), lr=args.lr)
    
    # If no optimizers were created, default to training GNN and association
    if not optimizers:
        print("No components selected for training. Defaulting to GNN and association.")
        optimizers['gnn'] = torch.optim.Adam(models['gnn'].parameters(), lr=args.lr)
        optimizers['association'] = torch.optim.Adam(models['association'].parameters(), lr=args.lr)
    
    # Print which components are being trained
    print(f"Training components: {', '.join(optimizers.keys())}")
    
    # Load datasets
    print("Loading MOT16 sequences...")
    train_sequences = []
    val_sequences = []
    
    # Use different sequences for training and validation
    train_seq_names = args.seq_names if hasattr(args, 'seq_names') else ['MOT16-02', 'MOT16-04', 'MOT16-05']
    val_seq_names = args.val_seq_names if hasattr(args, 'val_seq_names') else ['MOT16-09']
    
    for seq_name in train_seq_names:
        train_sequences.append(MOT16Sequences(seq_name, args.data_dir, load_seg=False)[0])
    
    for seq_name in val_seq_names:
        val_sequences.append(MOT16Sequences(seq_name, args.data_dir, load_seg=False)[0])
    
    # Create datasets
    train_dataset = MOT16JointDataset(train_sequences, max_frames=args.num_frames, temporal_window=getattr(args, 'temporal_window', 2))
    val_dataset = MOT16JointDataset(val_sequences, max_frames=args.num_frames//5 if args.num_frames else None, temporal_window=getattr(args, 'temporal_window', 2))
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=getattr(args, 'batch_size', 1),  # Process one temporal window at a time
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0),
        collate_fn=lambda x: x[0]  # Each batch is a single temporal window
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=getattr(args, 'batch_size', 1),
        shuffle=False,
        num_workers=getattr(args, 'num_workers', 0),
        collate_fn=lambda x: x[0]  # Each batch is a single temporal window
    )
    
    # Initialize training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Check if resuming from checkpoint
    checkpoint_path = os.path.join(args.output_dir, "joint_interrupted.pth")
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}, attempting to resume...")
        try:
            start_epoch, best_val_loss = load_checkpoint(checkpoint_path, models, optimizers, device)
            start_epoch += 1  # Start from next epoch
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            start_epoch = 0
    
    # Train for specified number of epochs
    if args.num_epochs > 1:
        print(f"Training for {args.num_epochs} epochs...")
        
        for epoch in range(start_epoch, args.num_epochs):
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            
            # Train
            train_loss = train_epoch(models, train_loader, optimizers, device, args)
            train_losses.append(train_loss['total'])
            
            # Validate
            val_loss = validate(models, val_loader, device, args)
            val_losses.append(val_loss['total'])
            
            print(f"  Train Loss: {train_loss['total']:.4f}")
            print(f"  Val Loss: {val_loss['total']:.4f}")
            
            # Save checkpoint if at save interval
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_path = os.path.join(args.output_dir, f"joint_epoch_{epoch+1}.pth")
                
                # Check if this is the best model
                is_best = val_loss['total'] < best_val_loss
                if is_best:
                    best_val_loss = val_loss['total']
                    print(f"  New best model with validation loss: {best_val_loss:.4f}")
                
                # Save checkpoint
                save_checkpoint(
                    epoch, models, optimizers, 
                    train_loss['total'], val_loss['total'], best_val_loss, 
                    checkpoint_path, is_best
                )
        
        # Save losses to a JSON file
        losses_path = os.path.join(args.output_dir, 'losses.json')
        with open(losses_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': list(range(start_epoch + 1, args.num_epochs + 1))
            }, f)
        print(f"Losses saved to {losses_path}")
        
        # Plot losses if requested
        if args.plot_losses:
            plot_losses(train_losses, val_losses, args.output_dir)
    
    # Process sequence with trained models
    process_sequence(args, models)


if __name__ == "__main__":
    main() 