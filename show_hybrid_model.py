import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
from tqdm import tqdm
import traceback
from scipy.optimize import linear_sum_assignment
import cv2
from PIL import Image
import glob

from track.data_track import MOT16Sequences
from track.joint_model_hybrid import HybridDetectionTrackingModel
from track.visualization import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize hybrid detection+GNN tracking model')
    parser.add_argument('--data_dir', type=str, default='./data/MOT16', help='MOT16 data directory')
    parser.add_argument('--seq_name', type=str, default='MOT16-04', help='Sequence name for visualization')
    parser.add_argument('--device', type=str, default='', help='cuda or cpu')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process')
    parser.add_argument('--save_video', action='store_true', help='Save the visualization as a video file')
    parser.add_argument('--output', type=str, default='tracking_hybrid.mp4', help='Output video filename')
    parser.add_argument('--detection_threshold', type=float, default=0.3, help='Detection score threshold')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Similarity threshold for tracking')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='IoU threshold for spatial matching')
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/hybrid_best.pth', help='Path to trained model checkpoint')
    parser.add_argument('--persons_only', action='store_true', help='Only show person/human detections')
    parser.add_argument('--mot_classes', action='store_true', help='Show MOT Challenge categories (person, car, motorcycle, bicycle)')
    parser.add_argument('--class_ids', type=str, default='1,3,4,2', help='Comma-separated list of class IDs to track')
    parser.add_argument('--jijel', action='store_true', help='Use Jijel image sequence from data/jijel')
    parser.add_argument('--resize', type=int, default=False, help='Resize images to this width (aspect ratio preserved)')
    parser.add_argument('--webcam', type=int, default=None, help='Camera index for webcam input (e.g., 0 for default camera)')
    parser.add_argument('--video', type=str, default=None, help='Path to video file for processing')
    parser.add_argument('--fps', type=int, default=30, help='FPS for webcam/video processing')
    parser.add_argument('--display_scale', type=float, default=1.0, help='Scale factor for display window')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print all arguments for debugging
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    return args

def create_edge_index(boxes, max_distance=0.5, device='cpu'):
    
    n = boxes.shape[0]
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Calculate box centers
    centers = torch.zeros((n, 2), device=device)
    centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    
    # Normalize coordinates to [0, 1] range
    x_min, _ = torch.min(boxes[:, 0], dim=0)
    y_min, _ = torch.min(boxes[:, 1], dim=0)
    x_max, _ = torch.max(boxes[:, 2], dim=0)
    y_max, _ = torch.max(boxes[:, 3], dim=0)
    
    width = x_max - x_min
    height = y_max - y_min
    
    centers[:, 0] = (centers[:, 0] - x_min) / (width + 1e-6)
    centers[:, 1] = (centers[:, 1] - y_min) / (height + 1e-6)
    
    # Calculate pairwise distance
    src_ids = []
    dst_ids = []
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = torch.sqrt(torch.sum((centers[i] - centers[j])**2))
                if dist < max_distance:
                    src_ids.append(i)
                    dst_ids.append(j)
    
    # Create edge index tensor
    if not src_ids:  # No edges created
        return torch.zeros((2, 0), dtype=torch.long, device=device)
        
    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long, device=device)
    return edge_index

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
    # Convert to numpy for easier calculation if tensors
    if isinstance(box1, torch.Tensor):
        box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.cpu().numpy()
        
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-8)  # Add small epsilon to avoid division by zero
    return iou

def associate_detections_to_tracks(prev_tracks, curr_boxes, curr_features, iou_threshold=0.3, device='cpu'):
    """
    Associate detections to existing tracks using appearance and spatial information
    
    Args:
        prev_tracks: List of dictionaries with previous track information
        curr_boxes: Tensor of current bounding boxes [N, 4]
        curr_features: Tensor of current embedding features [N, feature_dim]
        iou_threshold: IoU threshold for considering a match
        device: Device for computation
        
    Returns:
        assignments: List of (prev_idx, curr_idx) tuples for matched tracks
        unmatched_tracks: List of indices for unmatched previous tracks
        unmatched_detections: List of indices for unmatched current detections
    """
    if len(prev_tracks) == 0:
        # All detections are new
        return [], [], list(range(len(curr_boxes)))
        
    if len(curr_boxes) == 0:
        # All tracks are lost
        return [], list(range(len(prev_tracks))), []
        
    # Create cost matrix combining IoU and feature similarity
    cost_matrix = np.zeros((len(prev_tracks), len(curr_boxes)))
    
    # Extract previous features and boxes
    try:
        # Handle different possible formats of features
        prev_features_list = []
        for t in prev_tracks:
            feature = t['feature']
            # Check if feature is already a tensor
            if isinstance(feature, torch.Tensor):
                prev_features_list.append(feature)
            else:
                # Convert to tensor if it's not already
                prev_features_list.append(torch.tensor(feature, device=device))
        
        # Stack features ensuring they're all on the same device
        prev_features = torch.stack(prev_features_list).to(device)
        
        # Calculate feature similarity component (cosine similarity)
        feature_sim = torch.mm(prev_features, curr_features.t()).cpu().numpy()
        feature_cost = 1.0 - feature_sim  # Lower cost means higher similarity
    except Exception as e:
        print(f"Error processing features: {e}")
        # Fallback to IoU-only matching if feature comparison fails
        feature_cost = np.ones((len(prev_tracks), len(curr_boxes)))
    
    # Calculate IoU component
    iou_matrix = np.zeros((len(prev_tracks), len(curr_boxes)))
    for i, track in enumerate(prev_tracks):
        prev_box = track['box']
        for j, curr_box in enumerate(curr_boxes):
            iou_matrix[i, j] = calculate_iou(prev_box, curr_box)
    
    # Combine costs: 70% feature similarity, 30% IoU
    # Higher IoU and higher feature similarity should give lower cost
    iou_cost = 1.0 - iou_matrix
    cost_matrix = 0.7 * feature_cost + 0.3 * iou_cost
    
    # Apply Hungarian algorithm for optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out low-confidence assignments
    assignments = []
    unmatched_tracks = list(range(len(prev_tracks)))
    unmatched_detections = list(range(len(curr_boxes)))
    
    for row_idx, col_idx in zip(row_indices, col_indices):
        # Only consider matches with sufficient IoU and feature similarity
        if cost_matrix[row_idx, col_idx] < 0.7 and iou_matrix[row_idx, col_idx] > iou_threshold:
            assignments.append((row_idx, col_idx))
            if row_idx in unmatched_tracks:
                unmatched_tracks.remove(row_idx)
            if col_idx in unmatched_detections:
                unmatched_detections.remove(col_idx)
                
    return assignments, unmatched_tracks, unmatched_detections

def process_video_input(args, model, device):
    """
    Process video input from either webcam or video file
    
    Args:
        args: Command-line arguments
        model: The detection and tracking model
        device: Device for computation (CPU/CUDA)
    """
    # Determine source (webcam or video file)
    if args.webcam is not None:
        print(f"Opening webcam at index {args.webcam}")
        cap = cv2.VideoCapture(args.webcam)
        source_name = f"Webcam {args.webcam}"
    elif args.video is not None:
        print(f"Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        source_name = os.path.basename(args.video)
    else:
        print("Error: No video source specified")
        return
    
    # Check if opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = args.fps  # Use default FPS if not available from source
    
    print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")
    
    # Resize if requested
    if args.resize:
        display_width = args.resize
        display_height = int(frame_height * (display_width / frame_width))
    else:
        display_width = int(frame_width)
        display_height = int(frame_height)
    
    print(f"Display dimensions: {display_width}x{display_height}")
    
    # Initialize output video writer if saving is requested
    out = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (display_width, display_height))
        print(f"Saving output to {args.output}")
    
    # Initialize track management
    all_tracks = {}  # Dictionary to store all tracks: {track_id: track_info}
    next_track_id = 1  # Counter for assigning new track IDs
    
    # Store previous state for tracking
    prev_boxes = None
    prev_features = None
    prev_track_ids = None
    active_tracks = []  # List of active tracks from previous frame
    
    # Frame counter
    frame_idx = 0
    max_frames = args.max_frames if args.max_frames else float('inf')
    
    # Process frames
    print(f"Starting video processing from {source_name}...")
    output_file = os.path.join(f"{args.seq_name}.txt")
    
    while cap.isOpened() and frame_idx < max_frames:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if requested
        if args.resize or args.display_scale != 1.0:
            frame_rgb = cv2.resize(frame_rgb, (display_width, display_height))
        
        # Convert to tensor format
        img_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # Convert to CxHxW format
        
        # Process frame
        with torch.no_grad():
            input_tensor = img_tensor.to(device).unsqueeze(0)  # Add batch dimension
            
            # Create edge index from previous boxes if available
            edge_index = None
            if prev_boxes is not None and prev_features is not None:
                edge_index = create_edge_index(prev_boxes, device=device)
            
            # Forward pass
            outputs = model(
                input_tensor, 
                prev_features=prev_features, 
                prev_boxes=prev_boxes,
                edge_index=edge_index,
                prev_track_ids=prev_track_ids
            )
            
        # Get detections and ReID features
        detector_outputs = outputs['detector_outputs']
        boxes = detector_outputs['boxes']
        scores = detector_outputs['scores']
        labels = detector_outputs['labels']
        reid_features = outputs['reid_features']
        
        # Process detections
        current_frame_tracks = []
        if len(boxes) > 0:
            # Apply detection threshold
            detection_mask = scores >= args.detection_threshold
            
            # Check if we have valid detections after thresholding
            if detection_mask.sum() > 0:
                # Filter detections
                all_boxes = boxes[detection_mask]
                all_scores = scores[detection_mask]
                all_labels = labels[detection_mask]
                all_reid_features = reid_features[detection_mask] if reid_features is not None else torch.zeros((detection_mask.sum().item(), 128), device=device)
                
                # Apply persons only filter if specified
                if args.persons_only:
                    persons_mask = all_labels == 1  # Person class is 1 in COCO
                    if persons_mask.sum() > 0:  # If we have any persons
                        all_boxes = all_boxes[persons_mask]
                        all_scores = all_scores[persons_mask]
                        all_labels = all_labels[persons_mask]
                        all_reid_features = all_reid_features[persons_mask] if all_reid_features is not None else torch.zeros((persons_mask.sum().item(), 128), device=device)
                # Apply MOT classes filter if specified
                elif args.mot_classes:
                    # MOT Challenge classes: person (1), car (3), motorcycle (4), bicycle (2)
                    mot_classes = [1, 3, 4, 2]  # COCO classes for MOT Challenge categories
                    mot_mask = torch.zeros_like(all_labels, dtype=torch.bool)
                    for class_id in mot_classes:
                        mot_mask = mot_mask | (all_labels == class_id)
                    
                    if mot_mask.sum() > 0:  # If we have any MOT objects
                        all_boxes = all_boxes[mot_mask]
                        all_scores = all_scores[mot_mask]
                        all_labels = all_labels[mot_mask]
                        all_reid_features = all_reid_features[mot_mask] if all_reid_features is not None else torch.zeros((mot_mask.sum().item(), 128), device=device)
                # Custom class ids if specified
                elif args.class_ids != '1,3,4,2':
                    # Parse comma-separated class IDs
                    custom_classes = [int(class_id) for class_id in args.class_ids.split(',')]
                    custom_mask = torch.zeros_like(all_labels, dtype=torch.bool)
                    for class_id in custom_classes:
                        custom_mask = custom_mask | (all_labels == class_id)
                    
                    if custom_mask.sum() > 0:  # If we have any custom objects
                        all_boxes = all_boxes[custom_mask]
                        all_scores = all_scores[custom_mask]
                        all_labels = all_labels[custom_mask]
                        all_reid_features = all_reid_features[custom_mask] if all_reid_features is not None else torch.zeros((custom_mask.sum().item(), 128), device=device)
                
                total_dets = len(all_boxes)
                
                # First frame: initialize tracks
                if frame_idx == 0 or len(active_tracks) == 0:
                    for i in range(total_dets):
                        box = all_boxes[i].cpu().numpy()
                        feature = all_reid_features[i].detach().cpu()
                        score = all_scores[i].item()
                        label = all_labels[i].item()
                        
                        # Calculate center point
                        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                        
                        # Create new track
                        track = {
                            'id': next_track_id,
                            'box': box,
                            'score': score,
                            'feature': feature,
                            'position': np.append(center, [0]),  # Add z=0
                            'age': 1,
                            'time_since_update': 0,
                            'label': label
                        }
                        # Add to current frame and global track storage
                        current_frame_tracks.append(track)
                        all_tracks[next_track_id] = track
                        next_track_id += 1
                else:
                    # Association with previous tracks
                    assignments, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(
                        active_tracks, 
                        all_boxes,
                        all_reid_features,
                        iou_threshold=args.iou_threshold,
                        device=device
                    )
                    
                    # Update matched tracks
                    for track_idx, det_idx in assignments:
                        # Get previous track
                        prev_track = active_tracks[track_idx]
                        track_id = prev_track['id']
                        
                        # Update with new detection
                        box = all_boxes[det_idx].cpu().numpy()
                        feature = all_reid_features[det_idx].detach().cpu()
                        score = all_scores[det_idx].item()
                        label = all_labels[det_idx].item()
                        
                        # Calculate center point
                        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                        
                        # Update track
                        updated_track = {
                            'id': track_id,
                            'box': box,
                            'score': score,
                            'feature': feature,  # Use new feature
                            'position': np.append(center, [0]),
                            'age': prev_track['age'] + 1,
                            'time_since_update': 0,
                            'label': label
                        }
                        
                        # Add to current frame and update global track storage
                        current_frame_tracks.append(updated_track)
                        all_tracks[track_id] = updated_track
                    
                    # Add new tracks for unmatched detections
                    for det_idx in unmatched_detections:
                        box = all_boxes[det_idx].cpu().numpy()
                        feature = all_reid_features[det_idx].detach().cpu()
                        score = all_scores[det_idx].item()
                        label = all_labels[det_idx].item()
                        
                        # Calculate center point
                        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                        
                        # Create new track
                        track = {
                            'id': next_track_id,
                            'box': box,
                            'score': score, 
                            'feature': feature,
                            'position': np.append(center, [0]),
                            'age': 1,
                            'time_since_update': 0,
                            'label': label
                        }
                        
                        # Add to current frame and global track storage
                        current_frame_tracks.append(track)
                        all_tracks[next_track_id] = track
                        next_track_id += 1
                
                # Update state for next frame
                active_tracks = current_frame_tracks.copy()
                prev_boxes = all_boxes
                prev_features = all_reid_features
                prev_track_ids = [track['id'] for track in current_frame_tracks]
        
        # Draw results on frame
        frame_display = frame_rgb.copy()
        
        # Draw bounding boxes and IDs
        for track in current_frame_tracks:
            box = track['box']
            track_id = track['id']
            label = track['label']
            score = track['score']
            
            # Convert box to integers for drawing
            x1, y1, x2, y2 = map(int, box)
            
            # Generate a consistent color based on track ID
            color_id = track_id % 10  # Cycle through 10 colors
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                (0, 0, 128), (128, 128, 0)
            ]
            color = colors[color_id]
            
            # Draw box
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID and class
            class_names = {
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
                5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck'
            }
            class_name = class_names.get(label, f'class_{label}')
            text = f"ID:{track_id} {class_name} {score:.2f}"
            cv2.putText(frame_display, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(
            frame_display, 
            f"Frame: {frame_idx} - Tracks: {len(current_frame_tracks)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Convert back to BGR for display
        frame_display_bgr = cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR)
        
        # Display the frame
        #cv2.imshow(f'Tracking - {source_name}', frame_display_bgr)
        
        # Write frame if saving video
        if out is not None:
            out.write(frame_display_bgr)

        with open(output_file, "a") as f:  # Append mode instead of write
            for track in current_frame_tracks:
                box = track['box']
                track_id = track['id']
                score = track['score']
                label = track['label']

                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                line = f"{frame_idx + 1},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1"
                f.write(line + "\n")
        
        # Increment frame counter
        frame_idx += 1
        
        # Break on 'q' key press
        """if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted processing")
            break"""
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    #cv2.destroyAllWindows()
    
    print(f"Processed {frame_idx} frames from {source_name}")

def main():
    args = parse_args()
    
    # Set device
    if not args.device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    print("Creating hybrid model...")
    model = HybridDetectionTrackingModel(
        pretrained=True,
        num_classes=91,  # COCO dataset classes
        feature_dim=256,  # Match the dimensions used during training
        reid_dim=128,    # Match the dimensions used during training
        hidden_dim=256,
        num_gnn_layers=args.num_gnn_layers
    )
    
    # Set the similarity threshold for the GNN tracker
    if hasattr(model.gnn, 'similarity_threshold'):
        model.gnn.similarity_threshold = args.similarity_threshold
        print(f"Set GNN similarity threshold to {args.similarity_threshold}")
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("Hybrid model initialized")
    
    # Load checkpoint if available (without the detailed error output)
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"Successfully loaded checkpoint")
            else:
                print("Warning: checkpoint does not contain 'model_state_dict'")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"No checkpoint found at {args.checkpoint}, using untrained model")
    
    # Check if we're using webcam or video file input
    if args.webcam is not None or args.video is not None:
        process_video_input(args, model, device)
        return
    
    # Load sequence
    if args.jijel:
        print(f"Loading Jijel image sequence from data/jijel...")
        # Get sorted list of PNG files in data/jijel
        image_files = sorted(glob.glob('data/jijel/*.png'), 
                            key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        num_frames = len(image_files)
        if args.max_frames:
            num_frames = min(num_frames, args.max_frames)
            image_files = image_files[:num_frames]
        
        print(f"Loaded {num_frames} frames from Jijel sequence")
        
        # Load first frame to get image dimensions
        sample_img = cv2.imread(image_files[0])
        sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        
        # Resize if requested
        original_height, original_width = sample_img.shape[:2]
        if args.resize:
            # Calculate new height maintaining aspect ratio
            aspect_ratio = original_height / original_width
            new_width = args.resize
            new_height = int(new_width * aspect_ratio)
            print(f"Resizing images from {original_width}x{original_height} to {new_width}x{new_height}")
            
            # Resize sample image for visualization setup
            sample_img = cv2.resize(sample_img, (new_width, new_height))
        
        height, width, _ = sample_img.shape
        print(f"Processing images at dimensions: {width}x{height}")
        
        # Create a custom sequence structure similar to MOT16Sequences
        sequence = []
        for img_path in image_files:
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if requested
            if args.resize:
                img = cv2.resize(img, (new_width, new_height))
            
            # Convert to tensor (scale to [0,1] and convert to CxHxW format)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            
            # Create frame dict similar to MOT16Sequences
            frame = {
                'img': img_tensor,
                'gt': {},  # No ground truth for Jijel sequence
                'img_path': img_path
            }
            sequence.append(frame)
    else:
        print(f"Loading sequence {args.seq_name}...")
        sequence = MOT16Sequences(args.seq_name, args.data_dir, load_seg=True)[0]
        num_frames = len(sequence)
        if args.max_frames:
            num_frames = min(num_frames, args.max_frames)
        print(f"Loaded {num_frames} frames from sequence {args.seq_name}")
    
    # Initialize first frame for visualization setup
    first_frame = sequence[0]
    img = first_frame['img'].mul(255).permute(1, 2, 0).byte().numpy()
    height, width, _ = img.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Suppress warnings to reduce output noise
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create visualization object with specified dimensions
    print(f"Creating visualization at {width}x{height} resolution")
    viz = Visualizer(width, height)
    viz.init_display(img)
    viz.add_legend()
    print("Visualizer initialized")
    
    # Initialize track management
    all_tracks = {}  # Dictionary to store all tracks: {track_id: track_info}
    next_track_id = 1  # Counter for assigning new track IDs
    
    # Store detection counts for analysis
    detection_counts = []
    person_counts = []
    
    print("\n--- FRAME-BY-FRAME ANALYSIS ---")
    print("Frame | Total Detections | Person Detections | Notes")
    print("-" * 60)
    
    # Pre-process all frames to avoid multiple rendering of the same frame
    all_frames_data = []
    
    # Store previous state for tracking
    prev_boxes = None
    prev_features = None
    prev_track_ids = None
    active_tracks = []  # List of active tracks from previous frame
    
    print("Pre-processing all frames...")
    for frame_idx in range(num_frames):
        try:
            # Get the frame data
            frame = sequence[frame_idx]
            frame_img = frame['img'].mul(255).permute(1, 2, 0).byte().numpy()
            
            # Process frame
            with torch.no_grad():
                input_tensor = frame['img'].to(device).unsqueeze(0)  # Add batch dimension
                
                # Create edge index from previous boxes if available
                edge_index = None
                if prev_boxes is not None and prev_features is not None:
                    edge_index = create_edge_index(prev_boxes, device=device)
                
                # Forward pass
                outputs = model(
                    input_tensor, 
                    prev_features=prev_features, 
                    prev_boxes=prev_boxes,
                    edge_index=edge_index,
                    prev_track_ids=prev_track_ids
                )
                
            # Get detections and ReID features
            detector_outputs = outputs['detector_outputs']
            boxes = detector_outputs['boxes']
            scores = detector_outputs['scores']
            labels = detector_outputs['labels']
            reid_features = outputs['reid_features']
            tracking_info = outputs['tracking_info']
            
            # Save raw detection counts
            raw_total = len(boxes)
            raw_persons = (labels == 1).sum().item() if len(labels) > 0 else 0
            
            # Print raw detection count before filtering
            print(f"Frame {frame_idx}: Raw detection count: {raw_total} (before threshold)")
            
            # Process detections
            current_frame_tracks = []
            if len(boxes) > 0:
                # Apply detection threshold here for consistency
                detection_mask = scores >= args.detection_threshold
                
                # Check if we have valid detections after thresholding
                if detection_mask.sum() > 0:
                    # Filter detections
                    all_boxes = boxes[detection_mask]
                    all_scores = scores[detection_mask]
                    all_labels = labels[detection_mask]
                    all_reid_features = reid_features[detection_mask] if reid_features is not None else torch.zeros((detection_mask.sum().item(), 128), device=device)
                    
                    # Apply persons only filter if specified
                    if args.persons_only:
                        persons_mask = all_labels == 1  # Person class is 1 in COCO
                        if persons_mask.sum() > 0:  # If we have any persons
                            all_boxes = all_boxes[persons_mask]
                            all_scores = all_scores[persons_mask]
                            all_labels = all_labels[persons_mask]
                            all_reid_features = all_reid_features[persons_mask] if all_reid_features is not None else torch.zeros((persons_mask.sum().item(), 128), device=device)
                            print(f"Filtered to show only {persons_mask.sum().item()} person detections out of {len(persons_mask)} total")
                        else:
                            print(f"No person detections found in this frame!")
                    # Apply MOT classes filter if specified
                    elif args.mot_classes:
                        # MOT Challenge classes: person (1), car (3), motorcycle (4), bicycle (2)
                        mot_classes = [1, 3, 4, 2]  # COCO classes for MOT Challenge categories
                        mot_mask = torch.zeros_like(all_labels, dtype=torch.bool)
                        for class_id in mot_classes:
                            mot_mask = mot_mask | (all_labels == class_id)
                        
                        if mot_mask.sum() > 0:  # If we have any MOT objects
                            all_boxes = all_boxes[mot_mask]
                            all_scores = all_scores[mot_mask]
                            all_labels = all_labels[mot_mask]
                            all_reid_features = all_reid_features[mot_mask] if all_reid_features is not None else torch.zeros((mot_mask.sum().item(), 128), device=device)
                            
                            # Count per class for logging
                            class_counts = {}
                            for class_id in mot_classes:
                                class_counts[class_id] = (all_labels == class_id).sum().item()
                            
                            print(f"Filtered to show {mot_mask.sum().item()} MOT challenge objects: " + 
                                  f"Persons: {class_counts[1]}, Cars: {class_counts[3]}, " +
                                  f"Motorcycles: {class_counts[4]}, Bicycles: {class_counts[2]}")
                        else:
                            print(f"No MOT challenge objects found in this frame!")
                    # Custom class ids if specified
                    elif args.class_ids != '1,3,4,2':
                        # Parse comma-separated class IDs
                        custom_classes = [int(class_id) for class_id in args.class_ids.split(',')]
                        custom_mask = torch.zeros_like(all_labels, dtype=torch.bool)
                        for class_id in custom_classes:
                            custom_mask = custom_mask | (all_labels == class_id)
                        
                        if custom_mask.sum() > 0:  # If we have any custom objects
                            all_boxes = all_boxes[custom_mask]
                            all_scores = all_scores[custom_mask]
                            all_labels = all_labels[custom_mask]
                            all_reid_features = all_reid_features[custom_mask] if all_reid_features is not None else torch.zeros((custom_mask.sum().item(), 128), device=device)
                            print(f"Filtered to show {custom_mask.sum().item()} objects with class IDs: {args.class_ids}")
                        else:
                            print(f"No objects with specified class IDs found in this frame!")
                    
                    # Count filtered detections
                    total_dets = len(all_boxes)
                    person_dets = (all_labels == 1).sum().item()
                    
                    # Store counts for analysis
                    detection_counts.append(total_dets)
                    person_counts.append(person_dets)
                    
                    # Print frame-by-frame detection info
                    notes = ""
                    if len(detection_counts) > 1 and abs(detection_counts[-1] - detection_counts[-2]) > 10:
                        notes = f"⚠️ BIG DROP from previous frame ({detection_counts[-2]})"
                    
                    print(f"{frame_idx:5d} | {total_dets:16d} | {person_dets:17d} | {notes}")
                    
                    # First frame: initialize tracks
                    if frame_idx == 0 or len(active_tracks) == 0:
                        for i in range(total_dets):
                            box = all_boxes[i].cpu().numpy()
                            feature = all_reid_features[i].detach().cpu()
                            score = all_scores[i].item()
                            label = all_labels[i].item()
                            
                            # Calculate center point
                            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                            
                            # Create new track
                            track = {
                                'id': next_track_id,
                                'box': box,
                                'score': score,
                                'feature': feature,
                                'position': np.append(center, [0]),  # Add z=0
                                'age': 1,
                                'time_since_update': 0,
                                'label': label
                            }
                            
                            # Add to current frame and global track storage
                            current_frame_tracks.append(track)
                            all_tracks[next_track_id] = track
                            next_track_id += 1
                            
                        print(f"Frame {frame_idx}: Created {len(current_frame_tracks)} new tracks")
                    else:
                        # Association with previous tracks
                        assignments, unmatched_tracks, unmatched_detections = associate_detections_to_tracks(
                            active_tracks, 
                            all_boxes,
                            all_reid_features,
                            iou_threshold=args.iou_threshold,
                            device=device
                        )
                        
                        print(f"Frame {frame_idx}: Matched {len(assignments)} tracks, {len(unmatched_tracks)} lost, {len(unmatched_detections)} new")
                        
                        # Update matched tracks
                        for track_idx, det_idx in assignments:
                            # Get previous track
                            prev_track = active_tracks[track_idx]
                            track_id = prev_track['id']
                            
                            # Update with new detection
                            box = all_boxes[det_idx].cpu().numpy()
                            feature = all_reid_features[det_idx].detach().cpu()
                            score = all_scores[det_idx].item()
                            label = all_labels[det_idx].item()
                            
                            # Calculate center point
                            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                            
                            # Update track
                            updated_track = {
                                'id': track_id,
                                'box': box,
                                'score': score,
                                'feature': feature,  # Use new feature
                                'position': np.append(center, [0]),
                                'age': prev_track['age'] + 1,
                                'time_since_update': 0,
                                'label': label
                            }
                            
                            # Add to current frame and update global track storage
                            current_frame_tracks.append(updated_track)
                            all_tracks[track_id] = updated_track
                        
                        # Add new tracks for unmatched detections
                        for det_idx in unmatched_detections:
                            box = all_boxes[det_idx].cpu().numpy()
                            feature = all_reid_features[det_idx].detach().cpu()
                            score = all_scores[det_idx].item()
                            label = all_labels[det_idx].item()
                            
                            # Calculate center point
                            center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                            
                            # Create new track
                            track = {
                                'id': next_track_id,
                                'box': box,
                                'score': score, 
                                'feature': feature,
                                'position': np.append(center, [0]),
                                'age': 1,
                                'time_since_update': 0,
                                'label': label
                            }
                            
                            # Add to current frame and global track storage
                            current_frame_tracks.append(track)
                            all_tracks[next_track_id] = track
                            next_track_id += 1
                        
                    # Update state for next frame
                    active_tracks = current_frame_tracks.copy()
                    
                    # Extract features and boxes for next frame
                    prev_boxes = all_boxes
                    prev_features = all_reid_features
                    prev_track_ids = [track['id'] for track in current_frame_tracks]
                else:
                    # No detections above threshold
                    detection_counts.append(0)
                    person_counts.append(0)
                    print(f"{frame_idx:5d} | {0:16d} | {0:17d} | No detections above threshold")
            else:
                # No detections at all
                detection_counts.append(0)
                person_counts.append(0)
                print(f"{frame_idx:5d} | {0:16d} | {0:17d} | No detections from model")
                
            # Store frame data for animation
            all_frames_data.append({
                'img': frame_img,
                'tracks': current_frame_tracks,
                'frame_idx': frame_idx
            })
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            traceback.print_exc()
            # Add empty frame data as fallback
            all_frames_data.append({
                'img': frame_img if 'frame_img' in locals() else np.zeros((height, width, 3), dtype=np.uint8),
                'tracks': [],
                'frame_idx': frame_idx
            })
    
    print(f"Pre-processing complete. Created data for {len(all_frames_data)} frames.")
    
    # Create animation update function that uses pre-processed data
    def update(frame_idx):
        try:
            # Get pre-processed frame data
            frame_data = all_frames_data[frame_idx]
            
            # Update image display
            viz.image_display.set_data(frame_data['img'])
            
            # Clear old visualization
            viz.clear_boxes()
            
            # Display tracks
            viz.display_tracks(frame_data['tracks'])
            
            # Add frame number text
            frame_text = viz.ax.text(
                10, 30, 
                f"Frame: {frame_data['frame_idx']} - Tracks: {len(frame_data['tracks'])}" + 
                (f" (Persons Only)" if args.persons_only else 
                 f" (MOT Challenge)" if args.mot_classes else
                 f" (Classes: {args.class_ids})" if args.class_ids != '1,3,4,2' else ""), 
                fontsize=12, 
                color='white', 
                backgroundcolor='black',
                bbox=dict(boxstyle="round,pad=0.3", fc='black', ec="white", alpha=0.7)
            )
            viz.texts.append(frame_text)
            
            # Return all artists that need to be redrawn
            artists = [viz.image_display] + viz.rects + viz.texts + viz.arrows
            return artists
        except Exception as e:
            print(f"Error in animation update for frame {frame_idx}: {e}")
            traceback.print_exc()
            # Return just the image to keep the animation running
            return [viz.image_display]
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(
        viz.fig, update, frames=range(len(all_frames_data)),
        interval=100, blit=True
    )
    
    # Save video if requested
    if args.save_video:
        print(f"Saving video to {args.output}...")
        try:
            # Try with ffmpeg first
            anim.save(args.output, fps=10, extra_args=['-vcodec', 'libx264'])
        except TypeError:
            # Fall back to pillow if ffmpeg is not available
            print("Falling back to Pillow for video saving (no extra codec options)")
            anim.save(args.output, fps=10)
        print(f"Video saved to {args.output}")
    
    # Show the animation
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc() 