import torch
import torch.nn.functional as F
import numpy as np

def create_edge_index(boxes, max_distance=0.5, device=None):
    """Create edge indices for objects based on spatial proximity"""
    if device is None:
        device = boxes.device
    
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
    
    # Calculate pairwise distance - vectorized version
    src_ids, dst_ids = [], []
    
    # More efficient pairwise distance calculation
    for i in range(n):
        # Calculate distances from node i to all other nodes
        dists = torch.sqrt(torch.sum((centers[i].unsqueeze(0) - centers)**2, dim=1))
        # Find indices where distance is less than threshold and not self
        valid_edges = torch.nonzero((dists < max_distance) & (torch.arange(n, device=device) != i), as_tuple=True)[0]
        
        if len(valid_edges) > 0:
            src_ids.extend([i] * len(valid_edges))
            dst_ids.extend(valid_edges.tolist())
    
    # Create edge index tensor
    if not src_ids:  # No edges created
        return torch.zeros((2, 0), dtype=torch.long, device=device)
        
    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long, device=device)
    return edge_index

def create_temporal_edge_index(prev_features, curr_features, similarity_threshold=0.5, device=None):
    """
    Create edge indices between previous and current frame objects based on feature similarity.
    
    This function creates edges by comparing feature similarities between nodes in previous frames
    and current frames, rather than using spatial proximity.
    
    Args:
        prev_features: Features of objects from previous frame [num_prev, feature_dim]
        curr_features: Features of objects from current frame [num_curr, feature_dim]
        similarity_threshold: Threshold for cosine similarity to create an edge (default: 0.5)
        device: Device to use for computation
        
    Returns:
        edge_index: Tensor of edge indices [2, E] connecting nodes between frames
    """
    if device is None:
        device = prev_features.device
    
    num_prev = prev_features.shape[0]
    num_curr = curr_features.shape[0]
    
    # If either frame has no objects, return empty edge index
    if num_prev == 0 or num_curr == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # Normalize features for cosine similarity
    prev_features_norm = F.normalize(prev_features, p=2, dim=1)
    curr_features_norm = F.normalize(curr_features, p=2, dim=1)
    
    # Compute pairwise cosine similarity between all previous and current features
    # Shape: [num_prev, num_curr]
    similarity_matrix = torch.mm(prev_features_norm, curr_features_norm.t())
    
    # Initialize edge lists
    src_ids = []
    dst_ids = []
    
    # For each previous object
    for i in range(num_prev):
        # Find current objects with similarity above threshold
        high_similarity_indices = torch.nonzero(similarity_matrix[i] >= similarity_threshold, as_tuple=True)[0]
        
        if len(high_similarity_indices) > 0:
            # Current indices need to be offset by num_prev for the combined graph
            dst_indices = high_similarity_indices + num_prev
            
            # Add edges from previous object to similar current objects
            src_ids.extend([i] * len(dst_indices))
            dst_ids.extend(dst_indices.tolist())
    
    # Create edge index tensor
    if not src_ids:  # No edges created
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long, device=device)
    return edge_index

def create_comprehensive_edge_index(prev_features, curr_features, prev_boxes=None, curr_boxes=None, 
                                  similarity_threshold=0.5, spatial_threshold=0.5, device=None):
    """
    Create a comprehensive edge index that includes both:
    1. Spatial edges within each frame (based on box distances)
    2. Temporal edges between frames (based on feature similarities)
    
    Args:
        prev_features: Features of objects from previous frame [num_prev, feature_dim]
        curr_features: Features of objects from current frame [num_curr, feature_dim]
        prev_boxes: Bounding boxes from previous frame [num_prev, 4]
        curr_boxes: Bounding boxes from current frame [num_curr, 4]
        similarity_threshold: Threshold for feature similarity (default: 0.5)
        spatial_threshold: Threshold for spatial proximity (default: 0.5)
        device: Device to use for computation
        
    Returns:
        edge_index: Combined edge indices [2, E]
    """
    if device is None:
        if prev_features is not None:
            device = prev_features.device
        elif prev_boxes is not None:
            device = prev_boxes.device
        else:
            device = torch.device('cpu')
    
    num_prev = prev_features.shape[0] if prev_features is not None else 0
    num_curr = curr_features.shape[0] if curr_features is not None else 0
    
    # If we have no objects, return empty edge index
    if num_prev == 0 and num_curr == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # 1. Create temporal edges based on feature similarity
    temporal_edges = create_temporal_edge_index(
        prev_features, curr_features, 
        similarity_threshold=similarity_threshold, 
        device=device
    )
    
    # 2. Create spatial edges within previous frame if we have prev_boxes
    prev_spatial_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
    if prev_boxes is not None and prev_boxes.shape[0] > 1:
        prev_spatial_edges = create_edge_index(
            prev_boxes, 
            max_distance=spatial_threshold, 
            device=device
        )
    
    # 3. Create spatial edges within current frame if we have curr_boxes
    curr_spatial_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
    if curr_boxes is not None and curr_boxes.shape[0] > 1:
        curr_spatial_edges = create_edge_index(
            curr_boxes, 
            max_distance=spatial_threshold, 
            device=device
        )
        # Offset current frame indices by num_prev
        if curr_spatial_edges.shape[1] > 0:
            curr_spatial_edges = curr_spatial_edges + torch.tensor([[num_prev], [num_prev]], device=device)
    
    # 4. Combine all edge indices
    edge_indices = [temporal_edges, prev_spatial_edges, curr_spatial_edges]
    edge_indices = [e for e in edge_indices if e.shape[1] > 0]  # Filter out empty edge indices
    
    if not edge_indices:  # If all edge indices are empty
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    
    combined_edges = torch.cat(edge_indices, dim=1)
    
    return combined_edges

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # Get the coordinates of bounding boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def match_detections_with_gt(detections, gt_boxes, gt_ids, iou_threshold=0.5):
    """Match detections with ground truth boxes using IoU"""
    matched_ids = [-1] * len(detections)
    id_matches = {}
    
    for det_idx, det_box in enumerate(detections):
        best_iou = iou_threshold
        best_gt_id = -1
        
        for gt_id, gt_box in gt_boxes.items():
            iou = calculate_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_gt_id != -1:
            matched_ids[det_idx] = gt_ids[best_gt_id]
            id_matches[det_idx] = best_gt_id
    
    return matched_ids, id_matches

def compute_tracking_metrics(prev_det_gt_matches, curr_det_gt_matches, prev_track_ids, curr_track_ids):
    """Compute tracking metrics including identity switches"""
    id_switches = 0
    id_switch_pairs = []
    
    # Build a mapping from GT IDs to track IDs in previous frame
    gt_to_track = {}
    for det_idx, gt_id in prev_det_gt_matches.items():
        if gt_id != -1 and det_idx < len(prev_track_ids):
            track_id = prev_track_ids[det_idx]
            if track_id != -1:
                gt_to_track[gt_id] = track_id
    
    # Check for ID switches in current frame
    for det_idx, gt_id in curr_det_gt_matches.items():
        if gt_id != -1 and gt_id in gt_to_track and det_idx < len(curr_track_ids):
            prev_track_id = gt_to_track[gt_id]
            curr_track_id = curr_track_ids[det_idx]
            
            # Check if the same GT object has a different track ID now
            if curr_track_id != -1 and prev_track_id != curr_track_id:
                id_switches += 1
                id_switch_pairs.append((prev_track_id, curr_track_id, gt_id))
    
    return id_switches, id_switch_pairs

def compute_tracking_loss(id_switch_pairs, node_embeddings, num_prev_nodes):
    """Compute tracking loss based on identity switches"""
    device = node_embeddings.device if node_embeddings is not None else 'cpu'
    
    if not id_switch_pairs or node_embeddings is None:
        return torch.tensor(0.0, device=device).reshape(1)
    
    tracking_loss = 0
    
    for prev_id, curr_id, gt_id in id_switch_pairs:
        # Find the nodes in the GNN that correspond to these track IDs
        prev_node_idx = None
        curr_node_idx = None
        
        # Simple index mapping
        for i in range(num_prev_nodes):
            if prev_id == i:  
                prev_node_idx = i
                break
        
        for i in range(node_embeddings.shape[0] - num_prev_nodes):
            if curr_id == i:  
                curr_node_idx = num_prev_nodes + i
                break
        
        if prev_node_idx is not None and curr_node_idx is not None:
            # Get the embeddings
            prev_emb = node_embeddings[prev_node_idx]
            curr_emb = node_embeddings[curr_node_idx]
            
            # Compute similarity loss
            similarity_loss = 1.0 - F.cosine_similarity(prev_emb.unsqueeze(0), curr_emb.unsqueeze(0))
            tracking_loss += similarity_loss
    
    if tracking_loss > 0:
        tracking_loss = tracking_loss / len(id_switch_pairs)  # Average loss
    
    # Ensure the loss is a scalar tensor with shape [1] for proper broadcasting
    if isinstance(tracking_loss, torch.Tensor) and tracking_loss.dim() == 0:
        tracking_loss = tracking_loss.reshape(1)
    
    return tracking_loss

def extract_loss(val):
    """Extract numeric loss value from any data structure"""
    if val is None:
        return 0.0
    elif isinstance(val, (list, tuple)):
        return sum(extract_loss(v) for v in val)
    elif isinstance(val, dict):
        return sum(extract_loss(v) for v in val.values())
    elif isinstance(val, torch.Tensor):
        if val.numel() == 0:  
            return 0.0
        elif val.numel() == 1:  
            return val.item()
        else:  
            return val.mean().item()
    elif isinstance(val, (int, float)):
        return val
    else:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

def tensor_to_python(obj):
    """Recursively converts tensor values to Python native types for JSON serialization"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_python(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

def collate_fn(batch):
    """Custom collate function that simply returns the first item in the batch"""
    return batch[0] 