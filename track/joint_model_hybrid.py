import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import roi_align
from track.utils import create_edge_index, create_temporal_edge_index, create_comprehensive_edge_index
import torch_geometric
from torch_geometric.nn import GCNConv as PyGGCNConv
from scipy.optimize import linear_sum_assignment

class DetectionWithReID(nn.Module):
    """
    Simple model using pre-trained Faster R-CNN with appearance feature extraction for ReID
    """
    def __init__(self, 
                 pretrained=True,
                 num_classes=91, 
                 feature_dim=512,
                 reid_dim=256):
        super(DetectionWithReID, self).__init__()
        
        # Create pre-trained Faster R-CNN detector
        self.detector = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        
        # Feature dimensions
        self.feature_dim = feature_dim
        self.reid_dim = reid_dim
        
        # ReID embedding for appearance feature extraction
        self.feature_projection = nn.Linear(256, feature_dim)
        self.reid_head = nn.Sequential(
            nn.Linear(feature_dim, reid_dim),
            nn.ReLU(),
            nn.Linear(reid_dim, reid_dim),
            nn.BatchNorm1d(reid_dim)
        )
        
    def forward(self, images):
        # Ensure the detector is in eval mode for inference
        self.detector.eval()
        
        # Process the image with the detector
        with torch.no_grad():
            detector_outputs = self.detector(images)
        
        # Get all detections
        all_boxes = [output['boxes'] for output in detector_outputs]
        all_scores = [output['scores'] for output in detector_outputs]
        all_labels = [output['labels'] for output in detector_outputs]
        
        # For each image, filter detections based on confidence
        filtered_boxes, filtered_scores, filtered_labels, filtered_reid_features = [], [], [], []
        
        for i, (boxes, scores, labels) in enumerate(zip(all_boxes, all_scores, all_labels)):
            # Apply a lower score threshold for more consistent detections
            score_filter = scores >= 0.3
            
            if score_filter.sum() > 0:
                # We have some detections above threshold
                filtered_boxes.append(boxes[score_filter])
                filtered_scores.append(scores[score_filter])
                filtered_labels.append(labels[score_filter])
                
                # Extract features for these detections
                if boxes.shape[0] > 0:
                    # Get backbone features
                    features = self.detector.backbone(images[i:i+1])
                    
                    # Apply ROI align to get fixed-size feature maps for each detection
                    box_features = self.detector.roi_heads.box_roi_pool(
                        features, [boxes[score_filter]], [images[i].shape[-2:]]
                    )
                    
                    # Global average pooling
                    box_features_flat = torch.mean(box_features, dim=[2, 3])
                    
                    # Project features
                    box_features_flat = self.feature_projection(box_features_flat)
                    
                    # Get ReID embeddings
                    reid_features = self.reid_head(box_features_flat)
                    
                    # L2 normalize features
                    reid_features = F.normalize(reid_features, p=2, dim=1)
                    
                    filtered_reid_features.append(reid_features)
                else:
                    filtered_reid_features.append(torch.empty((0, self.reid_dim), device=boxes.device))
            else:
                # No detections above threshold, append empty tensors
                filtered_boxes.append(torch.empty((0, 4), device=boxes.device))
                filtered_scores.append(torch.empty(0, device=scores.device))
                filtered_labels.append(torch.empty(0, dtype=torch.long, device=labels.device))
                filtered_reid_features.append(torch.empty((0, self.reid_dim), device=boxes.device))
        
        # Return detection results and ReID features
        results = []
        for i in range(len(detector_outputs)):
            result = {
                'detector_outputs': {
                    'boxes': filtered_boxes[i],
                    'scores': filtered_scores[i],
                    'labels': filtered_labels[i]
                },
                'reid_features': filtered_reid_features[i]
            }
            results.append(result)
        
        return results

class GraphNN(nn.Module):
    """
    Graph Neural Network for tracking using GCN to model spatial-temporal relationships
    """
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2):
        super(GraphNN, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature projection to ensure consistent dimensions
        self.feature_projection = nn.Linear(256, 128)
        
        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # GCN layers - replaces message/update MLPs with GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Temporal attention module for cross-frame matching
        # This computes attention weights for edges connecting objects across different frames
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),    # Concatenated features from both frames
            nn.BatchNorm1d(hidden_dim),               # Normalization for training stability
            nn.ReLU(),
            nn.Dropout(0.2),                          # Regularization to prevent overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),   # Reduce dimension
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),            # Single attention score
            nn.Sigmoid()                              # Scale to [0,1]
        )
        
        # Edge predictor module
        # Predicts connection strength between nodes based on their features
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated node features -> hidden layer
            nn.BatchNorm1d(hidden_dim),             # Batch normalization for stable training
            nn.ReLU(),
            nn.Dropout(0.2),                        # Increased dropout for better generalization
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),          # Final prediction of edge strength
            nn.Sigmoid()                            # Sigmoid to get probability between 0 and 1
        )
        
        # ID classifier for track assignment
        self.id_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        # Track memory
        self.id_counter = 1
        self.known_tracks = {}
        self.similarity_threshold = 0.85
    
    def forward(self, node_features, edge_index, prev_track_ids=None):
        """
        Forward pass of GCN-based tracking with improved temporal edge handling
        
        Args:
            node_features: Features of detected objects
            edge_index: Connectivity between objects [2, num_edges]
            prev_track_ids: Previously assigned track IDs
            
        Returns:
            x: Updated node embeddings
            edge_weights: Predicted edge weights
            track_ids: Updated track IDs
        """
        # Handle dimension mismatch
        if node_features.size(1) != 128:
            node_features = self.feature_projection(node_features)
        
        # Initial node encoding
        x = self.node_encoder(node_features)
        previous_x = x  # Store for skip connection
        
        # Initialize variables to avoid undefined references later
        temporal_attention_weights = None
        temporal_edge_mask = None
        
        # Process edge connections (both spatial and temporal)
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            num_nodes = x.size(0)
            
            # Determine number of nodes from previous and current frames
            if prev_track_ids is not None:
                num_prev_nodes = len(prev_track_ids)
            else:
                # If no previous track information, estimate based on node count
                # Assuming roughly half are from previous frame
                num_prev_nodes = num_nodes // 2
            
            # Identify temporal edges (connections between frames)
            # A temporal edge connects a node from the previous frame to a node in the current frame
            is_temporal_edge = (src < num_prev_nodes) & (dst >= num_prev_nodes)
            is_reverse_temporal = (src >= num_prev_nodes) & (dst < num_prev_nodes)
            temporal_edge_mask = is_temporal_edge | is_reverse_temporal
            
            # Apply temporal attention to the edges between frames
            if torch.any(temporal_edge_mask):
                temporal_edges = edge_index[:, temporal_edge_mask]
                t_src, t_dst = temporal_edges
                
                # Get node features for temporal connections
                t_src_features = x[t_src]
                t_dst_features = x[t_dst]
                
                # Compute attention weights for temporal edges based on feature similarity
                temporal_features = torch.cat([t_src_features, t_dst_features], dim=1)
                temporal_attention_weights = self.temporal_attention(temporal_features).squeeze(-1)
        
        # GCN message passing for spatial-temporal modeling
        for layer_idx in range(self.num_layers):
            if edge_index.shape[1] > 0:  # If there are edges
                # Apply GCN convolution
                gcn_out = self.gcn_layers[layer_idx](x, edge_index)
                
                # Skip connection - add input from previous layer
                x = gcn_out + previous_x
                previous_x = x
        
        # Compute edge weights for association
        edge_weights = None
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            src_features = x[src]
            dst_features = x[dst]
            edge_features = torch.cat([src_features, dst_features], dim=1)
            edge_weights = self.edge_predictor(edge_features).squeeze(-1)
            
            # Apply temporal attention to edges between frames if available
            if temporal_attention_weights is not None and temporal_edge_mask is not None:
                if torch.any(temporal_edge_mask):
                    # Create a new tensor instead of modifying in-place
                    weighted_edge_weights = edge_weights.clone()
                    
                    # Get indices of temporal edges
                    temporal_mask_indices = torch.where(temporal_edge_mask)[0]
                    
                    # Apply a stronger weight boost for temporal connections
                    # The formula below puts more emphasis on high-confidence temporal connections
                    # while downplaying low-confidence ones
                    boost_factor = 1.5  # Increase importance of temporal connections
                    
                    # Boost weights based on feature similarity attention
                    # Higher attention values get boosted more (quadratic relationship)
                    weighted_edge_weights[temporal_mask_indices] = edge_weights[temporal_mask_indices] * \
                                                                (1.0 + boost_factor * temporal_attention_weights**2)
                    
                    # Replace old tensor with new one
                    edge_weights = weighted_edge_weights
        
        # Get identity features
        id_features = self.id_classifier(x)
        
        # Calculate track assignments
        track_ids = self.assign_track_ids(id_features, prev_track_ids)
        
        return x, edge_weights, track_ids
    
    def assign_track_ids(self, id_features, prev_track_ids=None):
        """Assign track IDs based on feature similarity using Hungarian algorithm"""
        # Normalize features
        normalized_features = F.normalize(id_features, p=2, dim=1)
        num_detections = normalized_features.shape[0]
        
        # Initialize track IDs
        track_ids = [-1] * num_detections
        
        # If no known tracks, assign new IDs to all
        if not self.known_tracks:
            for i in range(num_detections):
                track_ids[i] = self.id_counter
                self.known_tracks[self.id_counter] = normalized_features[i].detach()
                self.id_counter += 1
            return track_ids
        
        # Get known tracks
        known_ids = list(self.known_tracks.keys())
        known_features = torch.stack([self.known_tracks[id] for id in known_ids])
        
        # Compute similarity matrix between current features and known tracks
        similarity_matrix = torch.mm(normalized_features, known_features.t())
        
        # Convert to cost matrix (Hungarian algorithm minimizes cost)
        # FIX: Add detach() before numpy conversion
        cost_matrix = (1.0 - similarity_matrix).detach().cpu().numpy()
        
        # Apply Hungarian algorithm to find optimal assignment
        # Only run Hungarian if we have both detections and known tracks
        if num_detections > 0 and len(known_ids) > 0:
            # Get optimal assignment
            det_indices, track_indices = linear_sum_assignment(cost_matrix)
            
            # Initialize assigned flags
            assigned = [False] * num_detections
            
            # Process assignments
            for det_idx, track_idx in zip(det_indices, track_indices):
                # Only assign if similarity is above threshold
                similarity = similarity_matrix[det_idx, track_idx].item()
                
                if similarity > self.similarity_threshold:
                    track_id = known_ids[track_idx]
                    track_ids[det_idx] = track_id
                    assigned[det_idx] = True
                    
                    # Update track feature with exponential moving average
                    self.known_tracks[track_id] = 0.7 * self.known_tracks[track_id] + 0.3 * normalized_features[det_idx].detach()
                    self.known_tracks[track_id] = F.normalize(self.known_tracks[track_id], p=2, dim=0)
            
            # Assign new IDs to unmatched detections
            for i in range(num_detections):
                if not assigned[i]:
                    track_ids[i] = self.id_counter
                    self.known_tracks[self.id_counter] = normalized_features[i].detach()
                    self.id_counter += 1
        else:
            # If either is empty, assign new IDs to all detections
            for i in range(num_detections):
                track_ids[i] = self.id_counter
                self.known_tracks[self.id_counter] = normalized_features[i].detach()
                self.id_counter += 1
        
        # Remove old tracks
        if len(self.known_tracks) > 100:
            to_remove = [id for id in self.known_tracks.keys() if id not in track_ids]
            for id in to_remove[:len(to_remove)//2]:
                if id in self.known_tracks:
                    del self.known_tracks[id]
        
        return track_ids
    
    def reset_memory(self):
        """Reset the tracker memory"""
        self.id_counter = 1
        self.known_tracks = {}

class ChannelAttentionModule(nn.Module):
    """
    Channel Attention Module as shown in architecture diagram.
    Generates channel attention map MC which highlights important channels.
    """
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttentionModule, self).__init__()
        # Global pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling
        
        # Shared MLP for dimensionality reduction and feature extraction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Global average pooling and max pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)
        
        # Process through the shared FC layers
        avg_out = self.fc(avg_out).view(batch_size, channels, 1, 1)
        max_out = self.fc(max_out).view(batch_size, channels, 1, 1)
        
        # Generate channel attention map MC = σ(MLP(AvgPool(X)) + MLP(MaxPool(X)))
        MC = self.sigmoid(avg_out + max_out)
        
        # Return MC directly without applying it to the input
        # This follows the architecture where MC is applied separately
        return MC

class SpatialAttentionModule(nn.Module):
    """
    https://www.mdpi.com/2072-4292/12/1/188
    """
    def __init__(self, in_channels, feature_dim):
        super(SpatialAttentionModule, self).__init__()
        
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

class HybridDetectionTrackingModel(nn.Module):
    """
    Hybrid model that uses DetectionWithReID for detection and adds GNN-based tracking
    """
    def __init__(self, 
                 pretrained=True,
                 num_classes=91,  # COCO has 91 classes
                 feature_dim=512,  
                 reid_dim=256,     
                 hidden_dim=256,
                 num_gnn_layers=2):
        super(HybridDetectionTrackingModel, self).__init__()
        
        # Use the original detector with ReID
        self.detector = DetectionWithReID(
            pretrained=pretrained,
            num_classes=num_classes,
            feature_dim=feature_dim,
            reid_dim=reid_dim
        )
        
        # Feature dimensions
        self.feature_dim = reid_dim  # Use ReID features from detector
        self.hidden_dim = hidden_dim
        
        # Add spatial attention module
        # We'll use this to refine the ReID features from the original detector
        self.enhanced_reid = SpatialAttentionModule(
            in_channels=256,  # Assuming detector provides 256 channel features
            feature_dim=reid_dim
        )
        
        # Graph neural network for tracking
        self.gnn = GraphNN(
            feature_dim=reid_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers
        )
        
        # Association prediction head - exactly match checkpoint dimensions
        # The checkpoint expects shape [256, 512] for first layer
        self.association_head = nn.Sequential(
            nn.Linear(512, 256),  # Exactly match checkpoint dimensions [256, 512]
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Feature projection for association head input
        self.assoc_projection = nn.Linear(256, 512)  # Project from 256 to 512 before association_head
        
        # Add a feature fusion layer for concatenated features
        self.pair_feature_projection = nn.Sequential(
            nn.Linear(1024, 512),  # Reduce concatenated features from 1024 to 512
            nn.ReLU()
        )
        
        # Metric learning component for ReID features
        self.metric_fc = nn.Linear(reid_dim, reid_dim)
    
    def forward(self, images, targets=None, prev_features=None, prev_boxes=None, edge_index=None, prev_track_ids=None):
        """
        Forward pass for both detection and tracking
        
        Args:
            images: Input images tensor [B, C, H, W]
            targets: Detection targets for training
            prev_features: Features from previous frames
            prev_boxes: Boxes from previous frames
            edge_index: Edge indices for the graph
            prev_track_ids: Track IDs from previous frame
            
        Returns:
            Dict containing detector outputs and tracking information
        """
        # Step 1: Detection phase - use the detector with ReID (frozen)
        with torch.no_grad():
            detector_outputs = self.detector(images)
            
            # Extract features for current detections
            all_boxes = []
            all_scores = []
            all_labels = []
            all_reid_features = []
            all_roi_features = []  # Store ROI features for enhanced ReID
            
            for i, output in enumerate(detector_outputs):
                # Extract detected boxes, scores, labels, and ReID features
                detector_output = output['detector_outputs']
                boxes = detector_output['boxes']
                scores = detector_output['scores']
                labels = detector_output['labels']
                reid_features = output['reid_features']
                roi_features = output.get('roi_features', None)
                
                # Store results
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
                all_reid_features.append(reid_features)
                if roi_features is not None:
                    all_roi_features.append(roi_features)
        
        # Initialize tracking info
        tracking_info = None
        track_ids = None
        
        # Step 2: Enhanced ReID feature extraction (if we have ROI features)
        enhanced_reid_features = []
        for i, reid_features in enumerate(all_reid_features):
            if reid_features.shape[0] > 0:
                # If we have ROI features, process through enhanced ReID
                if len(all_roi_features) > 0 and i < len(all_roi_features):
                    roi_feats = all_roi_features[i]
                    if roi_feats is not None and roi_feats.shape[0] > 0:
                        # Process through enhanced ReID
                        enhanced_feats = self.enhanced_reid(roi_feats)
                        enhanced_reid_features.append(enhanced_feats)
                    else:
                        # Fallback to original ReID features
                        enhanced_reid_features.append(reid_features)
                else:
                    # Process through metric learning component
                    metric_feats = self.metric_fc(reid_features)
                    normalized_feats = F.normalize(metric_feats, p=2, dim=1)
                    enhanced_reid_features.append(normalized_feats)
        
        # If we have enhanced features, use them, otherwise use original
        if enhanced_reid_features:
            for i in range(len(all_reid_features)):
                if i < len(enhanced_reid_features):
                    all_reid_features[i] = enhanced_reid_features[i]
        
        # Step 3: GNN and Tracking phase
        # Only process tracking if we have detections and previous features
        if all_reid_features and all_reid_features[0].shape[0] > 0:
            curr_reid_features = all_reid_features[0]
            
            # If we have previous features, perform tracking
            if prev_features is not None and prev_features.shape[0] > 0:
                # Create edge index if not provided
                if edge_index is None:
                    # Use our new temporal edge creation approach based on feature similarities
                    if prev_boxes is not None and all_boxes[0].shape[0] > 0:
                        # Create comprehensive edge index with both spatial and temporal edges
                        edge_index = create_comprehensive_edge_index(
                            prev_features=prev_features,
                            curr_features=curr_reid_features,
                            prev_boxes=prev_boxes, 
                            curr_boxes=all_boxes[0],
                            similarity_threshold=0.5,  # Feature similarity threshold
                            spatial_threshold=0.5,     # Spatial distance threshold
                            device=images.device
                        )
                    else:
                        # If boxes aren't available, just use feature-based temporal edges
                        edge_index = create_temporal_edge_index(
                            prev_features=prev_features,
                            curr_features=curr_reid_features,
                            similarity_threshold=0.5,
                            device=images.device
                        )
                
                # Only perform GNN processing if we have valid edges
                if edge_index is not None:
                    try:
                        # Step 3a: Combine current and previous features for GNN processing
                        combined_features = torch.cat([prev_features, curr_reid_features], dim=0)
                        
                        # Step 3b: Process through GNN for tracking
                        # This runs our enhanced GNN with explicit message passing
                        node_embeddings, edge_weights, track_ids = self.gnn(combined_features, edge_index, prev_track_ids=prev_track_ids)
                        
                        # Extract the number of previous and current detections
                        num_prev = prev_features.shape[0]
                        num_curr = curr_reid_features.shape[0]
                        
                        # Get the embeddings for previous and current detections
                        prev_embeddings = node_embeddings[:num_prev]
                        curr_embeddings = node_embeddings[num_prev:]
                        
                        # Get the track IDs for current detections
                        if track_ids:
                            curr_track_ids = track_ids[num_prev:]
                        else:
                            curr_track_ids = None
                        
                        # Step 3c: Projection for association
                        # Project features for association head if needed
                        if prev_embeddings.size(1) != 512:
                            prev_embeddings = self.assoc_projection(prev_embeddings)
                        if curr_embeddings.size(1) != 512:
                            curr_embeddings = self.assoc_projection(curr_embeddings)
                        
                        # Step 3d: Compute pairwise affinities for associations
                        affinities = []
                        for i in range(num_prev):
                            for j in range(num_curr):
                                try:
                                    # Concatenate features from previous and current detections
                                    pair_features = torch.cat([prev_embeddings[i], curr_embeddings[j]], dim=0)
                                    
                                    # Project concatenated features to the expected dimension if needed
                                    if pair_features.size(0) == 1024:  # Check if we need to project
                                        pair_features_for_assoc = self.pair_feature_projection(pair_features.unsqueeze(0))
                                    else:
                                        # If they're already the right size, just unsqueeze
                                        pair_features_for_assoc = pair_features.unsqueeze(0)
                                    
                                    # Predict association score
                                    affinity = self.association_head(pair_features_for_assoc).squeeze()
                                    affinities.append(affinity)
                                except Exception as e:
                                    # Gracefully handle any errors in affinity computation
                                    print(f"Error computing affinity: {e}")
                                    affinities.append(torch.tensor(0.0, device=images.device))
                        
                        if affinities:
                            affinities = torch.stack(affinities)
                        else:
                            affinities = torch.tensor([], device=images.device)
                            
                        # Return track IDs from GNN
                        track_ids = curr_track_ids
                            
                        # Create tracking info dictionary with all computed data
                        tracking_info = {
                            'node_embeddings': node_embeddings,
                            'edge_weights': edge_weights,
                            'edge_index': edge_index,
                            'affinities': affinities,
                            'refined_features': curr_embeddings,  # Include refined features for visualization/debugging
                            'track_ids': track_ids
                        }
                    except Exception as e:
                        # Handle any errors from GNN processing
                        print(f"Error in GNN processing: {e}")
                        # Fall back to basic features without GNN refinement
                        tracking_info = {
                            'node_embeddings': None,
                            'edge_weights': None,
                            'edge_index': None,
                            'affinities': None,
                            'refined_features': curr_reid_features,
                            'track_ids': None
                        }
        
        # Return results for first image in batch (for simplicity)
        # In a real scenario, you might want to handle batch processing
        output = {
            'detector_outputs': {
                'boxes': all_boxes[0] if all_boxes else torch.empty((0, 4), device=images.device),
                'scores': all_scores[0] if all_scores else torch.empty(0, device=images.device),
                'labels': all_labels[0] if all_labels else torch.empty(0, dtype=torch.long, device=images.device)
            },
            'reid_features': all_reid_features[0] if all_reid_features else torch.empty((0, self.feature_dim), device=images.device),
            'tracking_info': tracking_info,
            'track_ids': track_ids
        }
        
        return output
    
    def reset_gnn_memory(self):
        """Reset the GNN tracking memory"""
        if hasattr(self.gnn, 'reset_memory'):
            self.gnn.reset_memory() 

# Add GCN layer implementation (from PyG if available, otherwise custom)
class GCNConv(nn.Module):
    """
    Graph Convolutional Network layer for spatial-temporal modeling
    using PyTorch Geometric implementation
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        # Use PyG's optimized implementation
        self.conv = PyGGCNConv(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        """
        Forward pass of GCN layer
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # Use PyG's implementation which is highly optimized
        return self.conv(x, edge_index)