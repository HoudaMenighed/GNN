import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.animation as animation

class Visualizer:
    def __init__(self, width, height):
        self.dpi = 96
        self.fig, self.ax = plt.subplots(1, dpi=self.dpi, figsize=(12, 9))  # Larger figure for better visibility
        
        # Setup display area
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(height, 0)
        plt.axis('off')
        
        self.rects = []
        self.texts = []
        self.arrows = []
        self.image_display = None
        
        # Color mapping for consistent track colors - use high contrast colors
        self.colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
            '#FF8000', '#8000FF', '#00FF80', '#FF0080', '#80FF00', '#0080FF',
            '#FF4040', '#40FF40', '#4040FF', '#FFFF40', '#FF40FF', '#40FFFF',
            '#FF8040', '#8040FF', '#40FF80', '#FF4080', '#80FF40', '#4080FF'
        ]
        self.id_to_color = {}
        
        # Title for legend
        self.fig.suptitle('GNN for Multi-Object Tracking', fontsize=16)
        
    def init_display(self, first_frame):
        self.image_display = self.ax.imshow(first_frame)
        return self.image_display
    
    def add_box(self, box, color='r', track_id=None, velocity=None, is_detection=False, score=None):
        """
        Add a bounding box with optional tracking information
        
        Args:
            box: [x1, y1, x2, y2] bounding box
            color: color string or RGB tuple
            track_id: optional track ID
            velocity: optional velocity vector [vx, vy, vz]
            is_detection: if True, this is a raw detection (not a tracked object)
            score: detection score (for raw detections)
        """
        # Get consistent color for track ID
        if track_id is not None and not is_detection:
            if track_id not in self.id_to_color:
                self.id_to_color[track_id] = self.colors[track_id % len(self.colors)]
            color = self.id_to_color[track_id]
        
        # Different styles for detections vs tracked objects
        if is_detection:
            # Detection boxes are dashed lines
            linestyle = '--'
            linewidth = 1.0
        else:
            # Tracked objects are solid lines with thicker width
            linestyle = '-'
            linewidth = 2.0
            
        # Create rectangle
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle,
            alpha=0.8 if is_detection else 1.0
        )
        self.ax.add_patch(rect)
        self.rects.append(rect)
        
        # Add text label - position directly on the bounding box
        if track_id is not None:
            # Position ID text directly on top-left corner of the bounding box
            # Calculate positions - centered above the box
            text_x = box[0] + 2  # Small offset from left edge
            text_y = box[1] + 2  # Small offset from top edge
            
            # Always show ID for both detections and tracks
            label_text = f"{track_id}"  # Just the ID number
            label_color = 'white'
            bg_color = color
            # Larger font for better visibility
            font_size = 9 if is_detection else 11
                
            # Create text with better visibility
            text = self.ax.text(
                text_x, text_y,
                label_text,
                fontsize=font_size,
                fontweight='bold',
                color=label_color,
                bbox=dict(
                    boxstyle="round,pad=0.1",  # Smaller padding
                    fc=bg_color,
                    ec="black",
                    alpha=0.9
                )
            )
            self.texts.append(text)
            
        # # Add velocity arrow if available
        # if velocity is not None and (velocity[0] != 0 or velocity[1] != 0):
        #     # Calculate center of the box
        #     center_x = (box[0] + box[2]) / 2
        #     center_y = (box[1] + box[3]) / 2
            
        #     # Scale velocity for visualization
        #     scale = 5.0
        #     arrow = self.ax.arrow(
        #         center_x, center_y,
        #         velocity[0] * scale, velocity[1] * scale,
        #         head_width=5, head_length=7, fc=color, ec=color, 
        #         alpha=0.9
        #     )
        #     self.arrows.append(arrow)
    
    def add_detection(self, box, score, track_id=None):
        """
        Add a detection bounding box specifically from Faster R-CNN
        
        Args:
            box: [x1, y1, x2, y2] bounding box
            score: detection confidence score
            track_id: optional track ID to display
        """
        # Use blue color for raw detections
        color = 'blue'
        self.add_box(box, color=color, is_detection=True, score=score, track_id=track_id)
    
    def clear_boxes(self):
        """Clear all visualization elements"""
        # Remove rectangles
        for rect in self.rects:
            rect.remove()
        self.rects = []
        
        # Remove texts
        for text in self.texts:
            text.remove()
        self.texts = []
        
        # Remove arrows
        for arrow in self.arrows:
            arrow.remove()
        self.arrows = []
        
    def add_tracked_object(self, track):
        """
        Add a tracked object with all its information
        
        Args:
            track: Dictionary with tracking information
        """
        try:
            box = track['box']
            track_id = track['id']
            
            # Extract velocity if available (only x,y components for 2D visualization)
            velocity = None
            if 'velocity' in track:
                velocity = track['velocity'][:2]  # Only use x,y components
            
            # Use a unique color for each track based on its ID
            if track_id not in self.id_to_color:
                # Ensure consistent color assignment
                self.id_to_color[track_id] = self.colors[track_id % len(self.colors)]
            color = self.id_to_color[track_id]
            
            # Create rectangle with thicker border for better visibility
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=3.0,  # Increased from 2.0 for better visibility
                edgecolor=color,
                facecolor='none',
                linestyle='-',  # Solid line for all objects
                alpha=1.0,      # Full opacity
                zorder=10       # Ensure boxes are drawn on top
            )
            self.ax.add_patch(rect)
            self.rects.append(rect)
            
            # Add text label with contrasting background for better visibility
            text_x = box[0] + 2
            text_y = box[1] + 2
            
            label_text = f"{track_id}"
            label_color = 'white'
            bg_color = color
            font_size = 12  # Increased font size
            
            text = self.ax.text(
                text_x, text_y,
                label_text,
                fontsize=font_size,
                fontweight='bold',
                color=label_color,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc=bg_color,
                    ec="black",
                    alpha=1.0,  # Full opacity
                    linewidth=2  # Thicker text box outline
                ),
                zorder=20  # Ensure text is drawn on top of everything
            )
            self.texts.append(text)
        except Exception as e:
            print(f"Error rendering box for track ID {track.get('id', 'unknown')}: {e}")
            
    def display_detections_and_tracks(self, detections, tracks):
        """
        Display both raw detections and tracked objects
        
        Args:
            detections: List of detection dictionaries with 'box', 'score', and optionally 'id'
            tracks: List of track dictionaries
        """
        self.clear_boxes()
        
        # Add detections first (so they appear behind tracks)
        for det in detections:
            track_id = det.get('id')  # Get the ID if it exists, otherwise None
            self.add_detection(det['box'], det['score'], track_id)
            
        # Then add tracked objects
        for track in tracks:
            self.add_tracked_object(track)
            
    def display_tracks(self, tracks):
        """
        Display all tracks on the visualization
        
        Args:
            tracks: list of dictionaries with information about each track
        """
        # Define a set of high contrast colors for better visibility
        # Using more vibrant, distinguishable colors for track visualization
        high_contrast_colors = [
            'red', 'lime', 'blue', 'yellow', 'magenta', 'cyan', 
            'orangered', 'limegreen', 'deepskyblue', 'gold', 'purple', 'hotpink',
            'crimson', 'springgreen', 'royalblue', 'orange', 'mediumorchid', 'aqua',
            'firebrick', 'lawngreen', 'dodgerblue', 'goldenrod', 'darkviolet', 'turquoise'
        ]
        
        # Set text properties for better visibility
        text_props = dict(
            fontsize=10, 
            fontweight='bold',
            color='white',
            bbox=dict(
                boxstyle="round,pad=0.3", 
                fc='black', 
                ec="white", 
                alpha=0.8
            )
        )
        
        for track in tracks:
            track_id = track['id']
            
            # Get the box coordinates
            box = track['box']
            
            # Determine color based on track ID
            color_idx = track_id % len(high_contrast_colors)
            color = high_contrast_colors[color_idx]
            
            # Get the width and height
            width = box[2] - box[0]
            height = box[3] - box[1]
            
            # Create a rectangle patch
            rect = patches.Rectangle(
                (box[0], box[1]), width, height, 
                linewidth=1,  # Reduced line width to 1px as requested
                edgecolor=color,
                facecolor='none',
                alpha=1.0    # Full opacity
            )
            
            # Add the rectangle to the axis
            self.ax.add_patch(rect)
            self.rects.append(rect)
            
            # Add ID text with improved visibility
            id_text = self.ax.text(
                box[0], box[1] - 10, 
                f"ID: {track_id}", 
                **text_props
            )
            
            self.texts.append(id_text)
            
            # Score label removed as requested
            
    def add_legend(self):
        """Add a legend to explain the visualization elements"""
        # Create simple legend for tracked objects
        track_patch = patches.Patch(color=self.colors[0], linestyle='-', linewidth=3.0, label='Tracked Object')
        
        # Add legend
        self.ax.legend(handles=[track_patch], 
                      loc='upper right', 
                      bbox_to_anchor=(1, 0),
                      fontsize=12)
            
    def create_animation(self, seq, tracker, frames=None, interval=50):
        """
        Create animation of tracking results
        
        Args:
            seq: Sequence object
            tracker: Tracker object
            frames: Number of frames to process (default: all)
            interval: Animation interval in milliseconds
            
        Returns:
            matplotlib animation object
        """
        if frames is None:
            frames = len(seq)
            
        def update(frame_idx):
            # Get frame data
            frame = seq[frame_idx]
            
            # Update image
            img = frame['img'].mul(255).permute(1, 2, 0).byte().numpy()
            self.image_display.set_data(img)
            
            # Clear previous visualization
            self.clear_boxes()
            
            # Process frame with tracker
            tracked_objects = tracker.process_frame(frame)
            
            # Display tracked objects
            self.display_tracks(tracked_objects)
            
            # Return all artists that need to be redrawn
            artists = [self.image_display] + self.rects + self.texts + self.arrows
            return artists
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, update, frames=frames, interval=interval, blit=True
        )
        
        return anim
