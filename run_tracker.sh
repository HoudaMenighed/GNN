#!/bin/bash

# run_tracker.sh - Script to run the hybrid detection and tracking visualization
# Usage: ./run_tracker.sh [options]

# Default parameters
DEVICE="cpu"            # Device to run on (cpu or cuda)
SEQUENCE="MOT16-04"     # Default sequence to visualize
MAX_FRAMES=10          # Default number of frames to process
DET_THRESHOLD=0.3       # Detection confidence threshold
SIM_THRESHOLD=0.7       # Feature similarity threshold
IOU_THRESHOLD=0.3       # IoU threshold for spatial matching
SAVE_VIDEO=false        # Whether to save the video (true/false)
OUTPUT_FILE="tracking_hybrid.mp4"  # Default output filename
HUMANS_ONLY=false       # Whether to only show human detections
MOT_CLASSES=true        # Whether to use MOT Challenge classes by default
CLASS_IDS="1,3,4,2"     # Comma-separated list of class IDs (person, car, motorcycle, bicycle)
VIDEO_FILE=""           # Path to video file (empty by default)
WEBCAM=-1               # Webcam index (-1 means disabled)
DISPLAY_SCALE=1.0       # Scale factor for display window
FPS=30                  # FPS for webcam/video processing

# Display usage information
function show_help {
    echo "Usage: ./run_tracker.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -d, --device VALUE        Set device (cpu or cuda)"
    echo "  -s, --sequence VALUE      Set sequence name (default: MOT16-04)"
    echo "  -f, --frames VALUE        Set maximum number of frames (default: 100)"
    echo "  -t, --threshold VALUE     Set detection threshold (default: 0.3)"
    echo "  -m, --similarity VALUE    Set similarity threshold (default: 0.7)"
    echo "  -i, --iou VALUE           Set IoU threshold (default: 0.3)"
    echo "  -o, --output VALUE        Set output filename (default: tracking_hybrid.mp4)"
    echo "  -v, --save-video          Save the visualization as a video"
    echo "  -p, --persons-only        Show only human/person detections"
    echo "  -c, --mot-classes         Show MOT Challenge categories (person, car, motorcycle, bicycle)"
    echo "  -l, --class-ids VALUE     Comma-separated list of class IDs to track (default: 1,3,4,2)"
    echo "  --video VALUE             Path to video file for processing"
    echo "  --webcam VALUE            Camera index for webcam input (e.g., 0 for default camera)"
    echo "  --fps VALUE               FPS for webcam/video processing (default: 30)"
    echo "  --scale VALUE             Scale factor for display window (default: 1.0)"
    echo ""
    echo "Examples:"
    echo "  ./run_tracker.sh --device cuda --sequence MOT16-05 --frames 50"
    echo "  ./run_tracker.sh --frames 30 --persons-only"
    echo "  ./run_tracker.sh --frames 50 --mot-classes"
    echo "  ./run_tracker.sh --class-ids 1,3 --frames 25  # Track only persons and cars"
    echo "  ./run_tracker.sh --video path/to/video.mp4 --persons-only"
    echo "  ./run_tracker.sh --webcam 0 --save-video --output webcam_tracking.mp4"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            ;;
        -d|--device)
            DEVICE="$2"
            shift
            shift
            ;;
        -s|--sequence)
            SEQUENCE="$2"
            shift
            shift
            ;;
        -f|--frames)
            MAX_FRAMES="$2"
            shift
            shift
            ;;
        -t|--threshold)
            DET_THRESHOLD="$2"
            shift
            shift
            ;;
        -m|--similarity)
            SIM_THRESHOLD="$2"
            shift
            shift
            ;;
        -i|--iou)
            IOU_THRESHOLD="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift
            shift
            ;;
        -v|--save-video)
            SAVE_VIDEO=true
            shift
            ;;
        -p|--persons-only)
            HUMANS_ONLY=true
            MOT_CLASSES=false
            shift
            ;;
        -c|--mot-classes)
            MOT_CLASSES=true
            HUMANS_ONLY=false
            shift
            ;;
        -l|--class-ids)
            CLASS_IDS="$2"
            MOT_CLASSES=false
            HUMANS_ONLY=false
            shift
            shift
            ;;
        --video)
            VIDEO_FILE="$2"
            shift
            shift
            ;;
        --webcam)
            WEBCAM="$2"
            shift
            shift
            ;;
        --fps)
            FPS="$2"
            shift
            shift
            ;;
        --scale)
            DISPLAY_SCALE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Start building the command
ARGS=()

# Add basic parameters
ARGS+=("--device" "$DEVICE")
ARGS+=("--max_frames" "$MAX_FRAMES")
ARGS+=("--detection_threshold" "$DET_THRESHOLD")
ARGS+=("--similarity_threshold" "$SIM_THRESHOLD")
ARGS+=("--iou_threshold" "$IOU_THRESHOLD")

# Add input source options
if [ ! -z "$VIDEO_FILE" ]; then
    # Video file input takes precedence
    ARGS+=("--video" "$VIDEO_FILE")
elif [ "$WEBCAM" -ge 0 ]; then
    # Webcam input
    ARGS+=("--webcam" "$WEBCAM")
else
    # Default to sequence
    ARGS+=("--seq_name" "$SEQUENCE")
fi

# Add FPS and scale if using video or webcam
if [ ! -z "$VIDEO_FILE" ] || [ "$WEBCAM" -ge 0 ]; then
    ARGS+=("--fps" "$FPS")
    ARGS+=("--display_scale" "$DISPLAY_SCALE")
fi

# Add optional arguments
if [ "$SAVE_VIDEO" = true ]; then
    ARGS+=("--save_video")
    ARGS+=("--output" "$OUTPUT_FILE")
fi

# Add class options
if [ "$HUMANS_ONLY" = true ]; then
    ARGS+=("--persons_only")
elif [ "$MOT_CLASSES" = true ]; then
    ARGS+=("--mot_classes")
else
    ARGS+=("--class_ids" "$CLASS_IDS")
fi

# Print the command being executed
echo "Running: python show_hybrid_model.py ${ARGS[@]}"
echo "============================================================="

# Execute the command
python show_hybrid_model.py "${ARGS[@]}" 