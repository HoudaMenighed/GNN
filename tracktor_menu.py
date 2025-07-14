#!/usr/bin/env python3
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import cv2
import threading
import numpy as np
import torch
import time

class TracktorMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Tracktor MOT Menu")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Set style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0", font=('Arial', 12))
        self.style.configure("TNotebook", background="#f0f0f0")
        self.style.configure("TNotebook.Tab", padding=[20, 5], font=('Arial', 11))
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Create frames for each tab
        self.tracking_frame = ttk.Frame(self.notebook)
        self.learning_frame = ttk.Frame(self.notebook)
        self.settings_frame = ttk.Frame(self.notebook)
        
        # Add frames to notebook
        self.notebook.add(self.tracking_frame, text="Tracking")
        self.notebook.add(self.learning_frame, text="Learning")
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Initialize settings
        self.device = tk.StringVar(value="cpu")
        self.max_frames = tk.IntVar(value=10)
        self.detection_threshold = tk.DoubleVar(value=0.7)
        self.similarity_threshold = tk.DoubleVar(value=0.7)
        self.iou_threshold = tk.DoubleVar(value=0.7)
        self.save_video = tk.BooleanVar(value=False)
        self.output_filename = tk.StringVar(value="tracking_output.mp4")
        self.persons_only = tk.BooleanVar(value=True)
        self.mot_classes = tk.BooleanVar(value=False)
        self.class_ids = tk.StringVar(value="1,3,4,2")
        
        # Setup the UI for each tab
        self._setup_tracking_tab()
        self._setup_learning_tab()
        self._setup_settings_tab()
        
        # Camera stream variables
        self.camera_stream = None
        self.is_streaming = False

    def _setup_tracking_tab(self):
        tracking_content = ttk.Frame(self.tracking_frame)
        tracking_content.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        ttk.Label(tracking_content, text="Object Tracking", font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        # Track from video file
        video_frame = ttk.Frame(tracking_content)
        video_frame.pack(fill="x", pady=10)
        
        ttk.Label(video_frame, text="Track objects from video file:").pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            video_frame, 
            text="Select Video", 
            command=self.track_from_video_file
        ).pack(side=tk.LEFT, padx=5)
        
        # Track from image sequence
        seq_frame = ttk.Frame(tracking_content)
        seq_frame.pack(fill="x", pady=10)
        
        ttk.Label(seq_frame, text="Track from MOT16 sequence:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.sequence_var = tk.StringVar(value="MOT16-04")
        seq_combo = ttk.Combobox(
            seq_frame, 
            textvariable=self.sequence_var,
            values=["MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10", "MOT16-11", "MOT16-13"]
        )
        seq_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            seq_frame, 
            text="Track Sequence", 
            command=self.track_sequence
        ).pack(side=tk.LEFT, padx=5)
        
        # Track from webcam
        webcam_frame = ttk.Frame(tracking_content)
        webcam_frame.pack(fill="x", pady=10)
        
        ttk.Label(webcam_frame, text="Track from webcam:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.camera_var = tk.IntVar(value=0)
        camera_combo = ttk.Combobox(
            webcam_frame, 
            textvariable=self.camera_var,
            values=[0, 1, 2, 3]  # Camera indices
        )
        camera_combo.pack(side=tk.LEFT, padx=5)
        
        self.webcam_button = ttk.Button(
            webcam_frame, 
            text="Start Webcam Tracking", 
            command=self.toggle_webcam_tracking
        )
        self.webcam_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(tracking_content, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Output console
        ttk.Label(tracking_content, text="Output Log:").pack(anchor=tk.W, pady=(20, 5))
        
        self.console = tk.Text(tracking_content, height=10, width=70, bg="#f8f8f8", fg="#333")
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)

    def _setup_learning_tab(self):
        learning_content = ttk.Frame(self.learning_frame)
        learning_content.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        ttk.Label(learning_content, text="Model Training", font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        # Quick start joint training
        quick_frame = ttk.LabelFrame(learning_content, text="Quick Start Joint Training")
        quick_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(quick_frame, text="Run a quick test with minimal parameters:").pack(anchor=tk.W, pady=(5, 0), padx=10)
        
        quick_params_frame = ttk.Frame(quick_frame)
        quick_params_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(quick_params_frame, text="Max Frames:").grid(row=0, column=0, sticky=tk.W, padx=5)
        quick_frames_entry = ttk.Entry(quick_params_frame, width=10)
        quick_frames_entry.insert(0, "300")
        quick_frames_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(quick_params_frame, text="Num Epochs:").grid(row=0, column=2, sticky=tk.W, padx=5)
        quick_epochs_entry = ttk.Entry(quick_params_frame, width=10)
        quick_epochs_entry.insert(0, "2")
        quick_epochs_entry.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Button(
            quick_frame,
            text="Start Quick Training",
            command=lambda: self.run_joint_training(
                max_frames=quick_frames_entry.get(),
                num_epochs=quick_epochs_entry.get(),
                batch_size=1
            )
        ).pack(anchor=tk.W, pady=10, padx=10)
        
        # Full joint training
        full_frame = ttk.LabelFrame(learning_content, text="Full Joint Training")
        full_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(full_frame, text="Run full training with custom parameters:").pack(anchor=tk.W, pady=(5, 0), padx=10)
        
        full_params_frame = ttk.Frame(full_frame)
        full_params_frame.pack(fill="x", pady=10, padx=10)
        
        # Parameters for full training
        params = [
            ("Data Dir:", "./data/MOT16", 0, 0),
            ("Output Dir:", "./checkpoints/joint", 0, 2),
            ("Batch Size:", "1", 1, 0),
            ("Num Epochs:", "20", 1, 2),
            ("Learning Rate:", "0.0001", 2, 0),
            ("Num Workers:", "4", 2, 2),
            ("Device:", "cuda", 3, 0),
            ("Num Frames:", "100", 3, 2)
        ]
        
        self.full_training_entries = {}
        
        for label, default, row, col in params:
            ttk.Label(full_params_frame, text=label).grid(row=row, column=col, sticky=tk.W, padx=5)
            entry = ttk.Entry(full_params_frame, width=10)
            entry.insert(0, default)
            entry.grid(row=row, column=col+1, sticky=tk.W, padx=5)
            # Store entry widget by parameter name
            self.full_training_entries[label.strip(":")] = entry
        
        # Component selection
        component_frame = ttk.LabelFrame(full_frame, text="Components to Train")
        component_frame.pack(fill="x", pady=10, padx=10)
        
        # Create variables for checkboxes
        self.train_detector_var = tk.BooleanVar(value=False)
        self.train_reid_var = tk.BooleanVar(value=False)
        self.train_gnn_var = tk.BooleanVar(value=True)
        self.train_association_var = tk.BooleanVar(value=True)
        
        # Add checkboxes for component selection
        ttk.Checkbutton(
            component_frame, 
            text="Train Detector", 
            variable=self.train_detector_var
        ).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        ttk.Checkbutton(
            component_frame, 
            text="Train ReID", 
            variable=self.train_reid_var
        ).grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        ttk.Checkbutton(
            component_frame, 
            text="Train GNN", 
            variable=self.train_gnn_var
        ).grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        
        ttk.Checkbutton(
            component_frame, 
            text="Train Association", 
            variable=self.train_association_var
        ).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Loss weights
        weights_frame = ttk.LabelFrame(full_frame, text="Loss Weights")
        weights_frame.pack(fill="x", pady=10, padx=10)
        
        # Create variables for loss weights
        self.detector_weight_var = tk.StringVar(value="1.0")
        self.reid_weight_var = tk.StringVar(value="0.5")
        self.gnn_weight_var = tk.StringVar(value="0.3")
        self.association_weight_var = tk.StringVar(value="0.7")
        
        # Add entry fields for loss weights
        ttk.Label(weights_frame, text="Detector Weight:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(
            weights_frame, 
            textvariable=self.detector_weight_var,
            width=5
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(weights_frame, text="ReID Weight:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(
            weights_frame, 
            textvariable=self.reid_weight_var,
            width=5
        ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(weights_frame, text="GNN Weight:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(
            weights_frame, 
            textvariable=self.gnn_weight_var,
            width=5
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(weights_frame, text="Association Weight:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(
            weights_frame, 
            textvariable=self.association_weight_var,
            width=5
        ).grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Other options
        options_frame = ttk.Frame(full_frame)
        options_frame.pack(fill="x", pady=10, padx=10)
        
        self.plot_losses_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame, 
            text="Plot Losses", 
            variable=self.plot_losses_var
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            full_frame,
            text="Start Full Training",
            command=self.run_full_joint_training
        ).pack(anchor=tk.W, pady=10, padx=10)

    def _setup_settings_tab(self):
        settings_content = ttk.Frame(self.settings_frame)
        settings_content.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Title
        ttk.Label(settings_content, text="Tracking Settings", font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        settings_frame = ttk.LabelFrame(settings_content, text="Tracking Parameters")
        settings_frame.pack(fill="x", pady=10, padx=10)
        
        # Device setting
        device_frame = ttk.Frame(settings_frame)
        device_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(device_frame, text="Device:").pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Radiobutton(
            device_frame, 
            text="CPU", 
            variable=self.device, 
            value="cpu"
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            device_frame, 
            text="CUDA", 
            variable=self.device, 
            value="cuda"
        ).pack(side=tk.LEFT, padx=5)
        
        # Max frames
        frames_frame = ttk.Frame(settings_frame)
        frames_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(frames_frame, text="Max Frames:").pack(side=tk.LEFT, padx=(0, 10))
        
        frames_entry = ttk.Entry(frames_frame, textvariable=self.max_frames, width=10)
        frames_entry.pack(side=tk.LEFT, padx=5)
        
        # Detection threshold
        det_frame = ttk.Frame(settings_frame)
        det_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(det_frame, text="Detection Threshold:").pack(side=tk.LEFT, padx=(0, 10))
        
        det_entry = ttk.Entry(det_frame, textvariable=self.detection_threshold, width=10)
        det_entry.pack(side=tk.LEFT, padx=5)
        
        # Similarity threshold
        sim_frame = ttk.Frame(settings_frame)
        sim_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(sim_frame, text="Similarity Threshold:").pack(side=tk.LEFT, padx=(0, 10))
        
        sim_entry = ttk.Entry(sim_frame, textvariable=self.similarity_threshold, width=10)
        sim_entry.pack(side=tk.LEFT, padx=5)
        
        # IoU threshold
        iou_frame = ttk.Frame(settings_frame)
        iou_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Label(iou_frame, text="IoU Threshold:").pack(side=tk.LEFT, padx=(0, 10))
        
        iou_entry = ttk.Entry(iou_frame, textvariable=self.iou_threshold, width=10)
        iou_entry.pack(side=tk.LEFT, padx=5)
        
        # Save video checkbox
        video_frame = ttk.Frame(settings_frame)
        video_frame.pack(fill="x", pady=10, padx=10)
        
        ttk.Checkbutton(
            video_frame, 
            text="Save Video", 
            variable=self.save_video
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(video_frame, text="Output Filename:").pack(side=tk.LEFT, padx=(20, 5))
        
        output_entry = ttk.Entry(video_frame, textvariable=self.output_filename, width=20)
        output_entry.pack(side=tk.LEFT, padx=5)
        
        # Class settings
        class_frame = ttk.LabelFrame(settings_content, text="Detection Classes")
        class_frame.pack(fill="x", pady=10, padx=10)
        
        # Persons only checkbox
        persons_check = ttk.Checkbutton(
            class_frame,
            text="Persons Only",
            variable=self.persons_only,
            command=lambda: self._toggle_class_settings(self.persons_only, [self.mot_classes])
        )
        persons_check.pack(anchor=tk.W, pady=5, padx=10)
        
        # MOT classes checkbox
        mot_check = ttk.Checkbutton(
            class_frame,
            text="MOT Challenge Classes (person, car, motorcycle, bicycle)",
            variable=self.mot_classes,
            command=lambda: self._toggle_class_settings(self.mot_classes, [self.persons_only])
        )
        mot_check.pack(anchor=tk.W, pady=5, padx=10)
        
        # Class IDs
        class_id_frame = ttk.Frame(class_frame)
        class_id_frame.pack(fill="x", pady=5, padx=10)
        
        ttk.Label(class_id_frame, text="Custom Class IDs (comma-separated):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        class_id_entry = ttk.Entry(class_id_frame, textvariable=self.class_ids, width=20)
        class_id_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

    def _toggle_class_settings(self, enabled_var, disabled_vars):
        """Toggle radio button behavior for class settings"""
        if enabled_var.get():
            for var in disabled_vars:
                var.set(False)

    def log_message(self, message):
        """Add a message to the console log"""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)  # Auto-scroll to the end
        self.console.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def track_from_video_file(self):
        """Track objects from a video file"""
        # Open file dialog to select video
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )
        
        if not video_path:
            return
        
        self.status_var.set(f"Processing video: {os.path.basename(video_path)}")
        self.log_message(f"Starting tracking for video: {video_path}")
        
        # Build command based on current settings
        cmd = ["./run_tracker.sh"]
        
        # Add device
        cmd.extend(["--device", self.device.get()])
        
        # Add video file option (this will be implemented as a mod to run_tracker.sh)
        cmd.extend(["--video", video_path])
        
        # Add max frames
        cmd.extend(["--frames", str(self.max_frames.get())])
        
        # Add thresholds
        cmd.extend(["--threshold", str(self.detection_threshold.get())])
        cmd.extend(["--similarity", str(self.similarity_threshold.get())])
        cmd.extend(["--iou", str(self.iou_threshold.get())])
        
        # Add video saving option if enabled
        if self.save_video.get():
            cmd.append("--save-video")
            cmd.extend(["--output", self.output_filename.get()])
        
        # Add class options
        if self.persons_only.get():
            cmd.append("--persons-only")
        elif self.mot_classes.get():
            cmd.append("--mot-classes")
        else:
            cmd.extend(["--class-ids", self.class_ids.get()])
        
        # Run the command in a separate thread to avoid freezing the UI
        threading.Thread(target=self._run_command, args=(cmd,)).start()

    def track_sequence(self):
        """Track objects from a MOT16 sequence"""
        sequence = self.sequence_var.get()
        
        self.status_var.set(f"Processing sequence: {sequence}")
        self.log_message(f"Starting tracking for sequence: {sequence}")
        
        # Build command based on current settings
        cmd = ["./run_tracker.sh"]
        
        # Add device
        cmd.extend(["--device", self.device.get()])
        
        # Add sequence
        cmd.extend(["--sequence", sequence])
        
        # Add max frames
        cmd.extend(["--frames", str(self.max_frames.get())])
        
        # Add thresholds
        cmd.extend(["--threshold", str(self.detection_threshold.get())])
        cmd.extend(["--similarity", str(self.similarity_threshold.get())])
        cmd.extend(["--iou", str(self.iou_threshold.get())])
        
        # Add video saving option if enabled
        if self.save_video.get():
            cmd.append("--save-video")
            cmd.extend(["--output", self.output_filename.get()])
        
        # Add class options
        if self.persons_only.get():
            cmd.append("--persons-only")
        elif self.mot_classes.get():
            cmd.append("--mot-classes")
        else:
            cmd.extend(["--class-ids", self.class_ids.get()])
        
        # Run the command in a separate thread to avoid freezing the UI
        threading.Thread(target=self._run_command, args=(cmd,)).start()

    def toggle_webcam_tracking(self):
        """Toggle webcam tracking on/off"""
        if self.is_streaming:
            self.is_streaming = False
            self.webcam_button.config(text="Start Webcam Tracking")
            self.status_var.set("Webcam tracking stopped")
            self.log_message("Webcam tracking stopped")
        else:
            self.is_streaming = True
            self.webcam_button.config(text="Stop Webcam Tracking")
            self.status_var.set("Starting webcam tracking...")
            self.log_message(f"Starting webcam tracking from camera index {self.camera_var.get()}")
            
            # Start webcam tracking in a separate thread
            threading.Thread(target=self._webcam_tracking).start()

    def _webcam_tracking(self):
        """Run tracking on webcam feed"""
        try:
            # Create command for webcam tracking
            cmd = [
                "python", "show_hybrid_model.py",
                "--device", self.device.get(),
                "--max_frames", str(self.max_frames.get()),
                "--detection_threshold", str(self.detection_threshold.get()),
                "--similarity_threshold", str(self.similarity_threshold.get()),
                "--iou_threshold", str(self.iou_threshold.get()),
                "--webcam", str(self.camera_var.get())
            ]
            
            # Add class options
            if self.persons_only.get():
                cmd.append("--persons_only")
            elif self.mot_classes.get():
                cmd.append("--mot_classes")
            else:
                cmd.extend(["--class_ids", self.class_ids.get()])
            
            if self.save_video.get():
                cmd.extend(["--save_video", "--output", self.output_filename.get()])
            
            # Log the command being run
            self.log_message("Running webcam tracking with command: " + " ".join(f'"{arg}"' if ' ' in arg else arg for arg in cmd))
            
            # Create a shell command string with proper quoting
            shell_cmd = "python show_hybrid_model.py"
            
            # Add options with proper quoting
            for i in range(2, len(cmd), 2):
                if i+1 < len(cmd):
                    # This is a key-value pair (--option value)
                    option = cmd[i]
                    value = cmd[i+1]
                    # Quote the value if it contains spaces
                    if ' ' in value:
                        shell_cmd += f' {option} "{value}"'
                    else:
                        shell_cmd += f' {option} {value}'
                else:
                    # This is a flag option (--option)
                    shell_cmd += f' {cmd[i]}'
            
            # Run the command as a shell command
            self.log_message(f"Executing shell command: {shell_cmd}")
            process = subprocess.Popen(
                shell_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                shell=True,
                bufsize=1
            )
            
            # Create a function to read from a stream and log it
            def read_stream(stream, prefix=""):
                while self.is_streaming:
                    line = stream.readline()
                    if not line:
                        # If we reach EOF, the process might have ended
                        if process.poll() is not None:
                            break
                        # Otherwise, just wait a bit and try again
                        time.sleep(0.1)
                        continue
                    self.log_message(prefix + line.strip())
            
            # Create and start threads to read stdout and stderr
            stdout_thread = threading.Thread(
                target=read_stream, 
                args=(process.stdout, "")
            )
            stderr_thread = threading.Thread(
                target=read_stream, 
                args=(process.stderr, "ERROR: ")
            )
            
            # Set as daemon threads so they exit when the main thread exits
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            # Start the threads
            stdout_thread.start()
            stderr_thread.start()
            
            # Monitor the process
            while self.is_streaming:
                # Check if process is still running
                if process.poll() is not None:
                    break
                time.sleep(0.5)
            
            # If we get here and is_streaming is still True, process ended unexpectedly
            if self.is_streaming:
                self.is_streaming = False
                self.webcam_button.config(text="Start Webcam Tracking")
                self.status_var.set("Webcam tracking stopped (process ended)")
                self.log_message("Webcam tracking process ended")
            
            # Kill the process if it's still running
            if process.poll() is None:
                process.terminate()
                
        except Exception as e:
            self.log_message(f"Error in webcam tracking: {str(e)}")
            self.is_streaming = False
            self.webcam_button.config(text="Start Webcam Tracking")
            self.status_var.set("Webcam tracking error")

    def run_joint_training(self, max_frames, num_epochs, batch_size):
        """Run quick joint training"""
        self.status_var.set("Starting quick joint training...")
        self.log_message(f"Starting quick joint training with {num_epochs} epochs and {max_frames} frames per sequence")
        
        cmd = [
            "python", "joint_training.py", 
            "--num_frames", str(max_frames),
            "--num_epochs", str(num_epochs),
            "--batch_size", str(batch_size),
            "--train_gnn",
            "--train_association",
            "--plot_losses"
        ]
        
        # Add device setting
        if self.device.get() == "cuda":
            cmd.extend(["--device", "cuda"])
        
        # Run the command in a separate thread to avoid freezing the UI
        threading.Thread(target=self._run_command, args=(cmd,)).start()

    def run_full_joint_training(self):
        """Run full joint training with all parameters"""
        self.status_var.set("Starting full joint training...")
        self.log_message("Starting full joint training with custom parameters")
        
        cmd = ["python", "joint_training.py"]
        
        # Add parameters from entry fields
        for param, entry in self.full_training_entries.items():
            # Convert parameter name to command line arg name (e.g., "Data Dir" -> "--data_dir")
            arg_name = "--" + param.lower().replace(" ", "_")
            cmd.extend([arg_name, entry.get()])
        
        # Add component training flags
        component_frame = ttk.Frame(self.learning_frame)
        if hasattr(self, 'train_detector_var') and self.train_detector_var.get():
            cmd.append("--train_detector")
        if hasattr(self, 'train_reid_var') and self.train_reid_var.get():
            cmd.append("--train_reid")
        if hasattr(self, 'train_gnn_var') and self.train_gnn_var.get():
            cmd.append("--train_gnn")
        if hasattr(self, 'train_association_var') and self.train_association_var.get():
            cmd.append("--train_association")
        
        # Add loss weights if specified
        if hasattr(self, 'detector_weight_var') and self.detector_weight_var.get():
            cmd.extend(["--detector_weight", self.detector_weight_var.get()])
        if hasattr(self, 'reid_weight_var') and self.reid_weight_var.get():
            cmd.extend(["--reid_weight", self.reid_weight_var.get()])
        if hasattr(self, 'gnn_weight_var') and self.gnn_weight_var.get():
            cmd.extend(["--gnn_weight", self.gnn_weight_var.get()])
        if hasattr(self, 'association_weight_var') and self.association_weight_var.get():
            cmd.extend(["--association_weight", self.association_weight_var.get()])
        
        # Add plot losses flag
        if hasattr(self, 'plot_losses_var') and self.plot_losses_var.get():
            cmd.append("--plot_losses")
        
        # Run the command in a separate thread to avoid freezing the UI
        threading.Thread(target=self._run_command, args=(cmd,)).start()

    def _run_command(self, cmd):
        """Run a command and capture its output in real-time"""
        try:
            # Log the command being run
            self.log_message("Running command: " + " ".join(f'"{arg}"' if ' ' in arg else arg for arg in cmd))
            
            # Create a single shell command for run_tracker.sh
            if cmd[0] == "./run_tracker.sh":
                # For run_tracker.sh, we'll create a single command string
                shell_cmd = cmd[0]
                
                # Add each argument with proper quoting
                for i in range(1, len(cmd), 2):
                    if i+1 < len(cmd):
                        # This is a key-value pair (--option value)
                        option = cmd[i]
                        value = cmd[i+1]
                        # Quote the value if it contains spaces
                        if ' ' in value:
                            shell_cmd += f' {option} "{value}"'
                        else:
                            shell_cmd += f' {option} {value}'
                    else:
                        # This is a flag option (--option)
                        shell_cmd += f' {cmd[i]}'
                
                # Run the command as a shell command
                self.log_message(f"Executing shell command: {shell_cmd}")
                process = subprocess.Popen(
                    shell_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    shell=True,
                    bufsize=1
                )
            else:
                # For other commands, use the list form
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1
                )
            
            # Create a function to read from a stream and log it
            def read_stream(stream, prefix=""):
                for line in iter(stream.readline, ''):
                    if not line:
                        break
                    self.log_message(prefix + line.strip())
            
            # Create and start threads to read stdout and stderr
            stdout_thread = threading.Thread(
                target=read_stream, 
                args=(process.stdout, "")
            )
            stderr_thread = threading.Thread(
                target=read_stream, 
                args=(process.stderr, "ERROR: ")
            )
            
            # Set as daemon threads so they exit when the main thread exits
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            # Start the threads
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for the process to complete
            process.wait()
            
            # Wait for the output threads to finish
            stdout_thread.join()
            stderr_thread.join()
            
            if process.returncode == 0:
                self.status_var.set("Command completed successfully")
                self.log_message("Command completed successfully")
            else:
                self.status_var.set(f"Command failed with exit code {process.returncode}")
                self.log_message(f"Command failed with exit code {process.returncode}")
                
        except Exception as e:
            self.log_message(f"Error running command: {str(e)}")
            self.status_var.set("Error running command")

if __name__ == "__main__":
    root = tk.Tk()
    app = TracktorMenu(root)
    root.mainloop() 