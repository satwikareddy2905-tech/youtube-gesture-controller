"""GUI module for the YouTube Gesture Controller.

Provides a modern dashboard interface built with CustomTkinter. Handles video
rendering, real-time overlays, toggles for execution, event logging panels, and
orderly thread termination on exit.
"""

from typing import Optional
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from camera import Camera
from config import AppConfig, GestureName, KeyboardAction
from controller import ActionController
from gesture_recognition import GestureRecognizer
from hand_detector import HandDetector
from logger import get_logger
from utils import FPSCalculator

logger = get_logger(__name__)


class GestureControllerApp(ctk.CTk):
    """Main CustomTkinter GUI Application for gesture control."""

    def __init__(
        self,
        config: AppConfig,
        camera: Camera,
        detector: HandDetector,
        recognizer: GestureRecognizer,
        controller: ActionController,
    ) -> None:
        """Initializes the GUI widgets, bindings, and states.

        Args:
            config: Aggregated settings object.
            camera: Threaded Camera reader.
            detector: MediaPipe hand detector.
            recognizer: Hand gesture classifier.
            controller: Keystroke dispatcher.
        """
        super().__init__()

        self._config = config
        self._camera = camera
        self._detector = detector
        self._recognizer = recognizer
        self._controller = controller

        # Thread-safety / state flags
        self._is_detection_running: bool = True
        self._process_fps_calculator = FPSCalculator(buffer_size=30)

        # Style configurations
        ctk.set_appearance_mode(self._config.gui.THEME_MODE)
        ctk.set_default_color_theme(self._config.gui.COLOR_THEME)

        self.title(self._config.gui.TITLE)
        self.geometry(self._config.gui.WINDOW_SIZE)
        self.resizable(False, False)

        # Configure Grid Layout (1 Row, 2 Columns: Video Feed and Sidebar)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=3)  # Video feed panel
        self.grid_columnconfigure(1, weight=1)  # Diagnostics and controls panel

        self._build_widgets()

        # Handle window closure gracefully
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Start camera thread
        self._camera.start()

        # Start main frame processing loop
        self._is_running = True
        self._processing_loop()

    def _build_widgets(self) -> None:
        """Constructs and positions the dashboard layout widgets."""
        # =====================================================================
        # COLUMN 0: Video Display Panel
        # =====================================================================
        self.video_frame = ctk.CTkFrame(self, corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Label acting as the Canvas for video frames
        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="Initializing Webcam Feed...",
            font=ctk.CTkFont(family="Inter", size=16, weight="bold"),
        )
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # =====================================================================
        # COLUMN 1: Control Panel and Settings Sidebar
        # =====================================================================
        self.sidebar_frame = ctk.CTkFrame(self, corner_radius=15, width=280)
        self.sidebar_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # Application Title
        self.title_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="GESTURE DASHBOARD",
            font=ctk.CTkFont(family="Outfit", size=20, weight="bold"),
        )
        self.title_label.pack(pady=(20, 10))

        # Status Separator
        self.status_card = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.status_card.pack(fill="x", padx=15, pady=5)

        # 1. Detected Gesture Status
        self.gesture_title_lbl = ctk.CTkLabel(
            self.status_card,
            text="Detected Gesture",
            font=ctk.CTkFont(family="Inter", size=12),
            text_color="gray",
        )
        self.gesture_title_lbl.pack(anchor="w")

        self.gesture_val_lbl = ctk.CTkLabel(
            self.status_card,
            text="None",
            font=ctk.CTkFont(family="Inter", size=22, weight="bold"),
            text_color="#1FB5FF",
        )
        self.gesture_val_lbl.pack(anchor="w", pady=(0, 10))

        # 2. Executed Action Status
        self.action_title_lbl = ctk.CTkLabel(
            self.status_card,
            text="Last Action Triggered",
            font=ctk.CTkFont(family="Inter", size=12),
            text_color="gray",
        )
        self.action_title_lbl.pack(anchor="w")

        self.action_val_lbl = ctk.CTkLabel(
            self.status_card,
            text="None",
            font=ctk.CTkFont(family="Inter", size=16, weight="bold"),
            text_color="#00C853",
        )
        self.action_val_lbl.pack(anchor="w", pady=(0, 20))

        # 3. Diagnostic Info
        self.diag_card = ctk.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.diag_card.pack(fill="x", padx=15, pady=5)

        self.camera_fps_lbl = ctk.CTkLabel(
            self.diag_card,
            text="Camera FPS: 0.00",
            font=ctk.CTkFont(family="Inter", size=12),
        )
        self.camera_fps_lbl.pack(pady=4)

        self.process_fps_lbl = ctk.CTkLabel(
            self.diag_card,
            text="Processing FPS: 0.00",
            font=ctk.CTkFont(family="Inter", size=12),
        )
        self.process_fps_lbl.pack(pady=4)

        # 4. Interactive Controls
        self.controls_card = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.controls_card.pack(fill="x", padx=15, pady=(20, 10))

        self.toggle_detection_btn = ctk.CTkButton(
            self.controls_card,
            text="Pause Gesture Detection",
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            command=self._toggle_detection,
            font=ctk.CTkFont(family="Inter", size=12, weight="bold"),
        )
        self.toggle_detection_btn.pack(fill="x", pady=5)

        # 5. Gesture Cheat Sheet Panel
        self.instructions_lbl = ctk.CTkLabel(
            self.sidebar_frame,
            text="Supported Controls",
            font=ctk.CTkFont(family="Inter", size=13, weight="bold"),
        )
        self.instructions_lbl.pack(pady=(15, 5))

        self.cheat_sheet = ctk.CTkTextbox(
            self.sidebar_frame,
            font=ctk.CTkFont(family="Courier", size=11),
            height=160,
            corner_radius=8,
        )
        self.cheat_sheet.pack(fill="x", padx=15, pady=5)
        self._populate_cheat_sheet()

    def _populate_cheat_sheet(self) -> None:
        """Fills the text box instruction panel with gestural mapping details."""
        sheet_text = (
            "Closed Fist    -> Play/Pause\n"
            "Open Palm      -> Mute\n"
            "Thumb Up       -> Vol Up\n"
            "Thumb Down     -> Vol Down\n"
            "Victory Sign   -> Fullscreen\n"
            "Three Fingers  -> Theater Mode\n"
            "Four Fingers   -> Captions\n"
            "Swipe Right    -> Skip Fwd 10s\n"
            "Swipe Left     -> Skip Back 10s"
        )
        self.cheat_sheet.insert("0.0", sheet_text)
        self.cheat_sheet.configure(state="disabled")

    def _toggle_detection(self) -> None:
        """Toggles the processing of gestures from the camera frame stream."""
        self._is_detection_running = not self._is_detection_running
        if self._is_detection_running:
            self.toggle_detection_btn.configure(
                text="Pause Gesture Detection", fg_color="#D32F2F", hover_color="#B71C1C"
            )
            self.gesture_val_lbl.configure(text="None", text_color="#1FB5FF")
        else:
            self.toggle_detection_btn.configure(
                text="Resume Gesture Detection", fg_color="#388E3C", hover_color="#1B5E20"
            )
            self.gesture_val_lbl.configure(text="PAUSED", text_color="yellow")

    def _processing_loop(self) -> None:
        """Recursive framework loop pulling frames, executing models, and updating UI."""
        if not self._is_running:
            return

        # 1. Fetch latest raw frame from Camera Capture thread
        success, frame = self._camera.get_frame()

        if success and frame is not None:
            # Mirror the frame horizontally for intuitive visual alignment
            frame = cv2.flip(frame, 1)

            # 2. Hand tracking & landmark processing
            hand_landmarks: Optional[HandLandmarks] = None
            if self._is_detection_running:
                # Draws hand connection overlay onto the frame
                frame, hand_landmarks = self._detector.process_frame(frame)

                # 3. Gesture recognition algorithm
                h, w, _ = frame.shape
                gesture = self._recognizer.process_landmarks(hand_landmarks, w, h)

                # 4. Automate Keystroke actions
                triggered, action, key = self._controller.trigger_action(gesture)
                if triggered and action is not None:
                    # Display triggered values on dashboard UI
                    self.action_val_lbl.configure(
                        text=f"{action.value.upper()} ('{key}')"
                    )

                # Update live detected gesture dashboard panel
                if gesture != GestureName.NONE:
                    self.gesture_val_lbl.configure(text=gesture.value)
                else:
                    self.gesture_val_lbl.configure(text="None")
            else:
                self._controller.reset_state()
                self.gesture_val_lbl.configure(text="PAUSED")

            # 5. Display Diagnostics metrics
            camera_fps = self._camera.get_fps()
            proc_fps = self._process_fps_calculator.update()

            self.camera_fps_lbl.configure(text=f"Camera FPS: {camera_fps:.2f}")
            self.process_fps_lbl.configure(text=f"Processing FPS: {proc_fps:.2f}")

            # 6. OpenCV BGR to Tkinter PhotoImage conversion
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Resize the image to display boundaries
            target_w, target_h = self._config.gui.FEED_DISPLAY_SIZE
            resized_image = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=resized_image)

            # Update Canvas Label
            self.video_label.configure(image=imgtk, text="")
            self.video_label.image = imgtk

        # Schedule next update tick in 15 milliseconds (targeting ~60 FPS update rate)
        self.after(15, self._processing_loop)

    def _on_closing(self) -> None:
        """Handles graceful application teardown and resource release on closure."""
        logger.info("Closing application interface...")
        self._is_running = False

        # Stop camera acquisition threads
        self._camera.stop()

        # Stop hand detection pipeline
        self._detector.close()

        # Terminate tkinter main loop
        self.destroy()
        logger.info("Application interface destroyed cleanly.")
