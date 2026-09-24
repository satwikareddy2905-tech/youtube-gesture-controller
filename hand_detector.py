"""Hand detection module using MediaPipe Hands.

Provides real-time hand landmark tracking, boundary box computation, and
handedness classification. Integrates with the application config and returns
strongly-typed hand landmark data.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig
from logger import get_logger
from utils import calculate_bounding_box

logger = get_logger(__name__)


@dataclass(frozen=True)
class HandLandmarks:
    """Structure representing 21 normalized 2D landmarks and 3D world landmarks."""

    # List of 21 normalized (x, y) coordinate tuples
    landmarks_2d: List[Tuple[float, float]] = field(default_factory=list)
    # List of 21 metric (x, y, z) 3D coordinate tuples representing physical space (in meters)
    landmarks_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    # "Left" or "Right" classification relative to the user
    handedness: str = "Unknown"
    # Bounding box in pixel coordinates (xmin, ymin, xmax, ymax)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


class HandDetector:
    """Wrapper around the MediaPipe Hands solution."""

    def __init__(self, config: AppConfig) -> None:
        """Initializes the HandDetector with specific configurations.

        Args:
            config: Aggregated application settings.
        """
        self._config = config

        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            logger.info("Downloading hand_landmarker.task model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(url, model_path)
                logger.info("Download complete.")
            except Exception as e:
                logger.error(f"Failed to download hand_landmarker.task: {e}")
                raise e

        # Initialize MediaPipe Tasks HandLandmarker
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=self._config.detector.MAX_NUM_HANDS,
            min_hand_detection_confidence=self._config.detector.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=self._config.detector.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self._config.detector.MIN_TRACKING_CONFIDENCE,
        )
        self._hands = HandLandmarker.create_from_options(options)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[HandLandmarks]]:
        """Processes an OpenCV BGR frame for hand landmark detection.

        Args:
            frame: Raw BGR numpy image array.

        Returns:
            A tuple containing:
                1. The output BGR frame with visual annotations drawn.
                2. An Optional[HandLandmarks] dataclass containing hand metrics.
        """
        if frame is None:
            return frame, None

        # MediaPipe requires RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Convert numpy RGB frame to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self._hands.detect(mp_image)
        except Exception as e:
            logger.error(f"Error processing frame in MediaPipe: {e}")
            return frame, None

        annotated_frame = frame.copy()
        hand_landmarks_result: Optional[HandLandmarks] = None

        if results.hand_landmarks and results.handedness:
            # Process only the primary (first) detected hand based on configured constraints
            landmarks = results.hand_landmarks[0]
            world_landmarks = results.hand_world_landmarks[0]
            handedness_info = results.handedness[0]

            # 1. Parse Handedness
            # Note: MediaPipe processes in mirrored/camera space.
            # We map this based on classification label.
            handedness = handedness_info[0].category_name

            # 2. Extract 2D & 3D Coordinates
            landmarks_2d = [(lm.x, lm.y) for lm in landmarks]
            landmarks_3d = [(wlm.x, wlm.y, wlm.z) for wlm in world_landmarks]

            # 3. Calculate Bounding Box
            h, w, _ = frame.shape
            bbox = calculate_bounding_box(landmarks_2d, w, h, margin_px=20)

            hand_landmarks_result = HandLandmarks(
                landmarks_2d=landmarks_2d,
                landmarks_3d=landmarks_3d,
                handedness=handedness,
                bbox=bbox,
            )

            # 4. Render Annotations
            # Draw Hand landmarks using new Tasks drawing utilities
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles
            mp_hand_connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            mp_drawing.draw_landmarks(
                annotated_frame,
                landmarks,
                mp_hand_connections,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Draw Custom Bounding Box with Label
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{handedness} Hand",
                (xmin, max(ymin - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated_frame, hand_landmarks_result

    def close(self) -> None:
        """Closes the MediaPipe Hands object and releases internal resources."""
        try:
            self._hands.close()
            logger.info("MediaPipe Hands resource closed successfully.")
        except Exception as e:
            logger.error(f"Error closing MediaPipe Hands: {e}")
