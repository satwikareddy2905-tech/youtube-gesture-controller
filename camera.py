"""Webcam frame capture module for the YouTube Gesture Controller.

Handles threaded webcam access, reconnection logic, thread-safe frame retrieval,
and frame rate measurement using OpenCV and utility helpers.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from config import AppConfig
from logger import get_logger
from utils import FPSCalculator

logger = get_logger(__name__)


class Camera:
    """Manages threaded capture from a webcam device."""

    def __init__(self, config: AppConfig) -> None:
        """Initializes the Camera controller with given settings.

        Args:
            config: Main application configuration.
        """
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._fps: float = 0.0
        self._fps_calculator = FPSCalculator(buffer_size=30)

        self._is_running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_connected: bool = False

    def start(self) -> None:
        """Starts the background frame capture thread."""
        if self._is_running:
            logger.warning("Camera thread is already running.")
            return

        self._is_running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="CameraCaptureThread", daemon=True
        )
        self._thread.start()
        logger.info("Camera capture thread started.")

    def stop(self) -> None:
        """Stops the background capture thread and releases webcam resources."""
        if not self._is_running:
            return

        logger.info("Stopping camera capture thread...")
        self._is_running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self._release_cap()
        logger.info("Camera capture thread stopped and resources released.")

    def _initialize_camera(self) -> bool:
        """Attempts to open the camera device and set properties.

        Returns:
            True if initialized successfully, False otherwise.
        """
        self._release_cap()
        device_idx = self._config.camera.DEVICE_INDEX

        try:
            # CAP_DSHOW prevents long startup delays on Windows
            self._cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW)
            if not self._cap or not self._cap.isOpened():
                # Fall back to default backend if CAP_DSHOW fails
                self._cap = cv2.VideoCapture(device_idx)

            if not self._cap or not self._cap.isOpened():
                logger.error(f"Failed to open video capture device at index {device_idx}")
                return False

            # Configure resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera.FRAME_HEIGHT)

            # Log actual dimensions set by the hardware
            actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            logger.info(
                f"Webcam opened at index {device_idx}. Target resolution: "
                f"{self._config.camera.FRAME_WIDTH}x{self._config.camera.FRAME_HEIGHT}. "
                f"Actual resolution set: {actual_w}x{actual_h}"
            )

            with self._lock:
                self._is_connected = True
            return True

        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            with self._lock:
                self._is_connected = False
            return False

    def _release_cap(self) -> None:
        """Safely releases the OpenCV VideoCapture resource."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
            finally:
                self._cap = None

        with self._lock:
            self._is_connected = False
            self._latest_frame = None

    def _capture_loop(self) -> None:
        """Continuously reads frames from the camera in a background loop."""
        reconnect_delay_sec = 2.0

        while self._is_running:
            # Handle connection or reconnection
            if self._cap is None or not self._cap.isOpened():
                logger.info("Attempting to connect/reconnect to camera...")
                if not self._initialize_camera():
                    time.sleep(reconnect_delay_sec)
                    continue

            # Read frame
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Webcam disconnected or failed to read frame. Retrying...")
                self._release_cap()
                time.sleep(reconnect_delay_sec)
                continue

            # Store the frame thread-safely
            with self._lock:
                # Store a copy to prevent data racing when accessed from main thread
                self._latest_frame = frame.copy()
                self._fps = self._fps_calculator.update()

            # Yield control slightly to avoid CPU starvation
            time.sleep(0.001)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the latest captured frame thread-safely.

        Returns:
            A tuple of (success, frame).
        """
        with self._lock:
            if not self._is_connected or self._latest_frame is None:
                return False, None
            return True, self._latest_frame

    def get_fps(self) -> float:
        """Retrieves the current frame capture rate thread-safely."""
        with self._lock:
            return self._fps

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and capturing thread-safely."""
        with self._lock:
            return self._is_connected
