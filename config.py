"""Configuration module for the YouTube Gesture Controller.

Defines all constants, settings, thresholds, keybindings, and UI configurations
using typed dataclasses and Enums to ensure type safety and readability.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple


class GestureName(str, Enum):
    """Supported hand gestures for control."""

    CLOSED_FIST = "Closed Fist"
    OPEN_PALM = "Open Palm"
    THUMB_UP = "Thumb Up"
    THUMB_DOWN = "Thumb Down"
    VICTORY = "Victory Sign"
    THREE_FINGERS = "Three Fingers"
    FOUR_FINGERS = "Four Fingers"
    SWIPE_RIGHT = "Swipe Right"
    SWIPE_LEFT = "Swipe Left"
    NONE = "None"


class KeyboardAction(str, Enum):
    """Mapped actions matching YouTube keyboard shortcuts."""

    PLAY_PAUSE = "play_pause"
    MUTE = "mute"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    FULLSCREEN = "fullscreen"
    THEATER_MODE = "theater_mode"
    TOGGLE_CAPTIONS = "toggle_captions"
    FORWARD_10S = "forward_10s"
    BACKWARD_10S = "backward_10s"


@dataclass(frozen=True)
class CameraSettings:
    """Configurations for OpenCV camera capture."""

    DEVICE_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30


@dataclass(frozen=True)
class HandDetectorSettings:
    """Configurations for MediaPipe Hand landmark detector."""

    MAX_NUM_HANDS: int = 1
    MODEL_COMPLEXITY: int = 1
    MIN_DETECTION_CONFIDENCE: float = 0.7
    MIN_TRACKING_CONFIDENCE: float = 0.7


@dataclass(frozen=True)
class GestureSettings:
    """Thresholds and timing parameters for gesture recognition."""

    # Stabilization, Smoothing & Hysteresis State Machine
    DEBOUNCE_FRAMES: int = 4  # Min consecutive frames required to transition static gesture
    SMOOTHING_WINDOW_SIZE: int = 5  # Rolling window size for temporal candidate voting
    HYSTERESIS_EXIT_FRAMES: int = 3  # Consecutive mismatches required to drop active gesture
    CONFIDENCE_THRESHOLD: float = 0.60  # Minimum confidence score to accept candidate gesture

    # Normalized Swipe Detection (scale-invariant, relative to palm size)
    SWIPE_MIN_NORMALIZED_DIST: float = 0.85  # Min displacement as a multiple of palm scale
    SWIPE_MAX_NORMALIZED_DEV: float = 0.65   # Max vertical deviation as a multiple of palm scale
    SWIPE_MIN_DISTANCE_PX: int = 80          # Fallback min pixel displacement
    SWIPE_MAX_VERTICAL_DEVIATION_PX: int = 70  # Fallback max vertical pixel deviation
    SWIPE_BUFFER_SIZE: int = 12              # Frame buffer length for checking swipes

    # Gesture Cooldown Settings (seconds)
    ACTION_COOLDOWN_SEC: float = 1.0  # Default cooldown for state toggles (e.g. fullscreen)
    VOLUME_COOLDOWN_SEC: float = 0.2  # Shorter cooldown for responsive volume control
    SWIPE_COOLDOWN_SEC: float = 0.5   # Cooldown between consecutive swipe-seek actions

    # Debugging
    DEBUG_GESTURES: bool = False      # Verbose gesture debug logging toggle


@dataclass(frozen=True)
class GUISettings:
    """Styling and window settings for the CustomTkinter GUI."""

    TITLE: str = "YouTube Gesture Controller"
    WINDOW_SIZE: str = "1100x650"
    THEME_MODE: str = "dark"  # Options: "dark", "light", "system"
    COLOR_THEME: str = "blue"  # Options: "blue", "green", "dark-blue"
    FEED_DISPLAY_SIZE: Tuple[int, int] = (640, 480)


@dataclass
class AppConfig:
    """Main application configuration aggregator."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: HandDetectorSettings = field(default_factory=HandDetectorSettings)
    gesture: GestureSettings = field(default_factory=GestureSettings)
    gui: GUISettings = field(default_factory=GUISettings)
    DEBUG: bool = False

    # Key mapping of action to PyAutoGUI key string.
    # Note: 'k' toggles play/pause, 'm' toggles mute, 'f' toggles fullscreen,
    # 't' theater mode, 'c' captions, 'l' jumps 10s forward, 'j' jumps 10s backward,
    # 'up' and 'down' control system/browser volume.
    KEY_MAPPINGS: Dict[KeyboardAction, str] = field(
        default_factory=lambda: {
            KeyboardAction.PLAY_PAUSE: "k",
            KeyboardAction.MUTE: "m",
            KeyboardAction.VOLUME_UP: "up",
            KeyboardAction.VOLUME_DOWN: "down",
            KeyboardAction.FULLSCREEN: "f",
            KeyboardAction.THEATER_MODE: "t",
            KeyboardAction.TOGGLE_CAPTIONS: "c",
            KeyboardAction.FORWARD_10S: "l",
            KeyboardAction.BACKWARD_10S: "j",
        }
    )

    # Maps each physical GestureName to the logical KeyboardAction
    GESTURE_ACTION_MAPPING: Dict[GestureName, KeyboardAction] = field(
        default_factory=lambda: {
            GestureName.CLOSED_FIST: KeyboardAction.PLAY_PAUSE,
            GestureName.OPEN_PALM: KeyboardAction.MUTE,
            GestureName.THUMB_UP: KeyboardAction.VOLUME_UP,
            GestureName.THUMB_DOWN: KeyboardAction.VOLUME_DOWN,
            GestureName.VICTORY: KeyboardAction.FULLSCREEN,
            GestureName.THREE_FINGERS: KeyboardAction.THEATER_MODE,
            GestureName.FOUR_FINGERS: KeyboardAction.TOGGLE_CAPTIONS,
            GestureName.SWIPE_RIGHT: KeyboardAction.FORWARD_10S,
            GestureName.SWIPE_LEFT: KeyboardAction.BACKWARD_10S,
        }
    )
