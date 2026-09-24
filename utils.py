"""Mathematical and helper utilities for gesture control and coordinate mapping.

Contains only pure functions and state tracking helpers for math calculations,
coordinate transformations, smoothing filters, and frame metrics. This module
is completely decoupled from OpenCV, MediaPipe, and PyAutoGUI.
"""

import math
import time
from typing import List, Sequence, Tuple


def calculate_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Calculate the Euclidean distance between two N-dimensional points."""
    return math.dist(p1, p2)


def normalize_distance(value: float, reference_distance: float) -> float:
    """Normalize a distance value relative to a reference distance to achieve scale-invariance."""
    if reference_distance <= 0:
        return 0.0
    return value / reference_distance


def calculate_angle(
    p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]
) -> float:
    """Calculate the angle in degrees at vertex p2 between vectors p2->p1 and p2->p3.

    Args:
        p1: Start point.
        p2: Vertex point (where the angle is measured).
        p3: End point.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0

    cosine_angle = dot_product / (mag1 * mag2)
    # Clamp cosine to avoid floating-point errors outside the mathematical range of [-1.0, 1.0]
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def is_finger_extended(
    tip_y: float, joint_y: float, threshold: float = 0.0, inverted: bool = False
) -> bool:
    """Determine finger extension based on normalized vertical coordinates.

    Note: In screen space, y-coordinates increase downwards. Therefore, a finger is
    considered extended when the tip's y-coordinate is smaller than the joint's y-coordinate.

    Args:
        tip_y: The y-coordinate of the finger's tip landmark.
        joint_y: The y-coordinate of the joint landmark (PIP or MCP).
        threshold: Minimum pixel or normalized distance separation.
        inverted: Set to True if checking under an upside-down orientation.
    """
    if inverted:
        return tip_y - joint_y > threshold
    return joint_y - tip_y > threshold


def calculate_bounding_box(
    landmarks: List[Tuple[float, float]],
    frame_width: int,
    frame_height: int,
    margin_px: int = 20,
) -> Tuple[int, int, int, int]:
    """Compute a pixel bounding box around a set of normalized 2D landmarks.

    Args:
        landmarks: List of normalized (x, y) coordinates from MediaPipe.
        frame_width: Horizontal resolution of the video frame.
        frame_height: Vertical resolution of the video frame.
        margin_px: Padding around the bounding box.

    Returns:
        A tuple of (xmin, ymin, xmax, ymax) pixel boundary coordinates.
    """
    if not landmarks:
        return 0, 0, 0, 0

    xs = [pt[0] for pt in landmarks]
    ys = [pt[1] for pt in landmarks]

    xmin = int(min(xs) * frame_width) - margin_px
    ymin = int(min(ys) * frame_height) - margin_px
    xmax = int(max(xs) * frame_width) + margin_px
    ymax = int(max(ys) * frame_height) + margin_px

    # Clamp coordinates to frame dimensions
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(frame_width, xmax)
    ymax = min(frame_height, ymax)

    return xmin, ymin, xmax, ymax


def denormalize_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """Convert normalized (0.0 to 1.0) coordinates to screen-space pixel coordinates."""
    return int(x * width), int(y * height)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a numerical value between a minimum and a maximum boundary."""
    return max(min_val, min(value, max_val))


def get_current_timestamp() -> float:
    """Get high-resolution monotonic time in seconds."""
    return time.perf_counter()


class FPSCalculator:
    """Calculates frame rendering and capture rates using a sliding time window."""

    def __init__(self, buffer_size: int = 30) -> None:
        self.buffer_size = buffer_size
        self.frame_times: List[float] = []

    def update(self) -> float:
        """Register the arrival of a new frame and compute current FPS.

        Returns:
            The rolling frames-per-second average.
        """
        now = get_current_timestamp()
        self.frame_times.append(now)

        if len(self.frame_times) > self.buffer_size:
            self.frame_times.pop(0)

        if len(self.frame_times) < 2:
            return 0.0

        total_duration = self.frame_times[-1] - self.frame_times[0]
        if total_duration <= 0.0:
            return 0.0

        return (len(self.frame_times) - 1) / total_duration


class MovingAverageFilter:
    """Smooths a stream of scalar floating-point values."""

    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.buffer: List[float] = []

    def filter(self, value: float) -> float:
        """Appends a value to the buffer and returns the moving average.

        Args:
            value: The latest raw value.
        """
        self.buffer.append(value)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        return sum(self.buffer) / len(self.buffer)


class PointSmoother:
    """Applies moving average filtering to 2D coordinate streams to reduce hand jitter."""

    def __init__(self, buffer_size: int = 5) -> None:
        self.buffer_size = buffer_size
        self.x_history: List[float] = []
        self.y_history: List[float] = []

    def smooth(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """Smooths a 2D point coordinate.

        Args:
            pt: A tuple of (x, y) coordinates.

        Returns:
            A tuple of filtered (x, y) coordinates.
        """
        self.x_history.append(pt[0])
        self.y_history.append(pt[1])

        if len(self.x_history) > self.buffer_size:
            self.x_history.pop(0)
            self.y_history.pop(0)

        smoothed_x = sum(self.x_history) / len(self.x_history)
        smoothed_y = sum(self.y_history) / len(self.y_history)

        return smoothed_x, smoothed_y
