"""Gesture recognition module for the YouTube Gesture Controller.

Systematic rule-based geometric algorithm on 21 hand landmarks.
Provides rotation-robust scale-invariant finger classification, confidence
scoring, temporal voting, hysteresis state machine stabilization, and scale-normalized
swipe tracking.
"""

from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

from config import AppConfig, GestureName
from hand_detector import HandLandmarks
from logger import get_logger
from utils import calculate_angle, calculate_distance

logger = get_logger(__name__)


class GestureRecognizer:
    """Processes hand landmarks to recognize static and dynamic gestures reliably."""

    def __init__(self, config: AppConfig) -> None:
        """Initializes the GestureRecognizer.

        Args:
            config: Aggregated application settings.
        """
        self._config = config

        # Buffer for centroid history: stores tuples of (x_px, y_px, palm_scale_px)
        self._centroid_history: deque = deque(
            maxlen=self._config.gesture.SWIPE_BUFFER_SIZE
        )

        # Temporal candidate history buffer for temporal voting smoothing
        self._candidate_history: deque = deque(
            maxlen=self._config.gesture.SMOOTHING_WINDOW_SIZE
        )

        # Hysteresis State Machine Tracking
        self._stable_gesture: GestureName = GestureName.NONE
        self._pending_candidate: GestureName = GestureName.NONE
        self._candidate_match_count: int = 0
        self._mismatch_count: int = 0

        # Detailed analysis data for debugging and testing harness
        self._last_analysis: Dict[str, Any] = {
            "finger_states": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
            "finger_scores": {"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "thumb": 0.0},
            "candidate_gesture": GestureName.NONE,
            "candidate_confidence": 0.0,
            "stable_gesture": GestureName.NONE,
            "palm_scale": 0.0,
        }

    def process_landmarks(
        self, hand_landmarks: HandLandmarks, frame_width: int, frame_height: int
    ) -> GestureName:
        """Analyzes landmarks from a single frame to identify the active gesture.

        Args:
            hand_landmarks: Current detected hand landmarks structure.
            frame_width: Width of the video frame in pixels.
            frame_height: Height of the video frame in pixels.

        Returns:
            The recognized, debounced, and stabilized GestureName.
        """
        if not hand_landmarks or not hand_landmarks.landmarks_2d:
            self._reset_state()
            return GestureName.NONE

        landmarks = hand_landmarks.landmarks_2d

        # 1. Compute Palm Scale Reference (Wrist[0] to Middle MCP[9])
        palm_scale = calculate_distance(landmarks[0], landmarks[9])
        if palm_scale <= 1e-6:
            palm_scale = 1.0

        # 2. Update Centroid History (using Middle Finger MCP [9] as anchor)
        centroid_norm = landmarks[9]
        centroid_px = (int(centroid_norm[0] * frame_width), int(centroid_norm[1] * frame_height))
        palm_scale_px = palm_scale * max(frame_width, frame_height)
        self._centroid_history.append((centroid_px[0], centroid_px[1], palm_scale_px))

        # 3. Check for dynamic swipe gestures first (high priority)
        swipe_gesture = self._detect_swipe(palm_scale_px)
        if swipe_gesture != GestureName.NONE:
            # Swipes are immediate dynamic triggers; clear history to prevent duplicate firing
            self._centroid_history.clear()
            self._candidate_history.clear()
            self._last_analysis["candidate_gesture"] = swipe_gesture
            self._last_analysis["candidate_confidence"] = 1.0
            self._last_analysis["stable_gesture"] = swipe_gesture
            return swipe_gesture

        # 4. Classify Finger Extension States & Scores using rotation-invariant geometry
        finger_states, finger_scores = self._classify_finger_states(landmarks, palm_scale)

        # 5. Candidate Static Gesture Classification & Confidence Scoring
        candidate_gesture, confidence = self._evaluate_candidate_gestures(
            finger_states, finger_scores, landmarks, palm_scale
        )

        # Filter candidates below confidence threshold
        if confidence < self._config.gesture.CONFIDENCE_THRESHOLD:
            candidate_gesture = GestureName.NONE

        self._candidate_history.append(candidate_gesture)

        # 6. Temporal Voting (Majority vote across candidate history window)
        top_voted_gesture = self._apply_majority_voting()

        # 7. Hysteresis State Machine Update
        stable_gesture = self._update_hysteresis_state(top_voted_gesture)

        # Save analysis context for debug logging and test harness
        self._last_analysis = {
            "finger_states": finger_states,
            "finger_scores": finger_scores,
            "candidate_gesture": candidate_gesture,
            "candidate_confidence": confidence,
            "top_voted_gesture": top_voted_gesture,
            "stable_gesture": stable_gesture,
            "palm_scale": palm_scale,
        }

        if self._config.gesture.DEBUG_GESTURES:
            logger.debug(
                f"[GESTURE] Cand: {candidate_gesture.value} ({confidence:.2f}) | "
                f"Voted: {top_voted_gesture.value} | Stable: {stable_gesture.value} | "
                f"Fingers: T={finger_states['thumb']}, I={finger_states['index']}, "
                f"M={finger_states['middle']}, R={finger_states['ring']}, P={finger_states['pinky']}"
            )

        return stable_gesture

    def get_last_analysis_details(self) -> Dict[str, Any]:
        """Returns analysis details from the most recent processing tick."""
        return dict(self._last_analysis)

    def _reset_state(self) -> None:
        """Resets history queues and hysteresis state machine counters."""
        self._centroid_history.clear()
        self._candidate_history.clear()
        self._stable_gesture = GestureName.NONE
        self._pending_candidate = GestureName.NONE
        self._candidate_match_count = 0
        self._mismatch_count = 0
        self._last_analysis = {
            "finger_states": {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False},
            "finger_scores": {"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0, "thumb": 0.0},
            "candidate_gesture": GestureName.NONE,
            "candidate_confidence": 0.0,
            "stable_gesture": GestureName.NONE,
            "palm_scale": 0.0,
        }

    def _classify_finger_states(
        self, landmarks: List[Tuple[float, float]], palm_scale: float
    ) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Computes rotation-invariant finger extension states and continuous scores.

        Uses multiple geometric signals:
        1. PIP Joint Angle (MCP -> PIP -> Tip): straight angle (> 135 deg) vs folded (< 115 deg).
        2. Distance Ratio: Tip-to-Wrist distance vs PIP-to-Wrist distance.
        3. Extension Ratio: Tip-to-MCP distance normalized by palm scale.

        Args:
            landmarks: 21 normalized 2D landmarks.
            palm_scale: Distance from Wrist (0) to Middle MCP (9).

        Returns:
            Tuple of (boolean_finger_states_dict, float_finger_scores_dict).
        """
        states: Dict[str, bool] = {}
        scores: Dict[str, float] = {}

        # 1. Four Main Fingers (Index, Middle, Ring, Pinky)
        finger_indices = {
            "index": (5, 6, 7, 8),
            "middle": (9, 10, 11, 12),
            "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20),
        }

        wrist = landmarks[0]

        for finger_name, (mcp_idx, pip_idx, dip_idx, tip_idx) in finger_indices.items():
            mcp = landmarks[mcp_idx]
            pip = landmarks[pip_idx]
            tip = landmarks[tip_idx]

            # Signal A: PIP Joint Angle (degrees)
            pip_angle = calculate_angle(mcp, pip, tip)
            angle_score = max(0.0, min(1.0, (pip_angle - 110.0) / (160.0 - 110.0)))

            # Signal B: Tip-to-Wrist vs PIP-to-Wrist distance ratio
            dist_tip_wrist = calculate_distance(tip, wrist)
            dist_pip_wrist = calculate_distance(pip, wrist)
            ratio = dist_tip_wrist / max(dist_pip_wrist, 1e-6)
            ratio_score = max(0.0, min(1.0, (ratio - 0.95) / (1.25 - 0.95)))

            # Signal C: Tip-to-MCP distance relative to palm scale
            dist_tip_mcp = calculate_distance(tip, mcp)
            mcp_score = max(0.0, min(1.0, (dist_tip_mcp / palm_scale - 0.35) / (0.80 - 0.35)))

            # Combined weighted confidence score for finger extension
            score = 0.45 * angle_score + 0.35 * ratio_score + 0.20 * mcp_score
            scores[finger_name] = round(score, 3)

            # Strict threshold with tolerance margin
            states[finger_name] = score > 0.50

        # 2. Thumb Analysis
        thumb_state, thumb_score = self._analyze_thumb_extension(landmarks, palm_scale)
        states["thumb"] = thumb_state
        scores["thumb"] = round(thumb_score, 3)

        return states, scores

    def _analyze_thumb_extension(
        self, landmarks: List[Tuple[float, float]], palm_scale: float
    ) -> Tuple[bool, float]:
        """Analyzes thumb extension and spread relative to palm geometry.

        Args:
            landmarks: 21 normalized 2D landmarks.
            palm_scale: Palm reference distance.

        Returns:
            Tuple of (is_thumb_extended, thumb_extension_score).
        """
        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        # Metric 1: Distance from Thumb Tip to Index MCP relative to palm scale
        dist_tip_index_mcp = calculate_distance(thumb_tip, index_mcp) / palm_scale

        # Metric 2: Distance from Thumb Tip to Thumb MCP relative to palm scale
        dist_tip_thumb_mcp = calculate_distance(thumb_tip, thumb_mcp) / palm_scale

        # Metric 3: Distance from Thumb Tip to Wrist relative to Thumb MCP to Wrist
        dist_tip_wrist = calculate_distance(thumb_tip, wrist)
        dist_mcp_wrist = calculate_distance(thumb_mcp, wrist)
        wrist_ratio = dist_tip_wrist / max(dist_mcp_wrist, 1e-6)

        s1 = max(0.0, min(1.0, (dist_tip_index_mcp - 0.22) / (0.45 - 0.22)))
        s2 = max(0.0, min(1.0, (dist_tip_thumb_mcp - 0.25) / (0.50 - 0.25)))
        s3 = max(0.0, min(1.0, (wrist_ratio - 1.0) / (1.4 - 1.0)))

        score = max(s1, 0.6 * s2 + 0.4 * s3)
        is_extended = (dist_tip_index_mcp > 0.28) or (dist_tip_thumb_mcp > 0.32) or (score > 0.50)

        return is_extended, score

    def _evaluate_candidate_gestures(
        self,
        finger_states: Dict[str, bool],
        finger_scores: Dict[str, float],
        landmarks: List[Tuple[float, float]],
        palm_scale: float,
    ) -> Tuple[GestureName, float]:
        """Evaluates geometric confidence for candidate static gestures.

        Args:
            finger_states: Boolean extended states of all 5 fingers.
            finger_scores: Continuous extension scores [0.0..1.0].
            landmarks: Normalized 2D landmarks.
            palm_scale: Palm reference distance.

        Returns:
            Tuple of (Candidate GestureName, Confidence Score [0.0..1.0]).
        """
        thumb = finger_states["thumb"]
        index = finger_states["index"]
        middle = finger_states["middle"]
        ring = finger_states["ring"]
        pinky = finger_states["pinky"]

        extended_count = sum([index, middle, ring, pinky])

        # 1. Check Thumb Up / Thumb Down / Closed Fist when 4 main fingers are folded
        if extended_count == 0:
            return self._classify_thumb_or_fist(landmarks, palm_scale, finger_scores)

        # 2. Victory Sign: Index & Middle extended; Ring & Pinky folded
        if index and middle and not ring and not pinky:
            conf = 0.5 * finger_scores["index"] + 0.5 * finger_scores["middle"]
            # Penalty if ring or pinky have high extension scores
            conf *= (1.0 - 0.5 * finger_scores["ring"]) * (1.0 - 0.5 * finger_scores["pinky"])
            return GestureName.VICTORY, max(0.0, min(1.0, conf))

        # 3. Three Fingers: Index, Middle & Ring extended; Pinky folded
        if index and middle and ring and not pinky:
            conf = (finger_scores["index"] + finger_scores["middle"] + finger_scores["ring"]) / 3.0
            conf *= (1.0 - 0.6 * finger_scores["pinky"])
            return GestureName.THREE_FINGERS, max(0.0, min(1.0, conf))

        # 4. Four Fingers vs Open Palm: All 4 main fingers extended
        if extended_count == 4:
            avg_four = (
                finger_scores["index"]
                + finger_scores["middle"]
                + finger_scores["ring"]
                + finger_scores["pinky"]
            ) / 4.0

            if thumb:
                conf = 0.7 * avg_four + 0.3 * finger_scores["thumb"]
                return GestureName.OPEN_PALM, max(0.0, min(1.0, conf))
            else:
                conf = 0.8 * avg_four + 0.2 * (1.0 - finger_scores["thumb"])
                return GestureName.FOUR_FINGERS, max(0.0, min(1.0, conf))

        # 5. Fallback for 3 fingers extended with minor noise
        if extended_count == 3 and index and middle and ring:
            return GestureName.THREE_FINGERS, 0.60

        return GestureName.NONE, 0.0

    def _classify_thumb_or_fist(
        self,
        landmarks: List[Tuple[float, float]],
        palm_scale: float,
        finger_scores: Dict[str, float],
    ) -> Tuple[GestureName, float]:
        """Distinguishes Thumb Up, Thumb Down, and Closed Fist when main 4 fingers are folded."""
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        wrist = landmarks[0]

        # Calculate vertical positioning in screen coordinates (smaller Y is higher up)
        # Thumb Tip relative to Index MCP and Thumb MCP normalized by palm scale
        dy_index = (index_mcp[1] - thumb_tip[1]) / palm_scale
        dy_mcp = (thumb_mcp[1] - thumb_tip[1]) / palm_scale

        # Inverse main finger fold quality
        fold_quality = sum(1.0 - finger_scores[f] for f in ["index", "middle", "ring", "pinky"]) / 4.0

        # Thumb Up: Thumb Tip is pointing UP (smaller Y than Index MCP & Thumb MCP)
        if dy_index > 0.15 and dy_mcp > 0.05 and thumb_tip[1] < wrist[1]:
            conf = min(1.0, 0.5 * fold_quality + 0.5 * min(1.0, dy_index / 0.4))
            return GestureName.THUMB_UP, conf

        # Thumb Down: Thumb Tip is pointing DOWN (larger Y than Index MCP & Thumb MCP)
        elif dy_index < -0.25 and dy_mcp < -0.10 and thumb_tip[1] > wrist[1]:
            conf = min(1.0, 0.5 * fold_quality + 0.5 * min(1.0, abs(dy_index) / 0.5))
            return GestureName.THUMB_DOWN, conf

        # Otherwise: Closed Fist
        conf = max(0.65, fold_quality)
        return GestureName.CLOSED_FIST, conf

    def _detect_swipe(self, palm_scale_px: float = 100.0) -> GestureName:
        """Analyzes normalized centroid displacement to detect rapid scale-invariant swipes.

        Args:
            palm_scale_px: Current palm reference scale in pixels (defaults to 100.0).

        Returns:
            GestureName.SWIPE_RIGHT, GestureName.SWIPE_LEFT, or GestureName.NONE.
        """
        buffer_len = len(self._centroid_history)
        min_required = max(2, self._config.gesture.SWIPE_BUFFER_SIZE // 2)
        if buffer_len < min_required:
            return GestureName.NONE

        xs = [pt[0] for pt in self._centroid_history]
        ys = [pt[1] for pt in self._centroid_history]
        scales = [pt[2] if len(pt) > 2 else palm_scale_px for pt in self._centroid_history]

        avg_scale = sum(scales) / len(scales) if scales else palm_scale_px
        if avg_scale <= 1.0:
            avg_scale = 100.0  # Fallback scale

        peak_dx_px = max(xs) - min(xs)
        peak_dy_px = max(ys) - min(ys)

        # Normalized displacements relative to palm scale
        norm_dx = peak_dx_px / avg_scale
        norm_dy = peak_dy_px / avg_scale

        min_norm_dist = self._config.gesture.SWIPE_MIN_NORMALIZED_DIST
        max_norm_dev = self._config.gesture.SWIPE_MAX_NORMALIZED_DEV

        # Check pixel fallbacks as well as normalized thresholds
        is_far_enough = (norm_dx >= min_norm_dist) or (peak_dx_px >= self._config.gesture.SWIPE_MIN_DISTANCE_PX)
        is_straight_enough = (norm_dy <= max_norm_dev) and (peak_dy_px <= self._config.gesture.SWIPE_MAX_VERTICAL_DEVIATION_PX)

        if not (is_far_enough and is_straight_enough):
            return GestureName.NONE

        # Directional Step Consistency (≥ 60% of frame-to-frame steps agree)
        pts = list(self._centroid_history)
        right_steps = sum(1 for i in range(1, len(pts)) if pts[i][0] > pts[i - 1][0])
        left_steps = sum(1 for i in range(1, len(pts)) if pts[i][0] < pts[i - 1][0])
        total_steps = len(pts) - 1

        if total_steps == 0:
            return GestureName.NONE

        if right_steps / total_steps >= 0.60:
            logger.info(
                f"[SWIPE] Scale-Invariant Swipe Right: norm_dx={norm_dx:.2f}, "
                f"peak_dx={peak_dx_px:.0f}px, steps={right_steps}/{total_steps}"
            )
            return GestureName.SWIPE_RIGHT

        elif left_steps / total_steps >= 0.60:
            logger.info(
                f"[SWIPE] Scale-Invariant Swipe Left: norm_dx={norm_dx:.2f}, "
                f"peak_dx={peak_dx_px:.0f}px, steps={left_steps}/{total_steps}"
            )
            return GestureName.SWIPE_LEFT

        return GestureName.NONE

    def _apply_majority_voting(self) -> GestureName:
        """Computes the majority-voted candidate gesture across candidate history."""
        if not self._candidate_history:
            return GestureName.NONE

        counts = Counter(self._candidate_history)
        # Get most common candidate
        top_gesture, top_count = counts.most_common(1)[0]
        return top_gesture

    def _update_hysteresis_state(self, candidate: GestureName) -> GestureName:
        """Updates the hysteresis state machine to prevent single-frame dropouts or flickering.

        Transitions:
        - If candidate matches stable gesture: reset mismatch counter.
        - If candidate differs: increment mismatch counter.
        - Transition to candidate occurs ONLY when candidate is consistent for
          DEBOUNCE_FRAMES or when mismatch counter exceeds HYSTERESIS_EXIT_FRAMES.

        Args:
            candidate: Top voted gesture from candidate voting.

        Returns:
            The stabilized output GestureName.
        """
        if candidate == self._stable_gesture:
            self._mismatch_count = 0
            self._pending_candidate = GestureName.NONE
            self._candidate_match_count = 0
            return self._stable_gesture

        # Candidate is different from current stable gesture
        if candidate == self._pending_candidate:
            self._candidate_match_count += 1
        else:
            self._pending_candidate = candidate
            self._candidate_match_count = 1

        self._mismatch_count += 1

        # Check transition criteria
        if (
            self._candidate_match_count >= self._config.gesture.DEBOUNCE_FRAMES
            or self._mismatch_count >= self._config.gesture.HYSTERESIS_EXIT_FRAMES
        ):
            self._stable_gesture = candidate
            self._pending_candidate = GestureName.NONE
            self._candidate_match_count = 0
            self._mismatch_count = 0

        return self._stable_gesture


def test_gesture_recognition() -> None:
    """Internal test suite creating synthetic 21-landmark sets to verify gesture classification."""
    config = AppConfig()
    recognizer = GestureRecognizer(config)

    def process_n_frames(hl: HandLandmarks, count: int = config.gesture.DEBOUNCE_FRAMES + 2) -> GestureName:
        result = GestureName.NONE
        for _ in range(count):
            result = recognizer.process_landmarks(hl, 640, 480)
        return result

    # 1. Open Palm
    open_palm_lm = [(0.0, 0.0)] * 21
    open_palm_lm[0] = (0.5, 0.8)   # Wrist
    open_palm_lm[1] = (0.45, 0.7)  # Thumb CMC
    open_palm_lm[2] = (0.4, 0.6)   # Thumb MCP
    open_palm_lm[3] = (0.3, 0.52)  # Thumb IP
    open_palm_lm[4] = (0.2, 0.45)  # Thumb Tip
    # Index (5, 6, 7, 8)
    open_palm_lm[5] = (0.4, 0.5)
    open_palm_lm[6] = (0.4, 0.38)
    open_palm_lm[7] = (0.4, 0.28)
    open_palm_lm[8] = (0.4, 0.18)
    # Middle (9, 10, 11, 12)
    open_palm_lm[9] = (0.5, 0.5)
    open_palm_lm[10] = (0.5, 0.35)
    open_palm_lm[11] = (0.5, 0.25)
    open_palm_lm[12] = (0.5, 0.15)
    # Ring (13, 14, 15, 16)
    open_palm_lm[13] = (0.6, 0.5)
    open_palm_lm[14] = (0.6, 0.38)
    open_palm_lm[15] = (0.6, 0.28)
    open_palm_lm[16] = (0.6, 0.18)
    # Pinky (17, 18, 19, 20)
    open_palm_lm[17] = (0.7, 0.52)
    open_palm_lm[18] = (0.7, 0.42)
    open_palm_lm[19] = (0.7, 0.32)
    open_palm_lm[20] = (0.7, 0.22)

    res = process_n_frames(HandLandmarks(landmarks_2d=open_palm_lm))
    assert res == GestureName.OPEN_PALM, f"Expected OPEN_PALM, got {res}"

    # 2. Closed Fist
    closed_fist_lm = list(open_palm_lm)
    # Fold thumb (4 near 5)
    closed_fist_lm[4] = (0.42, 0.52)
    # Fold index, middle, ring, pinky (tips near MCPs)
    for mcp_i, pip_i, dip_i, tip_i in [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]:
        closed_fist_lm[pip_i] = (closed_fist_lm[mcp_i][0], 0.48)
        closed_fist_lm[dip_i] = (closed_fist_lm[mcp_i][0], 0.52)
        closed_fist_lm[tip_i] = (closed_fist_lm[mcp_i][0], 0.55)

    res = process_n_frames(HandLandmarks(landmarks_2d=closed_fist_lm))
    assert res == GestureName.CLOSED_FIST, f"Expected CLOSED_FIST, got {res}"

    # 3. Thumb Up
    thumb_up_lm = list(closed_fist_lm)
    thumb_up_lm[3] = (0.35, 0.45)
    thumb_up_lm[4] = (0.3, 0.25)
    res = process_n_frames(HandLandmarks(landmarks_2d=thumb_up_lm))
    assert res == GestureName.THUMB_UP, f"Expected THUMB_UP, got {res}"

    # 4. Victory Sign
    victory_lm = list(closed_fist_lm)
    victory_lm[6] = open_palm_lm[6]
    victory_lm[7] = open_palm_lm[7]
    victory_lm[8] = open_palm_lm[8]
    victory_lm[10] = open_palm_lm[10]
    victory_lm[11] = open_palm_lm[11]
    victory_lm[12] = open_palm_lm[12]
    res = process_n_frames(HandLandmarks(landmarks_2d=victory_lm))
    assert res == GestureName.VICTORY, f"Expected VICTORY, got {res}"

    # 5. Three Fingers
    three_lm = list(victory_lm)
    three_lm[14] = open_palm_lm[14]
    three_lm[15] = open_palm_lm[15]
    three_lm[16] = open_palm_lm[16]
    res = process_n_frames(HandLandmarks(landmarks_2d=three_lm))
    assert res == GestureName.THREE_FINGERS, f"Expected THREE_FINGERS, got {res}"

    # 6. Four Fingers
    four_lm = list(three_lm)
    four_lm[18] = open_palm_lm[18]
    four_lm[19] = open_palm_lm[19]
    four_lm[20] = open_palm_lm[20]
    res = process_n_frames(HandLandmarks(landmarks_2d=four_lm))
    assert res == GestureName.FOUR_FINGERS, f"Expected FOUR_FINGERS, got {res}"

    print("All systematic gesture recognition test cases passed successfully!")


if __name__ == "__main__":
    test_gesture_recognition()

