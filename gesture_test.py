"""Standalone Gesture Test Harness for YouTube Gesture Controller.

Runs the video capture, MediaPipe hand landmarking, and systematic gesture
recognition pipeline in a dedicated OpenCV window. Displays finger states,
candidate gesture, confidence score, and stabilized gesture output.

IMPORTANT: Completely isolated from ActionController and PyAutoGUI.
Zero keyboard shortcuts or system actions will be dispatched.
"""

import sys
import cv2
import time

from camera import Camera
from config import AppConfig, GestureName
from hand_detector import HandDetector
from gesture_recognition import GestureRecognizer
from utils import FPSCalculator


def run_gesture_test() -> None:
    """Launches the standalone visual test harness for gesture recognition."""
    print("=" * 60)
    print("YOUTUBE GESTURE CONTROLLER - STANDALONE GESTURE TEST HARNESS")
    print("=" * 60)
    print("Press 'q' or ESC in the video window to exit.")
    print("Press 'd' to toggle verbose gesture debug logging in terminal.")
    print("=" * 60)

    config = AppConfig()
    camera = Camera(config)
    detector = HandDetector(config)
    recognizer = GestureRecognizer(config)
    fps_calc = FPSCalculator(buffer_size=30)

    camera.start()
    time.sleep(0.5)

    window_name = "Gesture Recognition Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Bring window to foreground and set TOPMOST property
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    import ctypes
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            ctypes.windll.user32.BringWindowToTop(hwnd)
        except Exception:
            pass

    debug_enabled = False
    first_frame = True

    try:
        while True:
            success, frame = camera.get_frame()
            if not success or frame is None:
                time.sleep(0.01)
                continue

            # Mirror frame horizontally for intuitive visual display
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # 1. Process hand landmarks
            annotated_frame, hand_landmarks = detector.process_frame(frame)

            # 2. Process gesture recognition
            stable_gesture = recognizer.process_landmarks(hand_landmarks, w, h)
            details = recognizer.get_last_analysis_details()

            fps = fps_calc.update()

            # 3. Render HUD Overlay on OpenCV Frame
            _draw_test_overlay(annotated_frame, stable_gesture, details, fps, debug_enabled)

            cv2.imshow(window_name, annotated_frame)

            if first_frame:
                first_frame = False
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                if hwnd:
                    try:
                        ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                        ctypes.windll.user32.SetForegroundWindow(hwnd)
                        ctypes.windll.user32.BringWindowToTop(hwnd)
                    except Exception:
                        pass

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                print("\nExiting gesture test harness...")
                break
            elif key in (ord('d'), ord('D')):
                debug_enabled = not debug_enabled
                config.gesture.DEBUG_GESTURES = debug_enabled
                print(f"[TEST HARNESS] Debug logging: {'ENABLED' if debug_enabled else 'DISABLED'}")

    except KeyboardInterrupt:
        print("\nTest harness interrupted by user.")
    finally:
        camera.stop()
        detector.close()
        cv2.destroyAllWindows()
        print("Test harness resources released cleanly.")


def _draw_test_overlay(
    frame, stable_gesture: GestureName, details: dict, fps: float, debug_enabled: bool
) -> None:
    """Draws diagnostic metrics on top of the OpenCV frame."""
    # Semi-transparent dark banner background for overlay text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # 1. Active Stable Gesture
    color = (0, 255, 0) if stable_gesture != GestureName.NONE else (180, 180, 180)
    cv2.putText(
        frame,
        f"STABLE: {stable_gesture.value}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    # 2. Candidate & Confidence Score
    cand = details.get("candidate_gesture", GestureName.NONE)
    conf = details.get("candidate_confidence", 0.0)
    cand_str = cand.value if isinstance(cand, GestureName) else str(cand)
    cv2.putText(
        frame,
        f"Candidate: {cand_str} ({conf * 100:.1f}%)",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    # 3. Finger States Breakdown
    states = details.get("finger_states", {})
    t = "T" if states.get("thumb") else "-"
    i = "I" if states.get("index") else "-"
    m = "M" if states.get("middle") else "-"
    r = "R" if states.get("ring") else "-"
    p = "P" if states.get("pinky") else "-"
    cv2.putText(
        frame,
        f"Fingers: [{t} {i} {m} {r} {p}]",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    # 4. Finger Scores
    scores = details.get("finger_scores", {})
    scores_str = f"Scores: I:{scores.get('index', 0):.2f} M:{scores.get('middle', 0):.2f} R:{scores.get('ring', 0):.2f} P:{scores.get('pinky', 0):.2f}"
    cv2.putText(
        frame,
        scores_str,
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    # 5. FPS & Debug Status
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Debug: {'ON' if debug_enabled else 'OFF'}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 255, 150),
        1,
    )


if __name__ == "__main__":
    run_gesture_test()
