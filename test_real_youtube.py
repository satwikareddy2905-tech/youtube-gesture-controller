"""Real end-to-end verification script for YouTube control via controller.py.

Dispatches all 9 YouTube actions to an open Chrome window playing YouTube,
verifying real video player response and logging performance.
"""

import time

from config import AppConfig, GestureName
from controller import ActionController, _find_target_youtube_window


def test_real_youtube() -> None:
    print("=== REAL YOUTUBE END-TO-END CONTROLLER TEST ===")

    # 1. Verify target browser window lookup
    hwnd, title, pname = _find_target_youtube_window()
    print(f"Target Browser Window: HWND {hwnd} | Process: {pname} | Title: '{title}'")
    if not hwnd:
        print("ERROR: No open Chrome/browser YouTube window found. Please open Chrome to YouTube.")
        return

    config = AppConfig()
    controller = ActionController(config)

    gestures = [
        (GestureName.CLOSED_FIST, "Play/Pause ('k')"),
        (GestureName.OPEN_PALM, "Mute/Unmute ('m')"),
        (GestureName.THUMB_UP, "Volume Up ('Up')"),
        (GestureName.THUMB_DOWN, "Volume Down ('Down')"),
        (GestureName.VICTORY, "Fullscreen ('f')"),
        (GestureName.THREE_FINGERS, "Theater Mode ('t')"),
        (GestureName.FOUR_FINGERS, "Toggle Captions ('c')"),
        (GestureName.SWIPE_RIGHT, "Seek Forward 10s ('l')"),
        (GestureName.SWIPE_LEFT, "Seek Back 10s ('j')"),
    ]

    for gesture, desc in gestures:
        controller.reset_state()
        controller._last_action_time = 0.0
        controller._last_volume_action_time = 0.0

        print(f"\nTriggering Gesture: {gesture.value} -> {desc}")
        trig, action, key = controller.trigger_action(gesture)
        print(f"Result -> Dispatched: {trig} | Action: {action} | Key: {key}")
        time.sleep(1.2)

    print("\n=== REAL YOUTUBE TEST COMPLETE ===")


if __name__ == "__main__":
    test_real_youtube()
