"""Main entry point for the YouTube Gesture Controller.

Bootstraps the configuration, configures centralized logging, instantiates the
hardware, mathematical processing layers, and launches the CustomTkinter GUI.
"""

import sys

from camera import Camera
from config import AppConfig
from controller import ActionController
from gesture_recognition import GestureRecognizer
from gui import GestureControllerApp
from hand_detector import HandDetector
from logger import get_logger, setup_logging


def main() -> None:
    """Bootstraps the components and starts the desktop application loop."""
    # 1. Initialize configuration and setup logging
    config = AppConfig()
    setup_logging(debug=config.DEBUG)

    logger = get_logger("main")
    logger.info("Initializing YouTube Gesture Controller system...")

    try:
        # 2. Instantiate components
        camera = Camera(config)
        detector = HandDetector(config)
        recognizer = GestureRecognizer(config)
        controller = ActionController(config)

        logger.info("Component initialization complete. Launching GUI App...")

        # 3. Create GUI Application (which starts camera threads internally)
        app = GestureControllerApp(
            config=config,
            camera=camera,
            detector=detector,
            recognizer=recognizer,
            controller=controller,
        )

        # 4. Start main graphical event loop
        app.mainloop()

    except KeyboardInterrupt:
        logger.info("Application execution interrupted by user (KeyboardInterrupt). Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception during startup or main lifecycle: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
