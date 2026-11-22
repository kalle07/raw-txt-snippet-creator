"""Application entry point."""

import multiprocessing

from gui import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
