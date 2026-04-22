"""Allow `python -m extensions.dashboard`."""
from .server import main
import sys

if __name__ == "__main__":
    sys.exit(main())
