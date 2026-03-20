# Standard library imports
import os
import sys

# ensure cvnn package is discoverable
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

# Local imports
from cvnn.cli import cli_main as main

if __name__ == "__main__":
    main()  # utilise configs/segmentation.yaml
