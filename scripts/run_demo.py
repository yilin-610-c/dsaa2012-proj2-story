from __future__ import annotations

import sys

from storygen.cli import main


if __name__ == "__main__":
    if "--profile" not in sys.argv:
        sys.argv.extend(["--profile", "demo_run"])
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", "test_set/01.txt"])
    main()
