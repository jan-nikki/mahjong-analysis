"""Backward-compatible entry point for the 2025 reference validation."""

import sys

if __package__:
    from analysis.validate_dealer_double_riichi import (
        _analyze_with_existing_implementation as _analyze_with_existing_implementation,
    )
    from analysis.validate_dealer_double_riichi import main
else:
    from validate_dealer_double_riichi import (
        _analyze_with_existing_implementation as _analyze_with_existing_implementation,
    )
    from validate_dealer_double_riichi import main


if __name__ == "__main__":
    raise SystemExit(main(["--year", "2025", *sys.argv[1:]]))
