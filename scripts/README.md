# Rare-yakuman article asset helpers

This directory contains the five one-off helpers that produced the retained
rare-yakuman article assets. They are preserved for reproducibility, but they
are not part of the statistical-analysis pipeline under `analysis/`.

## Green all

1. `build_ryuuiisou_capture_targets.py` reads the local Tenhou archive
   databases and writes the 135 retained targets.
2. `capture_ryuuiisou_all135.py` captures the table frame immediately before
   the result popup for all 135 targets.
3. `build_ryuuiisou_note_list.py` builds the corresponding article list.

## Four kans

1. `build_suukantsu_capture_targets.py` converts the retained yakuman scan into
   the three capture targets.
2. `capture_suukantsu_all3.py` captures the table frame immediately before the
   result popup for those three targets.

The scripts expect ignored local inputs under `data/archives/` and `outputs/`.
The capture steps additionally require Pillow, Playwright, and a Playwright
Chromium installation. Generated images and logs remain under `outputs/` and
are intentionally not committed.
