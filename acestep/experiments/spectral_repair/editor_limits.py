"""Shared excerpt and mask transport bounds for the local spectral laboratory."""

MIN_DURATION = 4
MAX_DURATION = 200
# 200 s / 48 kHz / FFT 2048 / hop 512 needs 19,219,775 cells.
MAX_MASK_CELLS = 24_000_000
MAX_PAYLOAD_CHARS = MAX_MASK_CELLS * 16 // 3 + 4096
