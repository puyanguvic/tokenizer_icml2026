# Theory Notes

This file is reserved for derivations and appendix material that should stay in sync
with the paper (e.g., gain--distortion optimization, distortion theory, and diagnostics).

## Distortion proxies

The codebase includes a lightweight label-entropy proxy used during induction when a
full probe is not available. This proxy penalizes tokens that mix labels by computing
entropy over token occurrences across labels.
