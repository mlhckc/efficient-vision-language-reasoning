# Tracked evidence export

This directory holds byte-identical copies of the small result artifacts
(JSON metrics, CSV tables and PNG figures) from the git-ignored results/
tree, one subdirectory per experiment plus v1/ for the legacy stage
results, so that the numbers behind the dissertation's claims survive the
node-local scratch disk.

MANIFEST.json pins every copied file and every large non-exported
artifact (the 116 model checkpoints, the HDF5 embedding and token stores,
the two v2_00 preservation inventories and the raw GQA question files) by
size and SHA-256, together with the git commit at export time. Entries
with "exported": false remain node-local and are identified by hash only.

The export is produced by experiments/tools/export_evidence.py, which is
deterministic and idempotent: rerunning it with unchanged sources copies
nothing and leaves MANIFEST.json byte-identical. Regenerate after any new
experiment by running:

    python -B experiments/tools/export_evidence.py

All exported numbers are V2/V3 development-set results (or V1 legacy
prototype results under v1/); no clean-test evaluation has occurred and
nothing here is derived from clean-test labels.
