# Release layout and export process

How this delivery was assembled from the research checkout, kept for provenance. The run instructions are in
`README.md`; this page does not describe anything the run needs.

## Layout

See the "Repository layout" section of `README.md`. The delivery is one model: mc7 + borders + poles with
speed-aware (dynamic) v_max and no learned layer, frozen as `configs/final/final-v1.yaml`. Older packaging
options are not shipped.

## Export process

1. Source files are copied by content (not by parent commit) with `tools/export_sources.py`, which takes an
   explicit source root and destination, records every file's sha256 and any adaptation in
   `manifests/source-export.json`, and refuses to overwrite a differing file.
2. The runtime closure is derived from the frozen configuration's code list and the import closure of the
   driver scripts, with an explicit exclusion list for research-only paths (learned layer, S3.1 training).
3. The asset bundle is built by `tools/build_assets.py`, which verifies the source and the copy against the
   research freeze manifest, and checked by `tools/verify_assets.py`. It is a release asset, not in Git.
4. The code, configuration and assets are coupled by hash; an upgrade is a new versioned snapshot with its own
   bundle and verification, never a silent change to an earlier one.

## Boundary

No raw organizer recordings, ground truth, credentials or machine-local paths are part of the delivery. The
research checkout is not needed at run time.
