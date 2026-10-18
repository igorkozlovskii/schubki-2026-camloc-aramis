# Vendored third-party subset — LightGlue / ALIKED

Upstream: https://github.com/cvg/LightGlue @ `eb42fee2d71449efb0aa5c10549752b5d75384d8`
Primary licence: Apache-2.0 (`external/S04-lightglue/LightGlue/LICENSE`,
sha256 `0e6031ed3e62dc91dc8b1c8f240abd200b1083f60773362e64e339fe01d1bf18`).
`lightglue/aliked.py` carries a BSD-3-Clause header (ALIKED).

## Files shipped (necessary subset only)

| file | disposition | sha256 |
|---|---|---|
| `LICENSE` | verbatim | `0e6031ed3e62dc91dc8b1c8f240abd200b1083f60773362e64e339fe01d1bf18` |
| `lightglue/aliked.py` | verbatim | (see `manifests/source-export.json`) |
| `lightglue/lightglue.py` | verbatim | (see `manifests/source-export.json`) |
| `lightglue/utils.py` | verbatim | (see `manifests/source-export.json`) |
| `lightglue/__init__.py` | **adapted** | original `cf4322abb908f69282198931fb23c09a1ef1101b066d57842402e6436a92eaa3` → resulting (see `manifests/source-export.json`) |

## What is deliberately NOT shipped

- `lightglue/superpoint.py` — **restrictive licence** (Magic Leap, Inc.; its
  header forbids redistribution without written permission). Not needed: camloc
  uses only ALIKED + LightGlue + `match_pair`. Excluded entirely.
- `lightglue/disk.py`, `lightglue/dog_hardnet.py`, `lightglue/sift.py`,
  `lightglue/viz2d.py` — unused backends / visualization. Excluded.
- LightGlue model weights, `.git`, examples, notebooks, benchmarks and tests.

## Adaptation of `lightglue/__init__.py`

Upstream `__init__.py` eagerly imports every backend
(`from .superpoint import SuperPoint`, `.disk`, `.dog_hardnet`, `.sift`), which
would fail once their source is removed. The adapted file drops those import
lines and keeps exactly:

```python
from .aliked import ALIKED      # noqa
from .lightglue import LightGlue # noqa
from .utils import match_pair    # noqa
```

This is the only transformed source in the source export. Its original and
resulting sha256 are recorded in `manifests/source-export.json`
(`role=third_party`, `adaptation` field).

## Other declared external resources

The full external-resource / weight provenance and the complete component
licence table are in this directory's `external-resources.json` and
`license-table.md` (copied verbatim from the research checkout). Weight
acquisition and the organizer-data terms for own-derived maps/masks are tracked
there; weights are not part of the source foundation.
