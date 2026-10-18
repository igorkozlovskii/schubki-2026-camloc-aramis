# Map building → Inference

[scheme.drawio](scheme.drawio) has one page: map building at the top, inference below. Dashed arrows connect stored training data to the stages that use it.

## Map building

Training frames → keyframes → two branches:

- DINOv2 → **training embeddings** for image retrieval.
- ALIKED → LightGlue between training frames → triangulation using known poses → **local features linked to 3D landmarks**.

## Inference

Query frame → two branches:

- DINOv2 → query embedding + training embeddings → keyframe retrieval.
- ALIKED → query features + selected keyframe features → LightGlue → 2D matches → 3D landmarks → PnP → visual pose.

Telemetry → motion prediction. Visual pose + border and pole corrections + prediction → EKF → final `x, y, yaw`.

The initial pose initializes the EKF and landmark prior; each camera has its own map. Borders and poles use separate maps and the current filter estimate. Calibration is used throughout the geometry stages. These connections are omitted from the diagram for readability.
