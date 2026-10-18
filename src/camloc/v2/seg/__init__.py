"""S1.4: paved-surface segmentation (labels from GT-projected map borders, small model, training, inference).

Train/infer modules (`config`, `dataset`, `augment`, `model`, `train`, `infer`) depend only on numpy/torch/cv2/yaml so a
run directory can be copied to a server; the label builder (`labels`, `calib_hook`) needs `camloc.geometry` and GT and
is OFFLINE ONLY (never imported by inference).
"""
CLASS_OTHER = 0
CLASS_ROAD = 1
CLASS_IGNORE = 255
CAMERAS = ("fl", "fr", "cl", "cr", "rl", "rr", "r")
TRAIN_LAPS = (0, 1, 3, 5)
LABEL_W, LABEL_H = 384, 192
