"""Calibrates FIRE_AREA_THRESHOLD the same way as calibrate_js_threshold.py, but for the
JS fire-color HSV pixel-ratio algorithm. Exact port of computeFireFeaturesJS."""
import cv2
import numpy as np
import os
from glob import glob

BASE = os.path.dirname(__file__)
FIRE_DIR = os.path.join(BASE, "real_dataset", "fire")


def js_fire_area_ratio(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    delta = mx - mn
    with np.errstate(divide='ignore', invalid='ignore'):
        hue_r = 60 * (((g - b) / np.where(delta == 0, 1, delta)) % 6)
        hue_g = 60 * (((b - r) / np.where(delta == 0, 1, delta)) + 2)
        hue_b = 60 * (((r - g) / np.where(delta == 0, 1, delta)) + 4)
    hue = np.where(mx == r, hue_r, np.where(mx == g, hue_g, hue_b))
    hue = np.where(delta > 0, hue, 0)
    hue = np.where(hue < 0, hue + 360, hue)
    sat = np.where(mx == 0, 0, delta / np.where(mx == 0, 1, mx))
    val = mx / 255.0
    fire_mask = ((hue <= 45) | (hue >= 340)) & (sat > 0.45) & (val > 0.5)
    return fire_mask.sum() / (img.shape[0] * img.shape[1])


def collect(label):
    vals = []
    for path in glob(os.path.join(FIRE_DIR, label, "*")):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (200, 150))
        vals.append(js_fire_area_ratio(img))
    return vals


if __name__ == "__main__":
    fire = collect("fire")
    no_fire = collect("no_fire")
    print(f"FIRE    (n={len(fire)}): min={min(fire):.4f} max={max(fire):.4f} "
          f"mean={np.mean(fire):.4f} median={np.median(fire):.4f}")
    print(f"NO_FIRE (n={len(no_fire)}): min={min(no_fire):.4f} max={max(no_fire):.4f} "
          f"mean={np.mean(no_fire):.4f} median={np.median(no_fire):.4f}")

    all_vals = sorted(set(fire + no_fire))
    best_thr, best_acc = None, -1
    for thr in all_vals:
        correct = sum(1 for v in fire if v > thr) + sum(1 for v in no_fire if v <= thr)
        acc = correct / (len(fire) + len(no_fire))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"\nBest single threshold: {best_thr:.4f}  (accuracy {best_acc:.3f})")
