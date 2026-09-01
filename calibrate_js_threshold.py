"""
calibrate_js_threshold.py
--------------------------
Ports the browser's computeTrafficFeaturesJS Sobel algorithm to Python EXACTLY
(same kernels, same normalization), runs it against the real heavy/light traffic
photos, and reports the actual value distribution so the in-browser threshold can
be calibrated against real data instead of guessed.
"""
import cv2
import numpy as np
import os
from glob import glob

BASE = os.path.dirname(__file__)
TRAFFIC_DIR = os.path.join(BASE, "real_dataset", "traffic")


def js_edge_density(img):
    """Exact port of the browser's Sobel-based computeTrafficFeaturesJS edgeDensity."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    # same 3x3 Sobel kernels as the JS version, applied to interior pixels only
    gx[1:-1, 1:-1] = (
        gray[0:-2, 0:-2] + 2*gray[0:-2, 1:-1] + gray[0:-2, 2:]
        - gray[2:, 0:-2] - 2*gray[2:, 1:-1] - gray[2:, 2:]
    )
    # NOTE: the JS code's gx/gy formulas mix indices in a specific way; replicate faithfully:
    # gx = tl + 2*l + bl - tr - 2*r - br   (this is actually closer to a vertical-edge kernel
    #      transposed from standard Sobel -- doesn't matter, we just need to match JS exactly)
    Gx = np.zeros_like(gray)
    Gy = np.zeros_like(gray)
    tl = gray[0:-2, 0:-2]; t = gray[0:-2, 1:-1]; tr = gray[0:-2, 2:]
    l  = gray[1:-1, 0:-2];                        r  = gray[1:-1, 2:]
    bl = gray[2:, 0:-2];   b = gray[2:, 1:-1];    br = gray[2:, 2:]
    Gx[1:-1, 1:-1] = tl + 2*l + bl - tr - 2*r - br
    Gy[1:-1, 1:-1] = tl + 2*t + tr - bl - 2*b - br
    mag = np.sqrt(Gx**2 + Gy**2)
    edge_sum = mag.sum()
    edge_density = min(1.0, edge_sum / (w*h) / 180.0)
    return edge_density


def collect(label):
    vals = []
    for path in glob(os.path.join(TRAFFIC_DIR, label, "*")):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (200, 150))  # match the browser thumbnail size roughly
        vals.append(js_edge_density(img))
    return vals


if __name__ == "__main__":
    heavy = collect("heavy")
    light = collect("light")
    print(f"HEAVY (n={len(heavy)}): min={min(heavy):.4f} max={max(heavy):.4f} "
          f"mean={np.mean(heavy):.4f} median={np.median(heavy):.4f}")
    print(f"LIGHT (n={len(light)}): min={min(light):.4f} max={max(light):.4f} "
          f"mean={np.mean(light):.4f} median={np.median(light):.4f}")

    # sweep thresholds to find the one that maximizes classification accuracy
    all_vals = sorted(set(heavy + light))
    best_thr, best_acc = None, -1
    for thr in all_vals:
        correct = sum(1 for v in heavy if v > thr) + sum(1 for v in light if v <= thr)
        acc = correct / (len(heavy) + len(light))
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    print(f"\nBest single threshold: {best_thr:.4f}  (accuracy {best_acc:.3f})")
