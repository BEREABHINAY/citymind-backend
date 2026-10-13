"""
classifier.py
-------------
Digital Image Processing (DIP) feature extraction + ML classification for:
  1. Traffic:  Yes/No  ->  if Yes: Light / Medium / Heavy
  2. Fire:     Yes/No   (deliberately rejects cigarette-lighter-scale flames)

DIP techniques used (classical, explainable — good for a project report):
  - Grayscale conversion + Canny edge detection        -> edge density
  - Contour detection on adaptive-thresholded regions   -> "vehicle blob" count
  - Color histogram spread (std-dev across channels)    -> visual clutter proxy
  - HSV color-space thresholding for fire-colored pixels (orange/yellow/red,
    high saturation & value)                            -> fire mask
  - Connected-component analysis on the fire mask       -> largest fire-region
    area ratio + region count (this is what separates a lighter flame from a
    real fire: AREA and SPREAD, not just "is there orange in the frame")

ML layer: HistGradientBoostingClassifier (scikit-learn) trained on the extracted
features. Swap TRAFFIC_DIR / FIRE_DIR to real dataset folders for your actual
submission (see generate_dataset.py header for real dataset names).
"""
import cv2
import numpy as np
import os
import random
from glob import glob
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(__file__)

# REAL, PUBLICLY-SOURCED IMAGES (not synthetic):
#   traffic/heavy, traffic/light  <- OlafenwaMoses/Traffic-Net (dense_traffic / sparse_traffic classes)
#   fire/fire                     <- OlafenwaMoses/Traffic-Net "fire" class + cair/Fire-Detection-Image-Dataset
#   fire/no_fire                  <- OlafenwaMoses/Traffic-Net "accident" + extra "sparse_traffic" scenes
#     (accident scenes are a deliberately hard negative: red brake lights / red vehicles look
#      fire-colored to a naive classifier, which is exactly what should NOT trigger "fire: yes")
# Falls back to the synthetic set only if the real folder isn't present.
REAL_DIR = os.path.join(BASE, "real_dataset")
SYNTH_DIR = os.path.join(BASE, "dataset")
TRAFFIC_DIR = os.path.join(REAL_DIR, "traffic") if os.path.isdir(REAL_DIR) else os.path.join(SYNTH_DIR, "traffic")
FIRE_DIR = os.path.join(REAL_DIR, "fire") if os.path.isdir(REAL_DIR) else os.path.join(SYNTH_DIR, "fire")
USING_REAL_DATA = os.path.isdir(REAL_DIR)

# ---------------------------------------------------------------- FEATURES --

_HOG = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)


def _hsv_hist(hsv):
    hh = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    sh = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten()
    vh = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten()
    hist = np.concatenate([hh, sh, vh])
    return hist / max(hist.sum(), 1)


def extract_traffic_features(img):
    """Rich DIP feature vector (1801 dims). The first 3 entries are the original interpretable
    features (edge density, vehicle-blob count, colour clutter); the rest add edge density by
    image band/quadrant, HSV colour histograms, sharpness, brightness stats and HOG texture."""
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    h = edges.shape[0]
    bands = [edges[i * h // 3:(i + 1) * h // 3].mean() / 255.0 for i in range(3)]
    quads = [edges[:112, :112].mean(), edges[:112, 112:].mean(),
             edges[112:, :112].mean(), edges[112:, 112:].mean()]
    quads = [q / 255.0 for q in quads]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_count = len([c for c in contours if 60 < cv2.contourArea(c) < 600])
    hist = _hsv_hist(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
    hog = _HOG.compute(cv2.resize(gray, (128, 128))).flatten()
    base = [edges.mean() / 255.0, blob_count, img.reshape(-1, 3).std(axis=0).mean() / 255.0]
    return np.concatenate([base, bands, quads, hist,
                           [sharp, gray.mean() / 255.0, gray.std() / 255.0], hog])


TRAFFIC_FEATURE_NAMES = ["edge_density", "vehicle_blob_count", "color_std"]  # first 3 of the vector


def extract_fire_features(img):
    """Rich fire feature vector (40 dims). First 3 entries are the original interpretable
    features (fire-pixel ratio, largest fire region, region count). The rest add looser/stricter
    HSV fire-colour masks, upper/lower-half fire spread, smoke-grey ratio, blown-out highlights,
    red dominance and HSV histograms -- so a lighter-sized flame and a real fire separate cleanly."""
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = 224 * 224
    feats = []
    for smin, vmin in [(120, 140), (80, 120), (150, 200)]:
        m = cv2.inRange(hsv, (0, smin, vmin), (35, 255, 255)) | \
            cv2.inRange(hsv, (160, smin, vmin), (180, 255, 255))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max([cv2.contourArea(c) for c in cnts], default=0) / total
        feats += [m.sum() / 255.0 / total, biggest, len([c for c in cnts if cv2.contourArea(c) > 4])]
        if smin == 120:
            feats += [m[:112].mean() / 255.0, m[112:].mean() / 255.0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoke = cv2.inRange(hsv, (0, 0, 60), (180, 40, 200)).mean() / 255.0
    bright = (gray > 230).mean()
    b, g, r = [img[..., i].astype(float) for i in range(3)]
    red_dom = ((r > g + 30) & (r > b + 60)).mean()
    return np.concatenate([feats, [smoke, bright, red_dom, gray.mean() / 255.0, gray.std() / 255.0],
                           _hsv_hist(hsv)])


FIRE_FEATURE_NAMES = ["fire_pixel_ratio", "largest_region_ratio", "fire_region_count"]  # first 3 of the vector

# -------------------------------------------------------------- DATASET IO --

def load_dataset(root, label_map):
    X, y, paths = [], [], []
    for cls, label in label_map.items():
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
            files += glob(os.path.join(root, cls, ext))
        for path in files:
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, (224, 224))  # normalize size across real photos of varying resolution
            X.append(img)
            y.append(label)
            paths.append(path)
    return X, y, paths


# ------------------------------------------------------------------ TRAIN --

def train_traffic_model():
    # real dataset ships 2 classes (light/heavy); synthetic ships 4 (none/light/medium/heavy)
    available = set(os.listdir(TRAFFIC_DIR)) if os.path.isdir(TRAFFIC_DIR) else set()
    full_map = {"none": "none", "light": "light", "medium": "medium", "heavy": "heavy"}
    label_map = {k: v for k, v in full_map.items() if k in available}
    imgs, labels, _ = load_dataset(TRAFFIC_DIR, label_map)
    X = [extract_traffic_features(im) for im in imgs]
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.25,
                                           random_state=42, stratify=labels)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    report = classification_report(yte, pred, zero_division=0)
    return clf, acc, report


def train_fire_model():
    # fire "Yes" ONLY for the large-spreading-fire class; the tiny lighter-scale
    # flame is explicitly labeled "no" -- this encodes exactly your requirement
    # that a cigarette lighter must NOT be classified as a fire incident.
    available = set(os.listdir(FIRE_DIR)) if os.path.isdir(FIRE_DIR) else set()
    full_map = {"no_fire": "no", "no_fire_small_flame": "no", "fire": "yes"}
    label_map = {k: v for k, v in full_map.items() if k in available}
    imgs, labels, _ = load_dataset(FIRE_DIR, label_map)
    X = [extract_fire_features(im) for im in imgs]
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.25,
                                           random_state=42, stratify=labels)
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    report = classification_report(yte, pred, zero_division=0)
    return clf, acc, report


# -------------------------------------------------------------- INFERENCE --

def classify_traffic_image(clf, img):
    feats = extract_traffic_features(img)
    level = clf.predict([feats])[0]
    proba = dict(zip(clf.classes_, clf.predict_proba([feats])[0]))
    is_traffic = "No" if level == "none" else "Yes"
    return {"traffic": is_traffic, "level": level if is_traffic == "Yes" else "-",
            "confidence": round(max(proba.values()), 3), "features": {k: float(v) for k, v in zip(TRAFFIC_FEATURE_NAMES, feats)}}


def classify_fire_image(clf, img):
    feats = extract_fire_features(img)
    verdict = clf.predict([feats])[0]
    proba = dict(zip(clf.classes_, clf.predict_proba([feats])[0]))
    return {"fire": "Yes" if verdict == "yes" else "No",
            "confidence": round(max(proba.values()), 3),
            "features": {k: float(v) for k, v in zip(FIRE_FEATURE_NAMES, feats)}}


# --------------------------------------------------------------------- CLI --
def _all_images(root):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files += glob(os.path.join(root, "*", ext))
    return files


if __name__ == "__main__":
    print(f"Data source: {'REAL images (Traffic-Net + cair fire dataset)' if USING_REAL_DATA else 'synthetic (fallback)'}")
    print("=" * 70)
    print("TRAINING TRAFFIC CLASSIFIER (rich DIP features -> gradient boosting)")
    print("=" * 70)
    traffic_clf, t_acc, t_report = train_traffic_model()
    print(f"Test accuracy: {t_acc:.3f}\n{t_report}")

    print("=" * 70)
    print("TRAINING FIRE CLASSIFIER (rich DIP features -> gradient boosting)")
    print("=" * 70)
    fire_clf, f_acc, f_report = train_fire_model()
    print(f"Test accuracy: {f_acc:.3f}\n{f_report}")

    # ---- demo: 4 random images from the dataset, as requested ----
    print("=" * 70)
    print("DEMO: classifying 4 random TRAFFIC images from the dataset")
    print("=" * 70)
    all_traffic = _all_images(TRAFFIC_DIR)
    random.seed()
    for path in random.sample(all_traffic, 4):
        img = cv2.imread(path)
        result = classify_traffic_image(traffic_clf, img)
        true_label = os.path.basename(os.path.dirname(path))
        print(f"  {os.path.basename(path):22s} true={true_label:7s} -> "
              f"Traffic: {result['traffic']:3s}  Level: {result['level']:6s}  "
              f"(confidence {result['confidence']})")

    print("=" * 70)
    print("DEMO: classifying 4 random FIRE images from the dataset")
    print("=" * 70)
    all_fire = _all_images(FIRE_DIR)
    for path in random.sample(all_fire, 4):
        img = cv2.imread(path)
        result = classify_fire_image(fire_clf, img)
        true_label = os.path.basename(os.path.dirname(path))
        print(f"  {os.path.basename(path):28s} true={true_label:20s} -> "
              f"Fire: {result['fire']:3s}  (confidence {result['confidence']})")
