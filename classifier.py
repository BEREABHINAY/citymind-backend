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

ML layer: RandomForestClassifier (scikit-learn) trained on the extracted
features. Swap TRAFFIC_DIR / FIRE_DIR to real dataset folders for your actual
submission (see generate_dataset.py header for real dataset names).
"""
import cv2
import numpy as np
import os
import random
from glob import glob
from sklearn.ensemble import RandomForestClassifier
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

def extract_traffic_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edge_density = edges.mean() / 255.0

    # isolate the road strip and find vehicle-like blobs via adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 5)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vehicle_like = [c for c in contours if 60 < cv2.contourArea(c) < 600]
    blob_count = len(vehicle_like)

    color_std = img.reshape(-1, 3).std(axis=0).mean() / 255.0

    return [edge_density, blob_count, color_std]


TRAFFIC_FEATURE_NAMES = ["edge_density", "vehicle_blob_count", "color_std"]


def extract_fire_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # fire color range: orange/yellow/red with high saturation and brightness
    lower1 = np.array([0, 120, 140]);  upper1 = np.array([35, 255, 255])
    lower2 = np.array([160, 120, 140]); upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    total_px = mask.shape[0] * mask.shape[1]
    fire_area_ratio = mask.sum() / 255.0 / total_px

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_area_ratio = max(cv2.contourArea(c) for c in contours) / total_px
        region_count = len([c for c in contours if cv2.contourArea(c) > 4])
    else:
        largest_area_ratio = 0.0
        region_count = 0

    return [fire_area_ratio, largest_area_ratio, region_count]


FIRE_FEATURE_NAMES = ["fire_pixel_ratio", "largest_region_ratio", "fire_region_count"]

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
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
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
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
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
            "confidence": round(max(proba.values()), 3), "features": dict(zip(TRAFFIC_FEATURE_NAMES, feats))}


def classify_fire_image(clf, img):
    feats = extract_fire_features(img)
    verdict = clf.predict([feats])[0]
    proba = dict(zip(clf.classes_, clf.predict_proba([feats])[0]))
    return {"fire": "Yes" if verdict == "yes" else "No",
            "confidence": round(max(proba.values()), 3),
            "features": dict(zip(FIRE_FEATURE_NAMES, feats))}


# --------------------------------------------------------------------- CLI --
def _all_images(root):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files += glob(os.path.join(root, "*", ext))
    return files


if __name__ == "__main__":
    print(f"Data source: {'REAL images (Traffic-Net + cair fire dataset)' if USING_REAL_DATA else 'synthetic (fallback)'}")
    print("=" * 70)
    print("TRAINING TRAFFIC CLASSIFIER (DIP features -> RandomForest)")
    print("=" * 70)
    traffic_clf, t_acc, t_report = train_traffic_model()
    print(f"Test accuracy: {t_acc:.3f}\n{t_report}")

    print("=" * 70)
    print("TRAINING FIRE CLASSIFIER (DIP features -> RandomForest)")
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
