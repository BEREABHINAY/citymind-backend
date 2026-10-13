"""Compare the original 3-feature RandomForest against richer DIP features + gradient boosting,
on the identical stratified 75/25 split of the real 1000-per-class dataset."""
import os, sys, time
import numpy as np
import cv2
from glob import glob
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

BASE = os.path.dirname(os.path.abspath(__file__))
TRAFFIC_DIR = os.path.join(BASE, "real_dataset", "traffic")
FIRE_DIR = os.path.join(BASE, "real_dataset", "fire")
HOG = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)


def traffic_rich(img):
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    h = edges.shape[0]
    bands = [edges[i * h // 3:(i + 1) * h // 3].mean() / 255.0 for i in range(3)]
    quads = [edges[:112, :112].mean(), edges[:112, 112:].mean(), edges[112:, :112].mean(), edges[112:, 112:].mean()]
    quads = [q / 255.0 for q in quads]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = len([c for c in cnts if 60 < cv2.contourArea(c) < 600])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    sh = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten()
    vh = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten()
    hist = np.concatenate([hh, sh, vh]); hist = hist / hist.sum()
    lap = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
    hog = HOG.compute(cv2.resize(gray, (128, 128))).flatten()
    base = [edges.mean() / 255.0, blobs, img.reshape(-1, 3).std(axis=0).mean() / 255.0]
    return np.concatenate([base, bands, quads, hist, [lap, gray.mean() / 255.0, gray.std() / 255.0], hog])


def fire_rich(img):
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    total = 224 * 224
    feats = []
    for smin, vmin in [(120, 140), (80, 120), (150, 200)]:
        m = cv2.inRange(hsv, (0, smin, vmin), (35, 255, 255)) | cv2.inRange(hsv, (160, smin, vmin), (180, 255, 255))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = max([cv2.contourArea(c) for c in cnts], default=0) / total
        feats += [m.sum() / 255.0 / total, big, len([c for c in cnts if cv2.contourArea(c) > 4])]
        if smin == 120:
            feats += [m[:112].mean() / 255.0, m[112:].mean() / 255.0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    smoke = cv2.inRange(hsv, (0, 0, 60), (180, 40, 200)).mean() / 255.0
    bright = (gray > 230).mean()
    b, g, r = [img[..., i].astype(float) for i in range(3)]
    red_dom = ((r > g + 30) & (r > b + 60)).mean()
    hh = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    sh = cv2.calcHist([hsv], [1], None, [6], [0, 256]).flatten()
    vh = cv2.calcHist([hsv], [2], None, [6], [0, 256]).flatten()
    hist = np.concatenate([hh, sh, vh]); hist = hist / hist.sum()
    return np.concatenate([feats, [smoke, bright, red_dom, gray.mean() / 255.0, gray.std() / 255.0], hist])


def load(root, classes):
    items = []
    for cls, label in classes.items():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            items += [(p, label) for p in glob(os.path.join(root, cls, ext))]
    return items


def run(name, root, classes, fn):
    t0 = time.time()
    items = load(root, classes)
    X = Parallel(n_jobs=-1)(delayed(lambda p: fn(cv2.imread(p)))(p) for p, _ in items)
    X = np.array(X); y = np.array([l for _, l in items])
    np.save(os.path.join(BASE, f"_feat_{name}_X.npy"), X); np.save(os.path.join(BASE, f"_feat_{name}_y.npy"), y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"[{name}] {len(items)} images, {X.shape[1]} features, extracted in {time.time()-t0:.0f}s")
    rf0 = RandomForestClassifier(n_estimators=150, random_state=42).fit(Xtr[:, :3] if name == "traffic" else Xtr[:, :3], ytr)
    print(f"   baseline RF on first 3 feats : {accuracy_score(yte, rf0.predict(Xte[:, :3])):.3f}")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    print(f"   RF on rich features          : {accuracy_score(yte, rf.predict(Xte)):.3f}")
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=42).fit(Xtr, ytr)
    print(f"   HistGradientBoosting rich    : {accuracy_score(yte, hgb.predict(Xte)):.3f}")


if __name__ == "__main__":
    run("traffic", TRAFFIC_DIR, {"heavy": "heavy", "light": "light"}, traffic_rich)
    run("fire", FIRE_DIR, {"fire": "yes", "no_fire": "no"}, fire_rich)
