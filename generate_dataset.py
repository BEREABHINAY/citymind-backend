"""
generate_dataset.py
--------------------
Stands in for real camera datasets, which you don't have access to (no hardware,
no downloadable dataset in this environment).

WHAT TO DO FOR YOUR ACTUAL SUBMISSION:
  Traffic classification -> download one of:
    - "Traffic Vehicles Object Detection" (Kaggle)
    - UA-DETRAC (public traffic surveillance dataset, has vehicle density labels)
    - PANDA / NGSIM aerial traffic datasets
  Fire classification -> download one of:
    - "Fire Detection Dataset" (Kaggle, ~1000 fire/non-fire images)
    - FireNet dataset (fire / smoke / normal, video frames)
    - FLAME dataset (aerial wildfire imagery, if you want a stronger fire angle)

  Then simply point TRAFFIC_DIR / FIRE_DIR in classifier.py at the real folders
  (organised as class-subfolders: light/medium/heavy/none for traffic,
  fire/no_fire for fire) and delete this synthetic generator. Everything else
  (feature extraction + ML classifier + evaluation) works unchanged on real data.

This script procedurally draws:
  - "traffic" scenes: an aerial road with 0 / few / many car-blobs on it
    (none / light / medium / heavy)
  - "fire" scenes: either a clean scene, a tiny cigarette-lighter-scale flame
    (must be classified NO), or a large spreading fire (must be classified YES)
"""
import cv2
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

W, H = 224, 224
OUT_DIR = os.path.join(os.path.dirname(__file__), "dataset")

TRAFFIC_LEVELS = {"none": (0, 1), "light": (2, 5), "medium": (6, 12), "heavy": (13, 22)}
FIRE_CLASSES = ["no_fire", "no_fire_small_flame", "fire"]  # small_flame = cigarette-lighter case -> labeled no_fire


def draw_road_scene(n_cars):
    img = np.full((H, W, 3), (60, 60, 60), dtype=np.uint8)  # asphalt
    cv2.rectangle(img, (0, 0), (W, H), (70, 90, 70), -1)     # roadside greenery base
    cv2.rectangle(img, (40, 0), (W - 40, H), (55, 55, 55), -1)  # road strip
    for lane in range(40, W - 40, 40):
        cv2.line(img, (lane, 0), (lane, H), (200, 200, 0), 1)
    car_colors = [(0, 0, 200), (200, 200, 200), (0, 100, 0), (150, 60, 0), (20, 20, 20)]
    for _ in range(n_cars):
        x = random.randint(45, W - 65)
        y = random.randint(5, H - 25)
        w, h = random.randint(14, 20), random.randint(9, 13)
        color = random.choice(car_colors)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 1)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def draw_fire_scene(kind):
    img = np.full((H, W, 3), (40, 45, 50), dtype=np.uint8)  # dim background (BGR)
    cv2.rectangle(img, (0, H - 40), (W, H), (30, 60, 90), -1)  # ground

    if kind == "no_fire":
        # maybe a red object that is NOT fire-colored-flickery (e.g. a red car / sign)
        if random.random() < 0.5:
            cv2.rectangle(img, (80, 80), (140, 120), (0, 0, 180), -1)
        return img

    if kind == "no_fire_small_flame":
        # cigarette-lighter scale flame: tiny, localized, low area
        cx, cy = random.randint(90, 130), random.randint(90, 130)
        r = random.randint(3, 6)
        cv2.circle(img, (cx, cy), r, (0, 140, 255), -1)   # orange
        cv2.circle(img, (cx, cy - 1), max(1, r - 2), (0, 220, 255), -1)  # yellow core
        return img

    if kind == "fire":
        # large spreading fire: several big blobs + smoke haze
        n_blobs = random.randint(4, 8)
        for _ in range(n_blobs):
            cx = random.randint(30, W - 30)
            cy = random.randint(H // 2, H - 20)
            r = random.randint(18, 38)
            cv2.circle(img, (cx, cy), r, (0, random.randint(90, 160), 255), -1)
            cv2.circle(img, (cx, cy - r // 3), int(r * 0.6), (0, 200, 255), -1)
        # smoke
        overlay = img.copy()
        for _ in range(6):
            cx = random.randint(20, W - 20)
            cy = random.randint(10, H // 2)
            r = random.randint(20, 45)
            cv2.circle(overlay, (cx, cy), r, (90, 90, 90), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        return img


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    traffic_dir = os.path.join(OUT_DIR, "traffic")
    fire_dir = os.path.join(OUT_DIR, "fire")
    for level in TRAFFIC_LEVELS:
        os.makedirs(os.path.join(traffic_dir, level), exist_ok=True)
    for cls in FIRE_CLASSES:
        os.makedirs(os.path.join(fire_dir, cls), exist_ok=True)

    n_per_class = 40
    for level, (lo, hi) in TRAFFIC_LEVELS.items():
        for i in range(n_per_class):
            n_cars = random.randint(lo, hi)
            img = draw_road_scene(n_cars)
            cv2.imwrite(os.path.join(traffic_dir, level, f"{level}_{i:03d}.png"), img)

    for cls in FIRE_CLASSES:
        for i in range(n_per_class):
            img = draw_fire_scene(cls)
            cv2.imwrite(os.path.join(fire_dir, cls, f"{cls}_{i:03d}.png"), img)

    print(f"Synthetic dataset written to {OUT_DIR}")
    print(f"  traffic: {len(TRAFFIC_LEVELS)} classes x {n_per_class} images")
    print(f"  fire:    {len(FIRE_CLASSES)} classes x {n_per_class} images")


if __name__ == "__main__":
    build()
