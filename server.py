"""
server.py
---------
Real backend for CityMind's AI/ML Systems panel. Loads/trains the ACTUAL
models from classifier.py (RandomForest on real DIP features, real dataset)
and dqn_routing.py (DQN route planner), and serves them over HTTP so the
web app calls the real Python models instead of a JS approximation.

Run:
    pip install flask flask-cors joblib opencv-python scikit-learn --break-system-packages
    python3 server.py
Then open citymind_ai_enhanced.html in your browser as normal — it will
auto-detect the server at http://localhost:5000 and switch from the
in-browser JS fallback to real model predictions. If the server isn't
running, the page still works standalone (JS fallback), it just won't be
using the trained Python models.

Endpoints:
    GET  /api/health
    GET  /api/locations
    GET  /api/samples?kind=traffic|fire&n=4   -> real images + real model predictions
    POST /api/route          {start, dest, blocked: [["from","to"], ...]}
    POST /api/garbage/nearest {trucks: [{id,status,at}], userLoc}
"""
import os
import cv2
import glob
import random
import base64
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

import time
import logging
from collections import deque

import classifier
import dqn_routing

# ---- logging: goes to stdout (Render > Logs tab) AND an in-memory buffer served at /api/logs
LOG_BUFFER = deque(maxlen=300)


class _BufferHandler(logging.Handler):
    def emit(self, record):
        LOG_BUFFER.append(self.format(record))


_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
log = logging.getLogger("citymind")
log.setLevel(logging.INFO)
for h in (logging.StreamHandler(), _BufferHandler()):
    h.setFormatter(_fmt)
    log.addHandler(h)
log.propagate = False
START_TIME = time.time()

app = Flask(__name__)
CORS(app)  # allow the HTML (opened via file:// or any origin) to call this API

BASE = os.path.dirname(__file__)
MODEL_CACHE = os.path.join(BASE, "_model_cache.joblib")

# ----------------------------------------------------------------- MODELS --
log.info("Loading/training models (real dataset, gradient boosting + DQN)...")
import sklearn


def _load_cached_models():
    """Use the pre-trained cache only if it was built with the same scikit-learn version
    (pickled models are not guaranteed portable across versions) -- otherwise retrain."""
    if not os.path.exists(MODEL_CACHE):
        return None
    try:
        cache = joblib.load(MODEL_CACHE)
        if cache.get("sklearn_version") != sklearn.__version__:
            log.warning(f"Cache built with scikit-learn {cache.get('sklearn_version')}, "
                        f"running {sklearn.__version__} -> retraining instead.")
            return None
        return cache
    except Exception as e:  # corrupt / incompatible pickle
        log.warning(f"Could not read model cache ({e}) -> retraining instead.")
        return None


cache = _load_cached_models()
if cache:
    traffic_clf, fire_clf = cache["traffic_clf"], cache["fire_clf"]
    log.info(f"Loaded cached classifiers (held-out accuracy: traffic {cache.get('traffic_acc', 0):.3f}, "
             f"fire {cache.get('fire_acc', 0):.3f}).")
else:
    traffic_clf, t_acc, _ = classifier.train_traffic_model()
    fire_clf, f_acc, _ = classifier.train_fire_model()
    joblib.dump({"traffic_clf": traffic_clf, "fire_clf": fire_clf, "sklearn_version": sklearn.__version__,
                 "traffic_acc": t_acc, "fire_acc": f_acc}, MODEL_CACHE)
    log.info(f"Trained fresh. Traffic acc={t_acc:.3f} Fire acc={f_acc:.3f}")

log.info("Training DQN router (takes ~15-20s)...")
_t0 = time.time()
router = dqn_routing.DQNRouter()
router.train(episodes=1200)
log.info(f"DQN router ready ({time.time() - _t0:.1f}s, 1200 episodes, "
         f"{len(dqn_routing.NODES)} nodes / {len(dqn_routing.EDGES)} directed roads).")

# The frontend (city_offline.html) uses short slug ids for locations (e.g. "lakeview"),
# while dqn_routing.py's graph is keyed by full display names ("Lakeview Apartments").
# This map keeps the two in sync -- IDs here MUST match NAMED_LOCATIONS in the HTML.
ID_TO_NAME = {
    "mg_road": "MG Road Junction", "lakeview": "Lakeview Apartments",
    "central_hosp": "Central Hospital", "tech_circle": "Tech Park Circle",
    "silicon_it": "Silicon Heights IT Park", "riverside_br": "Riverside Bridge",
    "greenfield": "Greenfield Apartments", "market_st": "Market Street",
    "old_town": "Old Town Square", "dump_yard": "Industrial Dump Yard Road",
    "sunrise_apt": "Sunrise Apartments", "sunrise_hosp": "Sunrise Multispecialty Hospital",
    "grand_hotel": "City Grand Hotel", "spice_resto": "Spice Route Restaurant",
    "neon_cafe": "Neon Cafe",
    "northgate_apt": "Northgate Apartments", "harbor_hotel": "Harbor View Hotel",
    "blue_moon": "Blue Moon Diner", "westside_hosp": "Westside General Hospital",
    "pixel_it": "Pixel IT Hub", "central_park_av": "Central Park Avenue",
}
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}

# ------------------------------------------------------------------ UTILS --

def img_to_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


# ------------------------------------------------------------------ ROUTES --

@app.before_request
def _t_start():
    request._t0 = time.time()


@app.after_request
def _log_request(resp):
    if request.path != "/api/logs":
        ms = (time.time() - getattr(request, "_t0", time.time())) * 1000
        log.info(f"{request.method} {request.full_path.rstrip('?')} -> {resp.status_code} ({ms:.0f} ms)")
    return resp


@app.route("/")
def home():
    return jsonify({"service": "CityMind Backend", "status": "ok"})


@app.route("/api/logs")
def logs():
    """Last 300 log lines as plain text -- open this URL in a browser during the review."""
    lines = "\n".join(LOG_BUFFER)
    return app.response_class(lines + "\n", mimetype="text/plain")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "CityMind Backend",
                     "usingRealData": classifier.USING_REAL_DATA,
                     "nodes": dqn_routing.NODES, "edges": len(dqn_routing.EDGES),
                     "uptime_s": round(time.time() - START_TIME)})


@app.route("/api/locations")
def locations():
    return jsonify([{"id": i, "name": n} for i, n in ID_TO_NAME.items()])


@app.route("/api/samples")
def samples():
    kind = request.args.get("kind", "traffic")
    n = int(request.args.get("n", 4))
    if kind == "traffic":
        files = glob.glob(os.path.join(classifier.TRAFFIC_DIR, "*", "*.jpg")) + \
                glob.glob(os.path.join(classifier.TRAFFIC_DIR, "*", "*.png"))
    else:
        files = glob.glob(os.path.join(classifier.FIRE_DIR, "*", "*.jpg")) + \
                glob.glob(os.path.join(classifier.FIRE_DIR, "*", "*.png"))
    picks = random.sample(files, min(n, len(files)))
    results = []
    for path in picks:
        img = cv2.imread(path)
        if img is None:
            continue
        true_label = os.path.basename(os.path.dirname(path))
        small = cv2.resize(img, (200, 150))
        if kind == "traffic":
            pred = classifier.classify_traffic_image(traffic_clf, img)
            log.info(f"  traffic sample {os.path.basename(path)}: true={true_label} "
                     f"pred={pred['level']} conf={pred['confidence']}")
            results.append({
                "kind": "traffic", "trueLabel": true_label,
                "image": img_to_b64(small),
                "traffic": pred["traffic"], "level": pred["level"],
                "confidence": pred["confidence"], "features": pred["features"],
            })
        else:
            pred = classifier.classify_fire_image(fire_clf, img)
            log.info(f"  fire sample {os.path.basename(path)}: true={true_label} "
                     f"pred={pred['fire']} conf={pred['confidence']}")
            results.append({
                "kind": "fire", "trueLabel": true_label,
                "image": img_to_b64(small),
                "fire": pred["fire"], "confidence": pred["confidence"],
                "features": pred["features"],
            })
    return jsonify(results)


@app.route("/api/route", methods=["POST"])
def route():
    data = request.get_json(force=True)
    start_id, dest_id = data["start"], data["dest"]
    start, dest = ID_TO_NAME.get(start_id, start_id), ID_TO_NAME.get(dest_id, dest_id)
    blocked_pairs = data.get("blocked", [])  # list of [from_id, to_id] -- hard road closures
    blocked_names = {(ID_TO_NAME.get(a, a), ID_TO_NAME.get(b, b)) for a, b in blocked_pairs}
    result = router.find_best_route(start, dest, blocked_edges=blocked_names)
    log.info(f"  route {start} -> {dest} | blocked={len(blocked_names)} | solver={result['usedSolver']} | "
             f"dqn_reached={result['dqnReached']} | {result['distance_km']} km, "
             f"{result['estimated_time_min']} min")
    result["path"] = [NAME_TO_ID.get(n, n) for n in result["path"]]
    result["dqnPolicyPath"] = [NAME_TO_ID.get(n, n) for n in result.get("dqnPolicyPath", [])]
    result["exactPath"] = [NAME_TO_ID.get(n, n) for n in result.get("exactPath", [])]
    return jsonify(result)


@app.route("/api/garbage/nearest", methods=["POST"])
def garbage_nearest():
    data = request.get_json(force=True)
    trucks = data["trucks"]
    user_loc = ID_TO_NAME.get(data["userLoc"], data["userLoc"])
    available = [t for t in trucks if t["status"] == "available"]
    if not available:
        return jsonify({"none": True})
    scored = []
    for t in available:
        truck_at = ID_TO_NAME.get(t["at"], t["at"])
        r = router.find_best_route(truck_at, user_loc)
        if r["reached"]:
            r["path"] = [NAME_TO_ID.get(n, n) for n in r["path"]]
            scored.append({"truck": t, "route": r})
    if not scored:
        return jsonify({"none": True})
    scored.sort(key=lambda x: x["route"]["estimated_time_min"])
    log.info(f"  garbage: nearest truck {scored[0]['truck']['id']} -> {user_loc} "
             f"ETA {scored[0]['route']['estimated_time_min']} min")
    return jsonify(scored[0])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # cloud hosts (Render, Railway, etc.) inject PORT
    log.info(f"CityMind AI backend ready at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
