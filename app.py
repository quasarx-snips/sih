"""
LunaX backend server
=====================
Wires the three modular stages the frontend needs:

  1. CLAHE enhancement  -> baseline image used for every downstream step
  2. SIH26166 matcher   -> lunax.run_lunax_from_arrays() (verification + registration)
  3. Hazard-map engine  -> main.generate_hazard_overlay() (AI U-Net + morphology)

Run:
    pip install -r requirements.txt
    python app.py

Then point the frontend's API_BASE constant (see upload.html / analysis.html)
at http://localhost:5000 (default) or wherever this is hosted.
"""

import os
import base64
import hashlib
import traceback
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import main as hazard_engine  # your uploaded main.py (one added line: saves the risk map)
import terrain_module  # Landing Zones / 3D Route, ported from lunar_map.ipynb
from lunax import PipelineConfig, run_lunax_from_arrays

app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------------------------
# Force JSON on ALL error responses (404, 405, 500, ...) instead of Flask's
# default HTML error pages. Without this, a stale server / typo'd route /
# unhandled exception returns "<!doctype html>..." and every frontend
# `await res.json()` call blows up with a confusing parse error instead of
# a readable message.
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def _not_found(e):
    return jsonify({
        "success": False,
        "error": f"No such API route: {request.method} {request.path}. "
                  f"Did you restart the Flask server after editing app.py?",
    }), 404


@app.errorhandler(405)
def _method_not_allowed(e):
    return jsonify({
        "success": False,
        "error": f"Method {request.method} not allowed on {request.path}.",
    }), 405


@app.errorhandler(500)
def _internal_error(e):
    return jsonify({"success": False, "error": "Internal server error. Check the Flask terminal log."}), 500

# In-memory cache of the last hazard-map run. Landing Zones / Route / 3D
# Terrain all reuse this instead of re-running the AI + morphology pipeline.
# Single-user local demo assumption -- not meant for concurrent sessions.
LAST_RESULT = {}

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Drop crater_unet.onnx (+ its .data sidecar) into backend/models/, or point
# this at your existing model folder via the CRATER_MODEL_PATH env var.
MODEL_PATH = os.environ.get(
    "CRATER_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "crater_unet.onnx"),
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# The UI and API can be deployed together behind one HTTPS origin.  Keeping
# the allow-list explicit avoids turning the project directory into a public
# file server while still allowing normal page navigation and IndexedDB helper.
PUBLIC_FILES = {
    "landing_page.html", "upload.html", "matching.html", "hazard-map.html",
    "analysis.html", "landing-zones.html", "route-analysis.html",
    "science-map.html", "team.html", "repo.html", "storage.js",
}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def data_url_to_cv2(data_url: str) -> np.ndarray:
    """Decode a base64 data URL (or bare base64 string) into a BGR OpenCV image."""
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img


def cv2_to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode the result image.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _json_value(value):
    """Convert NumPy values from the matching pipeline into API-safe values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.5, tile_grid_size=(8, 8)) -> np.ndarray:
    """CLAHE on the L channel of LAB space -- boosts local contrast without blowing out highlights."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def frontend_index():
    return send_from_directory(BASE_DIR, "landing_page.html")


@app.route("/<path:filename>", methods=["GET"])
def frontend_file(filename):
    if filename in PUBLIC_FILES:
        return send_from_directory(BASE_DIR, filename)
    return jsonify({"success": False, "error": "No such page or asset."}), 404


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_found": os.path.isfile(MODEL_PATH),
    })


@app.route("/api/enhance", methods=["POST"])
def enhance():
    """Step 1: CLAHE-enhance the raw upload. This becomes the baseline image."""
    try:
        payload = request.get_json(force=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"success": False, "error": "No image provided."}), 400

        img = data_url_to_cv2(image_data)
        enhanced = apply_clahe(img)
        return jsonify({"success": True, "image": cv2_to_data_url(enhanced)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/image-match", methods=["POST"])
def image_match():
    """SIH26166 correspondence, geometric verification and registration.

    This intentionally lives beside—not inside—the hazard engine: matching
    produces a trusted registered observation first; hazard analysis then
    continues with the selected uploaded observation.
    """
    try:
        payload = request.get_json(force=True) or {}
        source_data = payload.get("source")
        reference_data = payload.get("reference")
        if not source_data or not reference_data:
            return jsonify({"success": False, "error": "Both source and reference images are required."}), 400

        source = data_url_to_cv2(source_data)
        reference = data_url_to_cv2(reference_data)
        run_dir = os.path.join(OUTPUT_DIR, "matching_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        result = run_lunax_from_arrays(source, reference, PipelineConfig(
            verbose=False,
            save_outputs=True,
            output_dir=run_dir,
            crater_model_path=MODEL_PATH if os.path.isfile(MODEL_PATH) else None,
        ))
        if not result.success:
            return jsonify({"success": False, "error": result.error or "Image matching failed."}), 422

        artifacts = {}
        for name, path in result.diagnostics.get("artifact_paths", {}).items():
            if name == "report" or not os.path.isfile(path):
                continue
            image = cv2.imread(path)
            if image is not None:
                artifacts[name] = cv2_to_data_url(image)
        return jsonify({
            "success": True,
            "metrics": _json_value(result.metrics),
            "diagnostics": _json_value(result.diagnostics),
            "artifacts": artifacts,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def _image_hash(img_bgr: np.ndarray) -> str:
    """Cheap content fingerprint so we can tell 'same baseline as last time'
    apart from 'a different image was uploaded' without re-hashing megabytes
    of pixels on every request."""
    return hashlib.md5(img_bgr.tobytes()).hexdigest()


def _run_hazard_pipeline(img: np.ndarray):
    """The actual AI + morphology hazard computation, shared by /api/hazard-map
    AND the auto-fallback in Landing Zones / Route / 3D Terrain below. Caches
    everything downstream modules need in LAST_RESULT, keyed by image hash so
    a request for the SAME baseline never re-runs the expensive tiled AI pass."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Put crater_unet.onnx "
            "(and its .onnx.data sidecar) in backend/models/, or set "
            "the CRATER_MODEL_PATH environment variable."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = os.path.join(TEMP_DIR, f"baseline_{timestamp}.png")
    try:
        cv2.imwrite(temp_path, img)

        out_path = hazard_engine.generate_hazard_overlay(
            image_path=temp_path,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            show=False,
        )

        result_img = cv2.imread(out_path)
        if result_img is None:
            raise ValueError("Hazard overlay was generated but could not be read back.")

        risk_path = os.path.splitext(out_path)[0] + "_risk.npy"

        # Cache the baseline image + the exact numeric risk map main.py just
        # produced, so Landing Zones / Route / 3D Terrain reuse this hazard
        # map instead of recomputing it, as long as it's still the same image.
        LAST_RESULT.clear()
        LAST_RESULT["image_hash"] = _image_hash(img)
        LAST_RESULT["baseline_bgr"] = img
        LAST_RESULT["gray_u8"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        LAST_RESULT["total_risk"] = np.load(risk_path)
        LAST_RESULT["overlay_data_url"] = cv2_to_data_url(result_img)
        LAST_RESULT["output_path"] = out_path
        LAST_RESULT["masks"] = None  # computed lazily by _get_masks()
        LAST_RESULT["routes"] = None
        LAST_RESULT["start"] = None
        LAST_RESULT["goal"] = None

        return out_path
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _ensure_hazard_ready(image_data: str = None):
    """Make sure LAST_RESULT holds a hazard map for the CURRENT baseline.

    - If the caller supplied `image` (Landing Zones / Route / 3D Terrain all
      now do this, same as Hazard Map already did): decode it, and only
      re-run the expensive AI + morphology pipeline if we don't already have
      a cached result for this exact image. This is what makes these modules
      "just work" on a fresh page load or after a server restart, instead of
      requiring the user to visit the Hazard Map section and click a button
      first.
    - If no image was supplied (older frontend / direct API use): fall back
      to requiring an existing cache, same as before.
    """
    if image_data:
        img = data_url_to_cv2(image_data)
        cache_hit = LAST_RESULT.get("image_hash") == _image_hash(img)
        if not cache_hit:
            _run_hazard_pipeline(img)
        return cache_hit
    if "total_risk" not in LAST_RESULT:
        raise ValueError("Run hazard map analysis first.")


@app.route("/api/hazard-map", methods=["POST"])
def hazard_map():
    """Step 2: run the AI + morphology hazard engine on the CLAHE baseline image."""
    try:
        payload = request.get_json(force=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"success": False, "error": "No baseline image provided."}), 400

        cache_hit = _ensure_hazard_ready(image_data)
        img = LAST_RESULT["baseline_bgr"]

        return jsonify({
            "success": True,
            "image": LAST_RESULT["overlay_data_url"],
            "output_path": LAST_RESULT["output_path"],
            "cached": bool(cache_hit),
            "image_width": int(img.shape[1]),
            "image_height": int(img.shape[0]),
        })
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


def _get_masks():
    """Lazily compute + cache the full-image morphological masks Module 2 needs."""
    if LAST_RESULT.get("masks") is None:
        LAST_RESULT["masks"] = terrain_module.compute_terrain_masks(LAST_RESULT["baseline_bgr"])
    return LAST_RESULT["masks"]


@app.route("/api/landing-sites", methods=["POST"])
def landing_sites():
    """Module 2: LUNAX 5-factor landing-site suitability.

    Pass the current baseline image and this computes the hazard map on the
    fly if it isn't already cached for this exact image -- no need to visit
    the Hazard Map section first.
    """
    try:
        payload = request.get_json(force=True) or {}
        _ensure_hazard_ready(payload.get("image"))

        masks = _get_masks()
        sites = terrain_module.compute_landing_sites(
            gray_u8=LAST_RESULT["gray_u8"],
            crater_mask=masks["crater_mask"],
            boulder_mask=masks["boulder_mask"],
            steep_mask=masks["steep_mask"],
            total_risk=LAST_RESULT["total_risk"],
            grad_mag=masks["grad_mag"],
        )
        h, w = LAST_RESULT["gray_u8"].shape
        return jsonify({
            "success": True,
            "sites": sites,
            "image_width": int(w),
            "image_height": int(h),
            "hazard_image": LAST_RESULT.get("overlay_data_url"),
        })
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/route", methods=["POST"])
def route():
    """Module 3: risk-aware A* routing between a dropped start/goal pin.

    Also accepts the baseline image, same auto-compute fallback as above.
    """
    try:
        payload = request.get_json(force=True) or {}
        _ensure_hazard_ready(payload.get("image"))

        start = payload.get("start")
        goal = payload.get("goal")
        if not start or not goal:
            return jsonify({"success": False, "error": "Both start and goal pins are required."}), 400

        h, w = LAST_RESULT["gray_u8"].shape
        start_xy = (max(0, min(w - 1, int(start["x"]))), max(0, min(h - 1, int(start["y"]))))
        goal_xy = (max(0, min(w - 1, int(goal["x"]))), max(0, min(h - 1, int(goal["y"]))))

        routes = terrain_module.compute_routes(start_xy, goal_xy, LAST_RESULT["total_risk"])
        if not routes:
            return jsonify({"success": False, "error": "No valid path found between those points."}), 400

        LAST_RESULT["routes"] = routes
        LAST_RESULT["start"] = start_xy
        LAST_RESULT["goal"] = goal_xy

        return jsonify({"success": True, "routes": routes, "image_width": int(w), "image_height": int(h)})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/terrain3d", methods=["POST"])
def terrain3d():
    """Module 4: grounded 3D heightmap + route projection, for Plotly.js.

    Also accepts the baseline image, same auto-compute fallback as above --
    if called without having planned a route first, still returns the bare
    terrain surface (no route overlay), rather than erroring out.
    """
    try:
        payload = request.get_json(force=True) or {}
        _ensure_hazard_ready(payload.get("image"))

        data = terrain_module.compute_terrain_3d(
            gray_u8=LAST_RESULT["gray_u8"],
            total_risk=LAST_RESULT["total_risk"],
            routes=LAST_RESULT.get("routes"),
            start=LAST_RESULT.get("start"),
            goal=LAST_RESULT.get("goal"),
        )
        return jsonify({"success": True, **data})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Print the live route table on every startup -- since use_reloader=False
    # means code edits never take effect until you actually Ctrl+C and rerun
    # this script, this is the fastest way to confirm you're not talking to
    # a stale process.
    print("Registered API routes:")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        if rule.rule.startswith("/api"):
            print(f"  {','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'})):8s} {rule.rule}")

    # use_reloader=False and this __main__ guard both matter here: main.py
    # spins up a ProcessPoolExecutor, and on Windows (spawn start method)
    # each worker re-imports this file. Without the guard / with the
    # reloader on, that re-import would try to relaunch the Flask server
    # inside every worker process.
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)
