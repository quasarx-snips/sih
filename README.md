# LunaX — Team LunaX · SIH 2026

LunaX connects the browser flow to three modular processing stages:
CLAHE preprocessing, SIH26166 lunar correspondence verification and
registration (`lunax/`), then hazard detection and downstream terrain tools.

The browser flow is `landing_page.html` → `upload.html` → `matching.html` →
`hazard-map.html` → `analysis.html`, where Landing Sites, risk-aware A* + 3D
Route, Hazard Map, and Science Map remain available.

## 1. Install

```bash
cd /path/to/LunaX
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

## 2. Add your model

Copy `crater_unet.onnx` **and** `crater_unet.onnx.data` (the external-data
sidecar ONNX split off) into `backend/models/`, so you have:

```
backend/models/crater_unet.onnx
backend/models/crater_unet.onnx.data
```

Both files must sit in the same folder — onnxruntime resolves the `.data`
file by relative name next to the `.onnx` file.

Don't want to move the files? Instead set an environment variable pointing
at wherever they already live, e.g. on Windows:

```powershell
$env:CRATER_MODEL_PATH = "C:\Users\bijan\Desktop\Backup\models\crater_unet.onnx"
```

## 3. Run

```bash
python app.py
```

You should see Flask start on `http://localhost:5000`. Open the LunaX UI at
**http://localhost:5000/** (not by double-clicking an HTML file). This keeps
the frontend and API on one origin. The pages also fall back to the local API
when opened through `file://`, so existing local workflows still work.

Sanity-check the backend first:

```bash
curl http://localhost:5000/api/health
```

For a single-server deployment, this Flask app now serves both the API and
the UI: deploy it behind your HTTPS reverse proxy and open the site root.
The frontend automatically uses its current deployed origin. If you host the
frontend separately, define `window.LUNAX_API_BASE` before each page script
with the public HTTPS API URL.

`"model_found": true` means the ONNX model was located correctly.

### If the UI says “backend not running”

1. Keep the terminal running `python app.py`; do not close it after the
   startup message.
2. Visit `http://localhost:5000/api/health` in the browser. It must return
   JSON with `"status": "ok"`.
3. Open `http://localhost:5000/` in the same browser. Do not use a different
   port from an editor preview server unless you deliberately configure a
   separate API URL.
4. If Python exits before showing the Flask URL, install dependencies with
   `python -m pip install -r requirements.txt`, then run `python app.py`
   again. The most common missing package is OpenCV (`cv2`).
5. If port 5000 is already occupied on Windows, find and stop the old process:

   ```powershell
   Get-NetTCPConnection -LocalPort 5000 | Select-Object -Expand OwningProcess
   Stop-Process -Id <PROCESS_ID>
   ```

## Deploy to an Ubuntu VPS (recommended)

This uses one server for both the LunaX UI and API, so no frontend CORS or
API URL changes are needed. Use Ubuntu 22.04/24.04 with at least 4 GB RAM;
the ONNX model and CPU hazard inference are not a good fit for a tiny server.

1. Point your domain's `A` record to the server IP, then SSH in:

   ```bash
   ssh root@YOUR_SERVER_IP
   ```

2. Install system packages and create a non-root app user:

   ```bash
   apt update && apt upgrade -y
   apt install -y python3 python3-venv python3-pip nginx git certbot python3-certbot-nginx
   adduser --disabled-password --gecos "" lunax
   ```

3. Upload this project (including `models/crater_unet.onnx` and
   `models/crater_unet.onnx.data`) to `/var/www/lunax`, then grant ownership:

   ```bash
   mkdir -p /var/www/lunax
   chown -R lunax:lunax /var/www/lunax
   ```

   You can clone your own repository into that directory, or copy the project
   with `scp`/SFTP. Do not publish the model files separately—keep them in the
   project's `models/` directory.

4. Install Python packages as the app user:

   ```bash
   sudo -u lunax python3 -m venv /var/www/lunax/.venv
   sudo -u lunax /var/www/lunax/.venv/bin/pip install --upgrade pip
   sudo -u lunax /var/www/lunax/.venv/bin/pip install -r /var/www/lunax/requirements.txt
   ```

5. Check the app can load the model:

   ```bash
   cd /var/www/lunax
   sudo -u lunax .venv/bin/python -c "import app; print(app.MODEL_PATH, app.os.path.isfile(app.MODEL_PATH))"
   ```

   The final value must be `True`.

6. Create `/etc/systemd/system/lunax.service` with this content:

   ```ini
   [Unit]
   Description=LunaX web application
   After=network.target

   [Service]
   User=lunax
   Group=lunax
   WorkingDirectory=/var/www/lunax
   Environment="PATH=/var/www/lunax/.venv/bin"
   ExecStart=/var/www/lunax/.venv/bin/gunicorn --workers 1 --threads 4 --timeout 300 --bind 127.0.0.1:8000 app:app
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   One Gunicorn worker is intentional: the active hazard result is cached in
   memory and is shared by the Hazard Map, Landing Sites, A*, and 3D views.
   Start it with:

   ```bash
   systemctl daemon-reload
   systemctl enable --now lunax
   systemctl status lunax
   ```

7. Create `/etc/nginx/sites-available/lunax` (replace `lunax.example.com`):

   ```nginx
   server {
       server_name lunax.example.com;
       client_max_body_size 30m;
       proxy_read_timeout 300s;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Enable HTTPS:

   ```bash
   ln -s /etc/nginx/sites-available/lunax /etc/nginx/sites-enabled/lunax
   nginx -t && systemctl reload nginx
   certbot --nginx -d lunax.example.com
   ```

8. Open `https://lunax.example.com` and verify
   `https://lunax.example.com/api/health` reports `"model_found": true`.

For updates, upload/pull the new code, run the same `pip install -r ...`
command if requirements changed, then run `systemctl restart lunax`.

## API

| Endpoint            | Method | Body                          | Returns                                  |
|----------------------|--------|--------------------------------|-------------------------------------------|
| `/api/health`        | GET    | —                               | model path + whether it was found         |
| `/api/enhance`       | POST   | `{ "image": "<data URL>" }`    | `{ success, image: "<data URL>" }` (CLAHE) |
| `/api/image-match`   | POST   | `{ "source": "<data URL>", "reference": "<data URL>" }` | SIH26166 metrics + registration artifacts |
| `/api/hazard-map`    | POST   | `{ "image": "<data URL>" }`    | Overlay; `cached: true` when reused |

`/api/hazard-map` expects the **CLAHE-enhanced** image (the output of
`/api/enhance`) as its baseline, matching your intended pipeline:
raw upload → CLAHE baseline → correspondence verification → hazard map.

`/api/image-match` keeps the original SIH26166 implementation modular in
`lunax/`. It returns candidate/inlier counts, inlier ratio, reprojection
error statistics, coverage, evidence confidence, transform details, and
registration visualizations. Matching artifacts are saved as their own
timestamped directory under `outputs/`.

The first hazard run is cached by the CLAHE baseline's content hash. Opening
the Hazard Map again from the Analysis Suite, or using Landing Sites, A* and
3D Terrain with that same image, reuses the already-generated risk map rather
than running the ONNX/morphology pipeline again. The cache resets when the
server restarts or a different image is uploaded.

## Notes

- Tiling, ONNX inference, and morphology run exactly as written in your
  `main.py` (unmodified) — `ProcessPoolExecutor` with 4 workers, 512px
  tiles, 100px overlap.
- A processed image on a laptop CPU can take anywhere from several seconds
  to a couple of minutes depending on resolution — the frontend shows a
  progress state while it waits.
- Temp baseline files written to `backend/temp/` are deleted after each
  request; final overlays accumulate in `backend/outputs/` (same
  `_integrated_overlay.png` naming your script already uses).

---

## COMPREHENSIVE TECHNICAL DOCUMENTATION

### PROJECT ARCHITECTURE OVERVIEW

LunaX is a lunar surface analysis system that combines computer vision, machine learning, and web technologies to provide real-time hazard detection, image registration, and terrain analysis for lunar missions. The system follows a three-stage modular pipeline:

1. **Image Enhancement** - CLAHE preprocessing for contrast enhancement
2. **Image Registration** - SIH26166 correspondence verification and geometric transformation
3. **Hazard Analysis** - AI-powered crater detection and terrain assessment

### COMPLETE FILE STRUCTURE

```
SIH26166/
├── app.py                      # Main Flask web server and API orchestration
├── main.py                     # Hazard detection engine (AI + morphology)
├── terrain_module.py           # Terrain analysis (landing sites, routes, 3D)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusions
├── storage.js                  # IndexedDB helper for frontend storage
│
├── lunax/                      # LunaX image registration package
│   ├── __init__.py            # Package facade, exports main API
│   ├── pipeline.py            # End-to-end registration orchestration
│   ├── preprocessing.py       # Image normalization and CLAHE enhancement
│   ├── features.py            # Feature extraction (SIFT, terrain landmarks)
│   ├── matching.py            # Feature matching and correspondence
│   ├── geometry.py            # Geometric verification (RANSAC, transforms)
│   ├── registration.py        # Image warping and registration
│   ├── refinement.py          # Subpixel correspondence refinement
│   └── metrics.py             # Registration quality metrics
│
├── models/                     # AI model storage
│   ├── crater_unet.onnx       # U-Net model for crater detection
│   └── crater_unet.onnx.data  # ONNX external data sidecar
│
├── outputs/                    # Generated analysis outputs
│   └── matching_*/            # Timestamped registration results
│
└── Frontend HTML Files:
    ├── landing_page.html       # Application landing page
    ├── upload.html            # Image upload interface
    ├── matching.html          # Registration visualization
    ├── hazard-map.html        # Hazard overlay display
    ├── analysis.html          # Analysis suite navigation
    ├── landing-zones.html     # Landing site selection
    ├── route-analysis.html    # Route planning interface
    └── science-map.html       # Scientific visualization
```

### DETAILED FILE-BY-FILE FUNCTIONALITY

#### Core Application Files

**`app.py`** - Flask Web Server & API Orchestration
- **Purpose**: Main entry point, serves both frontend and backend API
- **Key Components**:
  - Flask application with CORS enabled for cross-origin requests
  - Custom error handlers returning JSON instead of HTML
  - In-memory caching system (`LAST_RESULT`) for hazard map reuse
  - Routes for serving static HTML files and API endpoints
- **API Endpoints**:
  - `GET /` - Serves landing page
  - `GET /<filename>` - Serves static files (HTML, JS)
  - `GET /api/health` - Health check and model availability
  - `POST /api/enhance` - CLAHE image enhancement
  - `POST /api/image-match` - SIH26166 image registration
  - `POST /api/hazard-map` - AI hazard detection
  - `POST /api/landing-sites` - Landing site suitability analysis
  - `POST /api/route` - Risk-aware A* pathfinding
  - `POST /api/terrain3d` - 3D terrain visualization data
- **Caching Strategy**: Uses MD5 hash of baseline image to cache expensive AI computations

**`main.py`** - Hazard Detection Engine
- **Purpose**: Core AI + morphological hazard detection pipeline
- **Key Algorithms**:
  - **AI Inference**: U-Net ONNX model for crater detection using tiled processing
  - **Morphological Analysis**: Multi-scale black-hat operations for crater detection
  - **Risk Mapping**: Distance transform for continuous risk assessment
- **Parameters**:
  - `TILE_SIZE = 512`, `OVERLAP = 100` - Image tiling for memory efficiency
  - `CRATER_SCALES = [0.012, 0.022, 0.035, 0.05]` - Multi-scale crater detection
  - `ProcessPoolExecutor` with 4 workers for parallel processing
- **Main Function**: `generate_hazard_overlay()` - Orchestrates full pipeline
- **Output**: Color-coded hazard overlay + numerical risk map (`_risk.npy`)

**`terrain_module.py`** - Terrain Analysis Module
- **Purpose**: Downstream terrain analysis using pre-computed risk maps
- **Key Functions**:
  - `compute_terrain_masks()` - Full-image morphological masks (crater, boulder, steep)
  - `compute_landing_sites()` - 5-factor landing site suitability engine
  - `astar_pathfinding()` - Risk-aware A* pathfinding algorithm
  - `compute_routes()` - Multi-strategy route generation (Distance/Balanced/Safety)
  - `compute_terrain_3d()` - 3D terrain data for Plotly.js visualization
- **Landing Site Factors**: Flatness, smoothness, hazard-free score, hard constraints
- **Route Strategies**: Different risk weights for safety vs. distance optimization

#### LunaX Registration Package

**`lunax/__init__.py`** - Package Facade
- **Exports**: `PipelineConfig`, `RegistrationResult`, `run_lunax_from_arrays`, `run_lunax_registration`, `print_lunax_pipeline_report`
- **Purpose**: Stable API surface for image registration

**`lunax/pipeline.py`** - Registration Orchestration
- **Purpose**: End-to-end lunar correspondence verification and registration
- **Key Classes**:
  - `PipelineConfig` - Configuration for all pipeline stages
  - `RegistrationResult` - Comprehensive result data structure
- **Main Function**: `run_lunax_from_arrays()` - Complete registration pipeline
- **Pipeline Stages**:
  1. Preprocessing (normalization + CLAHE)
  2. Feature extraction (SIFT + terrain landmarks)
  3. Feature matching (ratio test + mutual consistency)
  4. Geometric verification (RANSAC)
  5. Correspondence refinement (subpixel)
  6. Image registration (warping)
  7. Quality evaluation (metrics)

**`lunax/preprocessing.py`** - Image Preprocessing
- **Purpose**: Image normalization and contrast enhancement
- **Key Class**: `ImagePreprocessor`
- **Methods**:
  - `load()` - Load image from disk
  - `normalize()` - Convert to normalized grayscale
  - `enhance()` - Apply CLAHE in LAB color space
  - `process()` - Complete preprocessing pipeline

**`lunax/features.py`** - Feature Extraction
- **Purpose**: Multi-modal feature extraction for lunar terrain
- **Key Classes**:
  - `SiftDetector` - SIFT keypoint and descriptor extraction
  - `TerrainFeatureExtractor` - Unified terrain feature extraction
  - `ONNXCraterDetector` - AI-based crater detection
  - `RidgeDetector` - Ridge/line structure detection
  - `TextureGradientDetector` - Texture/gradient change detection
- **Feature Types**: Craters, ridges, texture, SIFT keypoints
- **Output**: Structured feature records with metadata and descriptors

**`lunax/matching.py`** - Feature Matching
- **Purpose**: Robust feature matching between image pairs
- **Key Functions**:
  - `match_descriptors()` - Descriptor matching with ratio test
  - `ratio_test()` - Lowe's ratio test for match filtering
  - `mutual_consistency_filter()` - Bidirectional consistency check
  - `match_feature_sets()` - Complete matching pipeline
  - `visualize_matches()` - Match visualization
- **Matching Strategies**: BF matcher, ratio test, mutual consistency

**`lunax/geometry.py`** - Geometric Verification
- **Purpose**: Robust geometric transformation estimation
- **Key Classes**:
  - `GeometricVerificationConfig` - RANSAC configuration
  - `VerificationDiagnostics` - Detailed verification results
  - `RegistrationResult` - Complete registration output
- **Transform Models**: Translation, Similarity, Affine, Homography
- **Algorithm**: RANSAC with adaptive iteration count and refitting
- **Quality Metrics**: Inlier ratio, reprojection error, spatial coverage

**`lunax/registration.py`** - Image Registration
- **Purpose**: Apply geometric transformations for image alignment
- **Key Functions**:
  - `register_image()` - Apply transformation and generate visualizations
  - `warp_image()` - Projective image warping
  - `create_overlay()` - Alpha-blended overlay visualization
  - `create_difference_map()` - Difference map for quality assessment
- **Output**: Registered image with comprehensive quality metrics

**`lunax/refinement.py`** - Correspondence Refinement
- **Purpose**: Subpixel accuracy improvement for correspondences
- **Key Functions**:
  - `refine_correspondence()` - Single point refinement
  - `refine_correspondences()` - Batch refinement
  - `calculate_local_registration_error()` - Error calculation
- **Method**: Local cross-correlation with subpixel peak fitting

**`lunax/metrics.py`** - Registration Quality Metrics
- **Purpose**: Comprehensive registration quality assessment
- **Key Functions**:
  - `calculate_inlier_ratio()` - Inlier percentage calculation
  - `calculate_reprojection_errors()` - Reprojection error computation
  - `calculate_rmse()` - Root mean square error
  - `calculate_spatial_coverage()` - Spatial distribution analysis
  - `evaluate_registration()` - Complete evaluation pipeline
- **Metrics**: RMSE, median error, spatial coverage, confidence score

### DATA FLOW ARCHITECTURE

#### Complete User Journey

1. **Landing Page** (`landing_page.html`)
   - User sees LunaX branding and launch button
   - Animation and visual effects for engagement

2. **Image Upload** (`upload.html`)
   - User uploads lunar surface image
   - Frontend sends image to `/api/enhance`
   - Server applies CLAHE enhancement
   - Enhanced image returned to frontend
   - User proceeds to matching

3. **Image Registration** (`matching.html`)
   - User selects reference image from database
   - Frontend sends source + reference to `/api/image-match`
   - LunaX pipeline processes both images:
     - Preprocessing (normalization + CLAHE)
     - Feature extraction (SIFT + terrain landmarks)
     - Feature matching (ratio test + mutual consistency)
     - Geometric verification (RANSAC)
     - Correspondence refinement (subpixel)
     - Image registration (warping)
   - Results displayed: metrics, visualizations, transformation matrix

4. **Hazard Detection** (`hazard-map.html`)
   - Frontend sends CLAHE-enhanced image to `/api/hazard-map`
   - Server checks cache for existing hazard map
   - If not cached:
     - Image tiled into 512×512 patches with 100px overlap
     - Each tile processed by AI model (U-Net) for crater detection
     - Morphological analysis for additional hazards (boulders, steep slopes)
     - Results merged and stitched back together
     - Continuous risk map generated using distance transform
     - Results cached in memory
   - Color-coded hazard overlay returned to frontend

5. **Analysis Suite** (`analysis.html`)
   - Navigation hub for advanced analysis tools
   - Four main analysis modules available

6. **Landing Sites** (`landing-zones.html`)
   - Frontend requests landing site analysis via `/api/landing-sites`
   - Server reuses cached hazard map (or computes if needed)
   - Terrain masks computed (crater, boulder, steep slopes)
   - 5-factor suitability analysis:
     - Flatness (inverse gradient magnitude)
     - Smoothness (local variance)
     - Hazard-free score (from risk map)
     - Hard constraints (crater/boulder/steep avoidance)
   - Top 5 landing sites identified with safety scores
   - Results displayed with threat classification

7. **Route Planning** (`route-analysis.html`)
   - User drops start and goal pins on map
   - Frontend sends coordinates to `/api/route`
   - Server runs risk-aware A* pathfinding:
     - Cost function: `distance + risk_weight × risk_map_value`
     - Three strategies generated:
       - Distance Priority (low risk weight)
       - Balanced (medium risk weight)
       - Max Safety (high risk weight)
   - Routes returned with distance, safety scores, and path coordinates

8. **3D Terrain** (`science-map.html`)
   - Frontend requests 3D terrain data via `/api/terrain3d`
   - Server generates pseudo-elevation map from grayscale values
   - Downsamples to 150×150 for browser performance
   - Projects computed routes onto 3D surface
   - Returns Plotly.js-compatible data structure

### TECHNICAL IMPLEMENTATION DETAILS

#### AI Model Integration
- **Model**: U-Net architecture for semantic segmentation
- **Format**: ONNX for cross-platform compatibility
- **Inference**: CPU-based using onnxruntime
- **Tiling Strategy**: 512×512 tiles with 100px overlap for memory efficiency
- **Parallel Processing**: 4-worker ProcessPoolExecutor
- **Post-processing**: Morphological dilation and multi-scale analysis

#### Computer Vision Algorithms
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization in LAB color space
- **SIFT**: Scale-Invariant Feature Transform for robust keypoint detection
- **RANSAC**: Random Sample Consensus for geometric verification
- **A* Pathfinding**: Modified with risk-aware cost function
- **Distance Transform**: For continuous risk mapping from binary hazards
- **Morphological Operations**: Black-hat, top-hat, connected components analysis

#### Performance Optimizations
- **Caching**: MD5-based image hash caching for expensive computations
- **Parallel Processing**: Multi-core CPU utilization for AI inference
- **Lazy Computation**: Terrain masks computed only when needed
- **Memory Management**: Temporary files cleaned up after processing
- **Progressive Loading**: Frontend shows progress during long operations

#### Quality Assurance
- **Multi-scale Analysis**: Multiple crater detection scales for robustness
- **Circularity Filtering**: Shape-based validation for crater detection
- **Spatial Coverage**: Ensures features are well-distributed across image
- **Reprojection Error**: Geometric accuracy measurement
- **Confidence Scoring**: Evidence-based quality assessment

### KEY ALGORITHMS EXPLAINED

#### Hazard Detection Pipeline
1. **Image Tiling**: Large images split into 512×512 overlapping tiles
2. **AI Inference**: Each tile processed by U-Net model for crater segmentation
3. **Morphological Enhancement**: Multi-scale black-hat operations for additional crater detection
4. **Feature Integration**: AI and morphological results combined
5. **Boulder Detection**: Top-hat filtering for small rock identification
6. **Slope Analysis**: Sobel gradient magnitude for steep slope detection
7. **Risk Mapping**: Distance transform converts binary hazards to continuous risk
8. **Result Merging**: Tiles stitched together with overlap averaging

#### Image Registration Pipeline
1. **Preprocessing**: Normalization to [0,1] range + CLAHE enhancement
2. **Feature Extraction**: SIFT keypoints + terrain landmarks (craters, ridges)
3. **Descriptor Computation**: 128-dimensional SIFT descriptors
4. **Feature Matching**: Ratio test (0.75) + mutual consistency filtering
5. **Geometric Verification**: RANSAC with adaptive iteration count
6. **Model Selection**: Auto-selection (translation → similarity → affine → homography)
7. **Correspondence Refinement**: Subpixel cross-correlation refinement
8. **Image Warping**: Projective transformation using estimated matrix
9. **Quality Evaluation**: RMSE, spatial coverage, confidence scoring

#### Landing Site Selection
1. **Terrain Analysis**: Compute crater, boulder, and steep slope masks
2. **Feature Scoring**:
   - Flatness: Inverse of gradient magnitude
   - Smoothness: Inverse of local variance
   - Safety: Complement of risk map
3. **Suitability Combination**: Weighted sum (40% flatness + 30% smoothness + 30% safety)
4. **Hard Constraints**: Exclude areas near hazards using morphological dilation
5. **Site Selection**: Iterative local maxima discovery with spacing constraints
6. **Threat Classification**: Primary threat identification per site

#### Risk-Aware Pathfinding
1. **Cost Function**: `cost = distance + risk_weight × risk_map_value`
2. **Heuristic**: Euclidean distance to goal
3. **Strategies**:
   - Distance Priority: risk_weight = 0.5
   - Balanced: risk_weight = 5.0
   - Max Safety: risk_weight = 20.0
4. **8-Connected Grid**: Allows diagonal movement
5. **Safety Scoring**: Based on average risk along path

### DEPENDENCIES AND TECHNOLOGIES

**Backend Technologies**:
- Python 3.x
- Flask (web framework)
- OpenCV (computer vision)
- NumPy (numerical computing)
- ONNX Runtime (AI model inference)
- Matplotlib (visualization)

**Frontend Technologies**:
- HTML5/CSS3
- Vanilla JavaScript
- Plotly.js (3D visualization)
- IndexedDB (client-side storage)

**AI/ML**:
- U-Net architecture (semantic segmentation)
- SIFT (feature extraction)
- RANSAC (geometric verification)
- A* algorithm (pathfinding)

### DEPLOYMENT CONSIDERATIONS

**System Requirements**:
- Minimum 4GB RAM
- Multi-core CPU recommended
- Python 3.8+
- ~500MB disk space for models

**Performance Characteristics**:
- Image enhancement: <1 second
- Image registration: 5-30 seconds (depending on image size)
- Hazard detection: 30 seconds - 2 minutes (depending on resolution)
- Landing site analysis: <5 seconds (cached hazard map)
- Route planning: <3 seconds (cached hazard map)

**Scalability**:
- Single-user design with in-memory caching
- Suitable for research and demonstration
- Can be scaled with distributed caching for multi-user scenarios

### JUDGES' FAQ POTENTIAL QUESTIONS

**Q: How does the system handle different lighting conditions?**
A: CLAHE enhancement normalizes contrast, and the multi-scale morphological analysis is robust to lighting variations. The system also includes dark shadow detection specifically for low-light areas.

**Q: What makes the hazard detection accurate?**
A: Combined AI + morphological approach: U-Net model learns crater patterns, while classical computer vision provides geometric validation. Multi-scale analysis ensures detection of craters of different sizes.

**Q: How is image registration accuracy ensured?**
A: Multi-stage verification: ratio test filters poor matches, RANSAC handles outliers, subpixel refinement improves accuracy, and comprehensive metrics (RMSE, spatial coverage) quantify quality.

**Q: Can the system handle very large images?**
A: Yes, through tiled processing (512×512 tiles with overlap) and parallel execution using ProcessPoolExecutor with 4 workers.

**Q: How are landing sites evaluated for safety?**
A: 5-factor analysis: flatness (gradient), smoothness (variance), hazard-free score (risk map), plus hard constraints to avoid craters, boulders, and steep slopes.

**Q: What makes the route planning "risk-aware"?**
A: Modified A* algorithm where cost function includes both distance and risk map values, with different risk weights for safety vs. distance optimization strategies.

**Q: How does the caching system work?**
A: MD5 hash of the baseline image is used as cache key. Expensive AI computations are cached and reused for subsequent requests with the same image, significantly improving performance.

**Q: What's the advantage of using ONNX for the AI model?**
A: Cross-platform compatibility, optimized CPU inference, no framework dependencies, and consistent behavior across different deployment environments.

**Q: How does the system handle partial or noisy data?**
A: RANSAC is inherently robust to outliers, mutual consistency filtering removes ambiguous matches, and the multi-scale approach provides redundancy in detection.

**Q: Can the system be extended for other planetary surfaces?**
A: Yes, the modular architecture allows swapping the AI model and adjusting morphological parameters for different surface characteristics (Mars, asteroids, etc.).
