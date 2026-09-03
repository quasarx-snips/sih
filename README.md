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
