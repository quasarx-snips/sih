import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor


TILE_SIZE = 512
OVERLAP = 100

# Original parameters
CRATER_SCALES         = [0.012, 0.022, 0.035, 0.05]
CRATER_PERCENTILE     = 96
CRATER_MIN_AREA_FRAC  = 0.008 ** 2
CRATER_CIRCULARITY    = 0.35
DARK_SHADOW_THRESH    = 45
DARK_MIN_AREA_FRAC    = 0.02 ** 2
DARK_CIRCULARITY      = 0.05
STEEP_PERCENTILE      = 93
MIN_STEEP_AREA        = 20
BOULDER_MIN_AREA      = 2
BOULDER_MAX_AREA      = 35
ALPHA                 = 1
CRATER_DANGER_RADIUS  = 0.05
HAZARD_DANGER_RADIUS  = 0.02


# ---------- Worker function (unchanged logic, processes ONE tile) ----------
def process_tile(args):
    tile_img, y1, y2, x1, x2 = args

    gray_u8 = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
    H, W = gray_u8.shape
    diag = float(np.sqrt(H**2 + W**2))
    den = cv2.medianBlur(gray_u8, 3)

    # Craters
    crater_radii = [max(3, int(diag * f)) for f in CRATER_SCALES]
    blackhat_stack = np.zeros((H, W), dtype=np.float32)
    for r in crater_radii:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        bh = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, k).astype(np.float32)
        blackhat_stack = np.maximum(blackhat_stack, bh)

    thr = np.percentile(blackhat_stack, CRATER_PERCENTILE)
    crater_bin = (blackhat_stack > thr).astype(np.uint8)
    crater_bin = cv2.morphologyEx(crater_bin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crater_bin, connectivity=8)
    crater_mask = np.zeros((H, W), dtype=bool)
    min_area = CRATER_MIN_AREA_FRAC * diag**2
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim**2)
        if circularity > CRATER_CIRCULARITY:
            crater_mask |= (labels == i)

    # Dark shadows
    _, dark_bin = cv2.threshold(den, DARK_SHADOW_THRESH, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(dark_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_min_area = DARK_MIN_AREA_FRAC * diag**2
    for c in contours:
        area = cv2.contourArea(c)
        if area < dark_min_area:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim**2)
        if circularity > DARK_CIRCULARITY:
            tmp = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(tmp, [c], -1, 1, -1)
            crater_mask |= tmp.astype(bool)

    # Boulders
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, k_small)
    boulder_bin = (tophat > np.percentile(tophat, 99.2)).astype(np.uint8)
    n_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(boulder_bin, connectivity=8)
    boulder_mask = np.zeros((H, W), dtype=bool)
    for i in range(1, n_labels_b):
        area = stats_b[i, cv2.CC_STAT_AREA]
        if BOULDER_MIN_AREA <= area <= BOULDER_MAX_AREA:
            boulder_mask |= (labels_b == i)
    boulder_mask &= ~crater_mask

    # Steep slopes
    gray = gray_u8.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5)
    gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx**2 + gy**2)
    steep_raw = grad_mag > np.percentile(grad_mag, STEEP_PERCENTILE)
    steep_clean = cv2.morphologyEx(steep_raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(steep_clean, connectivity=8)
    steep_mask = np.zeros((H, W), dtype=bool)
    for i in range(1, n_labels2):
        if stats2[i, cv2.CC_STAT_AREA] >= MIN_STEEP_AREA:
            steep_mask |= (labels2 == i)
    steep_mask &= ~crater_mask
    hazard_mask = steep_mask | boulder_mask

    # Continuous risk
    crater_buffer_px = diag * CRATER_DANGER_RADIUS
    hazard_buffer_px = diag * HAZARD_DANGER_RADIUS

    bg_crater = (~crater_mask).astype(np.uint8) * 255
    dist_crater = cv2.distanceTransform(bg_crater, cv2.DIST_L2, 5)
    bg_hazard = (~hazard_mask).astype(np.uint8) * 255
    dist_hazard = cv2.distanceTransform(bg_hazard, cv2.DIST_L2, 5)

    risk_crater = np.clip(1.0 - (dist_crater / crater_buffer_px), 0.0, 1.0)
    risk_hazard = np.clip(1.0 - (dist_hazard / hazard_buffer_px), 0.0, 1.0)
    total_risk = np.maximum(risk_crater, risk_hazard)

    return y1, y2, x1, x2, total_risk


def generate_hazard_overlay(image_path: str, output_dir: str = "Outlay", show: bool = True) -> str:
    """
    Run the tiled crater/boulder/steep-slope hazard analysis on `image_path`
    and save the resulting overlay image into `output_dir`.

    Folder behavior:
      - If `output_dir` already exists, the new overlay is added alongside
        whatever is already in there (nothing existing is touched/deleted).
      - If `output_dir` does not exist, it is created.

    Returns the full path to the saved overlay image.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Could not find image at: {image_path}")

    raw = cv2.imread(image_path)
    if raw is None:
        raise ValueError(f"OpenCV could not read image at: {image_path}")

    h, w = raw.shape[:2]
    final_risk_map = np.zeros((h, w), dtype=np.float32)

    tasks = []
    for y in range(0, h, TILE_SIZE - OVERLAP):
        for x in range(0, w, TILE_SIZE - OVERLAP):
            y2 = min(y + TILE_SIZE, h)
            x2 = min(x + TILE_SIZE, w)
            tile = raw[y:y2, x:x2]
            tasks.append((tile, y, y2, x, x2))

    print(f"Starting parallel processing on {len(tasks)} tiles...")
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_tile, tasks)

    print("Stitching map...")
    for y1, y2, x1, x2, risk_tile in results:
        final_risk_map[y1:y2, x1:x2] = np.maximum(final_risk_map[y1:y2, x1:x2], risk_tile)

    gray_u8_full = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    colormap = plt.get_cmap('RdYlGn_r')
    risk_color = colormap(final_risk_map)[..., :3]
    gray_rgb = np.stack([gray_u8_full, gray_u8_full, gray_u8_full], axis=-1)
    overlay = np.clip(gray_rgb * (1 - ALPHA) + risk_color * ALPHA, 0, 1)

    # --- Folder handling: create "Outlay" if missing, otherwise reuse it ---
    if os.path.isdir(output_dir):
        print(f"Folder '{output_dir}' already exists -- new image will be added to it.")
    else:
        os.makedirs(output_dir)
        print(f"Folder '{output_dir}' created.")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_hazard_overlay.png")

    # Don't overwrite a previous result for the same input image
    counter = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{base_name}_hazard_overlay_{counter}.png")
        counter += 1

    overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, overlay_bgr)
    print(f"Saved hazard overlay to: {out_path}")

    if show:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Original High-Res Image")
        ax[0].axis('off')
        ax[1].imshow(overlay)
        ax[1].set_title("Continuous Hazard Map")
        ax[1].axis('off')
        sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    return out_path


if __name__ == "__main__":
    user_image_path = r"C:\Users\bijan\Desktop\DevJams\Module1_2_Integrated\inputs\test.png"
    generate_hazard_overlay(user_image_path, show=False)