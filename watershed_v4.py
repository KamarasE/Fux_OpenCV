#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wet coverage on a wire (batch mode).

Pipeline:
1. Load images from INPUT_DIR.
2. Detect the wire (assumed to be darkest large object).
3. Find droplet candidates (bright specular + high gradient).
4. Refine droplet segmentation with watershed, constrained to the wire.
5. Compute coverage (% of wire pixels covered by droplets).
6. Categorize coverage into WC4–WC7.
7. Save overlay (wire + droplet outlines + labels) and masks.
"""

import os
from pathlib import Path
import cv2
import numpy as np

# Input folder with images, and output folder for results
INPUT_DIR = "/home/erik/Desktop/Fux/Fux_OpenCV/kepek/Csapvíz/Festett/20250725_112100 A"
OUTPUT_DIR = "/home/erik/Desktop/Fux/Fux_OpenCV/out/v4"


def categorize_wc(coverage_pct: float) -> str:
    """
    Convert numeric coverage (%) into WC category.
    WC4: < 10%
    WC5: 10–50%
    WC6: 90–99%
    WC7: 99–100%
    Anything else (50–90%) = WCx (currently undefined).
    """
    c = max(0.0, min(100.0, coverage_pct))  # clamp into [0,100]
    if c < 10.0:
        return "WC4"
    if 10.0 <= c <= 50.0:
        return "WC5"
    if 90.0 <= c < 99.0:
        return "WC6"
    if c >= 99.0:
        return "WC7"
    return "WCx"  # placeholder for undefined band


def detect_coverage_noresize(
    bgr,
    wire_morph_close=11,   # close gaps (helical grooves)
    wire_morph_open=5,     # remove specks
    wire_erode=2,          # shrink slightly
    spec_s_low_q=0.30,     # specular: low saturation threshold
    spec_v_high_q=0.90,    # specular: high brightness threshold
    gradient_quantile=0.92,# droplet edges: top quantile of gradients
    min_component_px=12,   # ignore tiny noise blobs
    surebg_dilate=7,       # expand sure background for watershed
    surefg_erode=1,        # shrink sure foreground for watershed
    final_dilate=2         # expand droplet result to full footprint
):
    # --- 1) Wire segmentation ---
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    V_blur = cv2.GaussianBlur(V, (7, 7), 0)

    # Invert brightness so dark wire appears bright, then Otsu threshold
    Vin = 255 - V_blur
    _, th = cv2.threshold(Vin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick largest connected component as wire
    num, labels = cv2.connectedComponents(th, connectivity=8)
    areas = [(labels == i).sum() for i in range(1, num)]
    wire_mask = np.zeros_like(th, dtype=np.uint8)
    if areas:
        k = 1 + int(np.argmax(areas))
        wire_mask[labels == k] = 255

    # Morphological smoothing/refinement
    if wire_morph_close > 0:
        kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (wire_morph_close, wire_morph_close // 3))
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, kclose)
    if wire_morph_open > 0:
        kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wire_morph_open, wire_morph_open))
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, kopen)
    if wire_erode > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wire_erode + 1,) * 2)
        wire_mask = cv2.erode(wire_mask, ker)

    wire_mask = (wire_mask > 0).astype(np.uint8)
    if wire_mask.sum() == 0:
        raise RuntimeError("Wire mask is empty.")

    # --- 2) Droplet candidate seeds ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Normalize HSV values
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    Vf = hsv[:, :, 2].astype(np.float32) / 255.0
    wp = wire_mask == 1  # wire pixels only

    # Specular highlights = low saturation & high brightness
    s_low = float(np.quantile(S[wp], spec_s_low_q))
    v_high = float(np.quantile(Vf[wp], spec_v_high_q))
    specular = ((S <= s_low) & (Vf >= v_high)).astype(np.uint8)

    # Gradient-based edges
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    gthr = float(np.quantile(mag[wp], gradient_quantile))
    gradmask = (mag >= gthr).astype(np.uint8)

    # Combine specular + gradient inside wire
    candidates = ((specular | gradmask) & wire_mask).astype(np.uint8)

    # Remove tiny blobs
    num, labels = cv2.connectedComponents(candidates, connectivity=8)
    clean = np.zeros_like(candidates, dtype=np.uint8)
    for i in range(1, num):
        if (labels == i).sum() >= min_component_px:
            clean[labels == i] = 1

    # --- 3) Watershed segmentation ---
    # Build sure foreground/background from candidates
    kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * surefg_erode + 1,) * 2)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * surebg_dilate + 1,) * 2)
    sure_fg = cv2.erode(clean, kernel_fg)
    sure_bg = cv2.dilate((wire_mask & (~clean)).astype(np.uint8), kernel_bg)

    # Label markers: 1 = background, 2+ = droplets
    markers = np.zeros_like(gray, dtype=np.int32)
    _, lbl_fg = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers[sure_bg == 1] = 1
    markers[sure_fg == 1] = (lbl_fg[sure_fg == 1] + 1)

    # Run watershed on gradient image
    grad_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_rgb = cv2.cvtColor(grad_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(grad_rgb, markers)
    droplet_ws = ((markers > 1) & (wire_mask == 1)).astype(np.uint8)

    # Expand result slightly (approximate droplet footprint)
    if final_dilate > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * final_dilate + 1,) * 2)
        droplet_ws = cv2.dilate(droplet_ws, kd)
        droplet_ws = (droplet_ws & wire_mask).astype(np.uint8)

    # --- 4) Coverage calculation ---
    wire_area = int(wire_mask.sum())
    wet_area = int(droplet_ws.sum())
    coverage = 100.0 * wet_area / max(1, wire_area)

    # --- 5) Visualization ---
    overlay = bgr.copy()
    # Darken the wire region for contrast
    overlay[wire_mask == 1] = (
        0.6 * overlay[wire_mask == 1] + 0.4 * np.array([40, 40, 40])
    ).astype(np.uint8)
    # Draw droplet contours in red
    contours, _ = cv2.findContours(droplet_ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=2)

    # Add text labels (top-left)
    vis = overlay.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, color = 1.0, 2, (255, 255, 255)

    wc_tag = categorize_wc(coverage)
    cv2.putText(vis, f"Coverage: {coverage:.1f}%", (20, 40), font, scale, color, thick, cv2.LINE_AA)
    cv2.putText(vis, wc_tag, (20, 80), font, scale, color, thick, cv2.LINE_AA)

    return coverage, wc_tag, wire_mask, droplet_ws, vis


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in sorted(in_dir.glob("image*.jpg")):
        print(f"Processing {fname.name}...")
        bgr = cv2.imread(str(fname))
        if bgr is None:
            print("  Skipped (could not read).")
            continue

        try:
            coverage, wc_tag, wire_mask, droplets, vis = detect_coverage_noresize(bgr)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        # Save outputs
        base = out_dir / fname.stem
        cv2.imwrite(f"{base}_overlay.jpg", vis)
        cv2.imwrite(f"{base}_wire_mask.png", (wire_mask * 255).astype(np.uint8))
        cv2.imwrite(f"{base}_droplets.png", (droplets * 255).astype(np.uint8))
        print(f"  Coverage: {coverage:.2f}%  ->  {wc_tag}  (saved {base}_*.jpg/png)")


if __name__ == "__main__":
    main()
