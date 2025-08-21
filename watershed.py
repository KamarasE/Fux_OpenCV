#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wet coverage on a wire (no resizing, path set via variable).

- Assumes the wire is the darkest large object in the image.
- Detects droplet seeds from specular (low S / high V) + high gradient.
- Uses watershed constrained to the wire to grow droplet regions.
- Saves an overlay, the wire mask, and the droplet mask.
"""

import os
from pathlib import Path
import cv2
import numpy as np

# >>> Edit these <<<
IMAGE_PATH = "/home/erik/Desktop/Fux/image1.jpg"
OUT_BASE   = "/home/erik/Desktop/Fux/out/wire1"  # files: *_overlay.jpg, *_wire_mask.png, *_droplets.png


def detect_coverage_noresize(
    bgr,
    # wire mask shaping
    wire_morph_close=11,    # close gaps along helical grooves
    wire_morph_open=5,      # remove small dark specks off the wire
    wire_erode=2,           # shave a tiny overspill
    # droplet seed thresholds
    spec_s_low_q=0.30,
    spec_v_high_q=0.90,
    gradient_quantile=0.92,
    min_component_px=12,
    # watershed shaping
    surebg_dilate=7,
    surefg_erode=1,
    final_dilate=2          # expand post-watershed to approximate droplet footprint
):
    if bgr is None:
        raise ValueError("Input image is None. Check the path.")

    # -------- 1) Wire mask via "darkest large object" --------
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    V_blur = cv2.GaussianBlur(V, (7, 7), 0)

    # Invert so the wire (dark) becomes bright, then Otsu
    Vin = 255 - V_blur
    _, th = cv2.threshold(Vin, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Largest connected component -> wire
    num, labels = cv2.connectedComponents(th, connectivity=8)
    areas = [(labels == i).sum() for i in range(1, num)]
    wire_mask = np.zeros_like(th, dtype=np.uint8)
    if areas:
        k = 1 + int(np.argmax(areas))
        wire_mask[labels == k] = 255

    # Morphology to fill grooves and smooth boundary
    if wire_morph_close > 0:
        kclose = cv2.getStructuringElement(cv2.MORPH_RECT, (wire_morph_close, wire_morph_close // 3))
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_CLOSE, kclose, iterations=1)
    if wire_morph_open > 0:
        kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wire_morph_open, wire_morph_open))
        wire_mask = cv2.morphologyEx(wire_mask, cv2.MORPH_OPEN, kopen, iterations=1)
    if wire_erode > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * wire_erode + 1, ) * 2)
        wire_mask = cv2.erode(wire_mask, ker, iterations=1)

    wire_mask = (wire_mask > 0).astype(np.uint8)
    if wire_mask.sum() == 0:
        raise RuntimeError("Wire mask is empty. Try adjusting wire_* parameters.")

    # -------- 2) Droplet seed candidates (specular + gradient) --------
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    S = hsv[:, :, 1].astype(np.float32) / 255.0
    Vf = hsv[:, :, 2].astype(np.float32) / 255.0

    wp = wire_mask == 1
    s_low = float(np.quantile(S[wp], spec_s_low_q))
    v_high = float(np.quantile(Vf[wp], spec_v_high_q))

    specular = ((S <= s_low) & (Vf >= v_high)).astype(np.uint8)

    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    gthr = float(np.quantile(mag[wp], gradient_quantile))
    gradmask = (mag >= gthr).astype(np.uint8)

    candidates = ((specular | gradmask) & wire_mask).astype(np.uint8)

    # Remove tiny components
    num, labels = cv2.connectedComponents(candidates, connectivity=8)
    clean = np.zeros_like(candidates, dtype=np.uint8)
    for i in range(1, num):
        if (labels == i).sum() >= min_component_px:
            clean[labels == i] = 1

    # -------- 3) Watershed (constrained to the wire) --------
    kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * surefg_erode + 1, ) * 2)
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * surebg_dilate + 1, ) * 2)
    sure_fg = cv2.erode(clean, kernel_fg, iterations=1)
    sure_bg = cv2.dilate((wire_mask & (~clean)).astype(np.uint8), kernel_bg, iterations=1)

    markers = np.zeros_like(gray, dtype=np.int32)
    _, lbl_fg = cv2.connectedComponents(sure_fg.astype(np.uint8))
    markers[sure_bg == 1] = 1                       # background label
    markers[sure_fg == 1] = (lbl_fg[sure_fg == 1] + 1)  # droplets > 1

    # Watershed over gradient image to favor boundaries on edges
    grad_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_rgb = cv2.cvtColor(grad_u8, cv2.COLOR_GRAY2BGR)
    cv2.watershed(grad_rgb, markers)
    droplet_ws = ((markers > 1) & (wire_mask == 1)).astype(np.uint8)

    # Optional expansion to cover full footprint
    if final_dilate > 0:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * final_dilate + 1, ) * 2)
        droplet_ws = cv2.dilate(droplet_ws, kd, iterations=1)
        droplet_ws = (droplet_ws & wire_mask).astype(np.uint8)

    # -------- 4) Coverage & visualization --------
    wire_area = int(wire_mask.sum())
    wet_area = int(droplet_ws.sum())
    coverage = 100.0 * wet_area / max(1, wire_area)

    overlay = bgr.copy()
    # Darken wire region a bit
    overlay[wire_mask == 1] = (0.6 * overlay[wire_mask == 1] + 0.4 * np.array([40, 40, 40])).astype(np.uint8)
    # Paint droplets red
    overlay[droplet_ws == 1] = np.array([0, 0, 255], dtype=np.uint8)
    vis = overlay.copy()
    cv2.putText(vis, f"Coverage: {coverage:.1f}%", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    return coverage, wire_mask, droplet_ws, vis


def main():
    # Load from your variable path
    bgr = cv2.imread(IMAGE_PATH)
    if bgr is None:
        raise SystemExit(f"Could not read image: {IMAGE_PATH}")

    coverage, wire_mask, droplets, vis = detect_coverage_noresize(bgr)

    # Save outputs to your chosen base path
    Path(os.path.dirname(OUT_BASE)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{OUT_BASE}_overlay.jpg", vis)
    cv2.imwrite(f"{OUT_BASE}_wire_mask.png", (wire_mask * 255).astype(np.uint8))
    cv2.imwrite(f"{OUT_BASE}_droplets.png", (droplets * 255).astype(np.uint8))

    print(f"Estimated wet coverage: {coverage:.2f}%")
    print(f"Saved:\n  {OUT_BASE}_overlay.jpg\n  {OUT_BASE}_wire_mask.png\n  {OUT_BASE}_droplets.png")


if __name__ == "__main__":
    main()
