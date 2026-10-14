"""
Utility functions for Handwritten Digit Recognition project.
Includes:
- Robust image preprocessing matching the official MNIST specification:
  1. RGBA alpha channel compositing
  2. Polarity inversion (dark ink to white on black)
  3. Bounding box cropping & aspect-ratio preserving resize into 20x20 box
  4. Adaptive stroke dilation/thickening to match MNIST stroke widths
  5. Center of Mass (COM) translation to (13.5, 13.5)
- Top-3 prediction extraction
- Confidence threshold evaluation
- Preprocessing diagnostics metadata
"""

import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, maximum_filter


def preprocess_image(image_input):
    """
    Preprocess an arbitrary input image (PIL Image, numpy array, or file path)
    into a standardized 28x28 normalized grayscale image matching the official
    MNIST dataset generation process (LeCun et al.).
    
    Steps:
    1. Composite RGBA transparency over a solid white canvas.
    2. Convert to single-channel luminosity (grayscale).
    3. Detect background polarity: MNIST has white digits on black background.
       If input has light background with dark ink, invert it.
    4. Detect digit foreground bounding box via thresholding.
    5. Resize digit preserving aspect ratio so its longest dimension is 20 pixels.
    6. Place the 20x20 digit inside a 28x28 black canvas.
    7. Adaptive stroke dilation: If stroke is too thin compared to MNIST, dilate slightly.
    8. Center of Mass (COM) alignment: Translate digit so its center of mass
       is located at (13.5, 13.5), matching MNIST training distribution.
    9. Normalize pixel values from [0, 255] to [0.0, 1.0].
    
    Returns:
        processed_array: np.ndarray of shape (28, 28), float32 in [0, 1]
        display_image: PIL Image of size 28x28 for visual inspection
        is_blank: bool indicating if image had no detectable digit
        diagnostics: dict containing metadata about the image and preprocessing steps
    """
    diagnostics = {
        "original_mode": None,
        "original_size": None,
        "polarity_detected": None,
        "inverted": False,
        "bounding_box": None,
        "digit_width": 0,
        "digit_height": 0,
        "aspect_ratio": 1.0,
        "foreground_detected": False,
        "center_of_mass": (13.5, 13.5),
        "com_shift": (0, 0)
    }

    # 1. Load and handle RGBA alpha channels
    if isinstance(image_input, str):
        img_raw = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img_raw = image_input
    else:
        img_raw = Image.fromarray(np.uint8(image_input))

    diagnostics["original_mode"] = img_raw.mode
    diagnostics["original_size"] = img_raw.size

    # Composite transparent PNGs over solid white
    if img_raw.mode in ("RGBA", "LA") or (img_raw.mode == "P" and "transparency" in img_raw.info):
        alpha_img = img_raw.convert("RGBA")
        white_bg = Image.new("RGBA", alpha_img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(white_bg, alpha_img)
        img = composite.convert("L")
    else:
        img = img_raw.convert("L")

    arr = np.array(img, dtype=np.float32)

    # 2. Polarity Detection
    border_pixels = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    mean_border = float(np.mean(border_pixels))

    # Fast-path: Image is already 28x28 with dark borders (standard MNIST format)
    if img.size == (28, 28) and mean_border < 35:
        normalized = arr / 255.0
        display_pil = Image.fromarray(np.uint8(arr))
        diagnostics["polarity_detected"] = "MNIST Standard (28x28 Native)"
        diagnostics["inverted"] = False
        diagnostics["foreground_detected"] = True
        diagnostics["bounding_box"] = [0, 0, 28, 28]
        diagnostics["digit_width"] = 28
        diagnostics["digit_height"] = 28
        return normalized, display_pil, False, diagnostics

    if mean_border > 127:
        # Light background, dark ink -> Invert to white on black
        arr = 255.0 - arr
        diagnostics["polarity_detected"] = "Light Background (Dark Ink)"
        diagnostics["inverted"] = True
    else:
        diagnostics["polarity_detected"] = "Dark Background (White Ink - MNIST Native)"
        diagnostics["inverted"] = False

    # Clean low-level background noise
    min_val = np.percentile(arr, 5)
    arr = np.clip(arr - min_val, 0, 255)
    max_val = np.max(arr)
    if max_val > 0:
        arr = (arr / max_val) * 255.0

    # 3. Detect foreground bounding box
    foreground_mask = arr > 35
    if not np.any(foreground_mask) or max_val < 30:
        blank_canvas = np.zeros((28, 28), dtype=np.float32)
        blank_pil = Image.fromarray(np.uint8(blank_canvas * 255))
        diagnostics["foreground_detected"] = False
        return blank_canvas, blank_pil, True, diagnostics

    diagnostics["foreground_detected"] = True
    rows = np.any(foreground_mask, axis=1)
    cols = np.any(foreground_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop_h = int(rmax - rmin + 1)
    crop_w = int(cmax - cmin + 1)
    diagnostics["bounding_box"] = [int(cmin), int(rmin), int(cmax), int(rmax)]
    diagnostics["digit_width"] = crop_w
    diagnostics["digit_height"] = crop_h
    diagnostics["aspect_ratio"] = round(crop_w / max(1, crop_h), 3)

    cropped = arr[rmin:rmax + 1, cmin:cmax + 1]

    # 4. Resize preserving aspect ratio into a 20x20 box (MNIST specification)
    crop_pil = Image.fromarray(np.uint8(cropped))
    if crop_h > crop_w:
        new_h = 20
        new_w = max(1, int(round((crop_w / crop_h) * 20)))
    else:
        new_w = 20
        new_h = max(1, int(round((crop_h / crop_w) * 20)))

    resized_digit = crop_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_arr = np.array(resized_digit, dtype=np.float32)

    # 5. Place in 28x28 canvas initially centered by geometry
    canvas = np.zeros((28, 28), dtype=np.float32)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

    # 6. Adaptive Stroke Dilation (ensures faint/thin pen strokes match MNIST brush width)
    active_pixel_count = np.sum(canvas > 50)
    if active_pixel_count < 75 and active_pixel_count > 5:
        canvas = maximum_filter(canvas, size=2)

    # 7. Center of Mass (COM) Alignment (Official MNIST Standard)
    total_mass = np.sum(canvas)
    if total_mass > 0:
        cy, cx = center_of_mass(canvas)
        diagnostics["center_of_mass"] = (round(float(cy), 2), round(float(cx), 2))
        
        # Shift so Center of Mass is positioned at exactly (13.5, 13.5)
        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))
        diagnostics["com_shift"] = (int(shift_y), int(shift_x))

        if shift_y != 0 or shift_x != 0:
            shifted_canvas = np.zeros_like(canvas)
            for r in range(28):
                for c in range(28):
                    orig_r = r - shift_y
                    orig_c = c - shift_x
                    if 0 <= orig_r < 28 and 0 <= orig_c < 28:
                        shifted_canvas[r, c] = canvas[orig_r, orig_c]
            canvas = shifted_canvas

    # 8. Normalize pixel values to [0.0, 1.0]
    normalized_array = canvas / 255.0
    display_pil = Image.fromarray(np.uint8(np.clip(canvas, 0, 255)))

    return normalized_array, display_pil, False, diagnostics


def get_top_k_predictions(probabilities, k=3):
    """
    Extract top-k predictions with digit classes and probability percentages.
    """
    prob_array = np.array(probabilities).flatten()
    top_indices = np.argsort(prob_array)[::-1][:k]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        score = float(prob_array[idx])
        results.append({
            "rank": rank,
            "digit": int(idx),
            "score": score,
            "percentage": f"{score * 100:.2f}%",
            "percent_val": score * 100
        })
    return results


def check_confidence(top_score, threshold=0.70):
    """
    Evaluate if the top prediction meets the configurable confidence threshold.
    """
    score_pct = f"{top_score * 100:.2f}%"
    thresh_pct = f"{threshold * 100:.1f}%"
    
    if top_score >= threshold:
        return {
            "passed": True,
            "score_pct": score_pct,
            "threshold_pct": thresh_pct,
            "message": f"High confidence ({score_pct} >= {thresh_pct}). Passed selected threshold.",
            "status_type": "success"
        }
    else:
        return {
            "passed": False,
            "score_pct": score_pct,
            "threshold_pct": thresh_pct,
            "message": f"Low-confidence prediction ({score_pct} < {thresh_pct}). The model is uncertain about this digit. Please try another clearer image.",
            "status_type": "warning"
        }
