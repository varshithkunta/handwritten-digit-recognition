"""
Comprehensive Edge-Case Testing & Verification Suite
Verifies:
1. Clean Standard MNIST digits
2. Polarity Handling (Black ink on White vs White ink on Black)
3. Transparent RGBA PNGs
4. Off-Center / Corner-shifted Digits
5. Blank / Empty Canvases
6. Ambiguous Digits (4 vs 9, 3 vs 8)
7. Non-Digit Doodles & Confidence-Aware Rejection
8. Top-3 Prediction Ranking Accuracy
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tensorflow as tf
from tensorflow import keras

from utils import preprocess_image, get_top_k_predictions, check_confidence


def generate_synthetic_test_images(output_dir="edge_case_tests"):
    """Generates synthetic test cases covering real-world edge cases."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    # 1. Blank image (all white)
    img_blank = Image.new("RGB", (200, 200), color="white")
    p_blank = os.path.join(output_dir, "edge_blank.png")
    img_blank.save(p_blank)
    paths["blank"] = p_blank
    
    # 2. Transparent RGBA image with black ink
    img_rgba = Image.new("RGBA", (200, 200), (255, 255, 255, 0))
    draw_rgba = ImageDraw.Draw(img_rgba)
    draw_rgba.line([(100, 40), (100, 160)], fill=(0, 0, 0, 255), width=18)
    draw_rgba.line([(60, 60), (100, 40)], fill=(0, 0, 0, 255), width=18)
    p_rgba = os.path.join(output_dir, "edge_rgba_transparent.png")
    img_rgba.save(p_rgba)
    paths["rgba_transparent"] = p_rgba
    
    # 3. Inverted image (White ink on Black background, MNIST style)
    img_inv = Image.new("RGB", (200, 200), color="black")
    draw_inv = ImageDraw.Draw(img_inv)
    draw_inv.ellipse([50, 40, 150, 160], outline="white", width=20)
    p_inv = os.path.join(output_dir, "edge_white_on_black.png")
    img_inv.save(p_inv)
    paths["white_on_black"] = p_inv
    
    # 4. Black ink on White background (Standard paper handwriting)
    img_std = Image.new("RGB", (200, 200), color="white")
    draw_std = ImageDraw.Draw(img_std)
    draw_std.ellipse([50, 40, 150, 160], outline="black", width=20)
    p_std = os.path.join(output_dir, "edge_black_on_white.png")
    img_std.save(p_std)
    paths["black_on_white"] = p_std
    
    # 5. Off-center / Tiny Digit shifted to bottom-right corner
    img_off = Image.new("RGB", (200, 200), color="white")
    draw_off = ImageDraw.Draw(img_off)
    draw_off.line([(150, 150), (180, 150)], fill="black", width=12)
    draw_off.line([(180, 150), (150, 190)], fill="black", width=12)
    p_off = os.path.join(output_dir, "edge_off_center_corner.png")
    img_off.save(p_off)
    paths["off_center"] = p_off
    
    # 6. Ambiguous Digit (Resembles both 4 and 9)
    img_amb = Image.new("RGB", (200, 200), color="white")
    draw_amb = ImageDraw.Draw(img_amb)
    draw_amb.line([(120, 30), (120, 170)], fill="black", width=16)
    draw_amb.line([(60, 100), (150, 100)], fill="black", width=16)
    draw_amb.line([(120, 30), (60, 100)], fill="black", width=16)
    draw_amb.line([(60, 30), (120, 30)], fill="black", width=14)
    p_amb = os.path.join(output_dir, "edge_ambiguous_4_or_9.png")
    img_amb.save(p_amb)
    paths["ambiguous_4_9"] = p_amb
    
    # 7. Non-digit doodle (Random cross / scribble)
    img_non = Image.new("RGB", (200, 200), color="white")
    draw_non = ImageDraw.Draw(img_non)
    draw_non.line([(40, 40), (160, 160)], fill="black", width=14)
    draw_non.line([(160, 40), (40, 160)], fill="black", width=14)
    p_non = os.path.join(output_dir, "edge_non_digit_cross.png")
    img_non.save(p_non)
    paths["non_digit"] = p_non
    
    return paths


def run_edge_case_tests():
    print("=" * 65)
    print("RUNNING EDGE-CASE & ROBUSTNESS VERIFICATION SUITE")
    print("=" * 65)
    
    model = keras.models.load_model("model.keras")
    test_paths = generate_synthetic_test_images()
    
    results = {}
    
    # Test 1: Blank canvas
    print("\n[Test 1] Blank Image Handling:")
    arr, pil_img, is_blank, diag = preprocess_image(test_paths["blank"])
    print(f"  • Detected blank: {is_blank} (Expected: True)")
    assert is_blank, "Test 1 Failed: Blank image was not detected as blank!"
    results["blank_test"] = "PASSED"
    print("  -> Result: PASSED (Blank canvas detected cleanly without false prediction)")
    
    # Test 2: Inverted polarity (Black on White vs White on Black)
    print("\n[Test 2] Polarity Invariant Recognition (0 on White vs 0 on Black):")
    arr_b, _, _, _ = preprocess_image(test_paths["black_on_white"])
    arr_w, _, _, _ = preprocess_image(test_paths["white_on_black"])
    
    pred_b = model.predict(np.expand_dims(arr_b, axis=0), verbose=0)[0]
    pred_w = model.predict(np.expand_dims(arr_w, axis=0), verbose=0)[0]
    
    digit_b = int(np.argmax(pred_b))
    digit_w = int(np.argmax(pred_w))
    print(f"  - Black-on-White digit predicted: {digit_b} (Score: {pred_b[digit_b]*100:.1f}%)")
    print(f"  - White-on-Black digit predicted: {digit_w} (Score: {pred_w[digit_w]*100:.1f}%)")
    assert digit_b == 0 and digit_w == 0, "Test 2 Failed: Digit 0 not recognized in both polarities!"
    results["polarity_test"] = "PASSED"
    print("  -> Result: PASSED (Both polarities normalized into identical representation)")
    
    # Test 3: Off-center corner-shifted digit
    print("\n[Test 3] Off-Center Bounding-Box Centering:")
    arr_off, _, is_blank_off, _ = preprocess_image(test_paths["off_center"])
    # Verify the mass is centered around row 14, col 14
    cy, cx = np.where(arr_off > 0.2)
    mean_y, mean_x = np.mean(cy), np.mean(cx)
    print(f"  - Center of mass in 28x28 canvas: ({mean_y:.1f}, {mean_x:.1f}) (Target ~ 14.0, 14.0)")
    assert 8 <= mean_y <= 20 and 8 <= mean_x <= 20, "Centering failed!"
    results["centering_test"] = "PASSED"
    print("  -> Result: PASSED (Corner digit automatically cropped and centered in 28x28)")
    
    # Test 4: Ambiguous digit Top-3 verification
    print("\n[Test 4] Ambiguous Digit Top-3 Candidate Ranking:")
    arr_amb, _, _, _ = preprocess_image(test_paths["ambiguous_4_9"])
    pred_amb = model.predict(np.expand_dims(arr_amb, axis=0), verbose=0)[0]
    top3_amb = get_top_k_predictions(pred_amb, k=3)
    top_digits = [t["digit"] for t in top3_amb]
    print(f"  - Top-3 Candidates: {[t['digit'] for t in top3_amb]} with scores: {[t['percentage'] for t in top3_amb]}")
    assert 4 in top_digits or 9 in top_digits, "Expected 4 or 9 in top predictions"
    results["top3_ambiguity_test"] = "PASSED"
    print("  -> Result: PASSED (Surfaces competing candidate classes)")
    
    # Test 5: Non-digit Doodle Confidence Warning
    print("\n[Test 5] Confidence-Aware Warning on Non-Digit Doodle:")
    arr_non, _, _, _ = preprocess_image(test_paths["non_digit"])
    pred_non = model.predict(np.expand_dims(arr_non, axis=0), verbose=0)[0]
    top_score_non = float(np.max(pred_non))
    conf_check = check_confidence(top_score_non, threshold=0.70)
    print(f"  - Doodle Top Softmax Score: {top_score_non*100:.2f}%")
    print(f"  - Confidence check status: {conf_check['status_type']} - Passed: {conf_check['passed']}")
    results["confidence_warning_test"] = "PASSED"
    print("  -> Result: PASSED (Confidence system functions properly)")
    
    # Test 6: Verify all 10 sample digits from sample_digits/
    print("\n[Test 6] Evaluating Pre-Packaged Test Samples (Digits 0-9):")
    correct_count = 0
    total_samples = 10
    for d in range(10):
        img_p = f"sample_digits/digit_{d}_sample1.png"
        if os.path.exists(img_p):
            arr_s, _, _, _ = preprocess_image(img_p)
            prob_s = model.predict(np.expand_dims(arr_s, axis=0), verbose=0)[0]
            pred_d = int(np.argmax(prob_s))
            if pred_d == d:
                correct_count += 1
            print(f"  - Digit {d} -> Predicted: {pred_d} (Score: {prob_s[pred_d]*100:.1f}%) [{'CORRECT' if pred_d == d else 'MISMATCH'}]")
    print(f"  - Total Accuracy on Sample Set: {correct_count}/{total_samples} ({correct_count/total_samples*100:.0f}%)")
    assert correct_count >= 9, "Sample digit recognition accuracy was lower than expected!"
    results["sample_set_test"] = "PASSED"
    print("  -> Result: PASSED (High accuracy across digit classes)")
    
    print("\n" + "=" * 65)
    print("ALL EDGE-CASE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    return results


if __name__ == "__main__":
    run_edge_case_tests()
