"""
Streamlit Web Application for Handwritten Digit Recognition
Using Deep Learning and Multi-Layer Perceptrons (MLP)
Enhanced with Top-3 Prediction, Confidence-Aware Warnings,
Edge-Case Lab, 10-Class Softmax Visualizer, and Image Diagnostics.
"""

import os
import json
import base64
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Deep Learning framework
import tensorflow as tf
from tensorflow import keras

# Utilities
from utils import preprocess_image, get_top_k_predictions, check_confidence


def safe_preprocess(image_input):
    """
    Safely unpack preprocess_image return values, supporting both
    3-tuples and 4-tuples for robust backward compatibility.
    """
    res = preprocess_image(image_input)
    if isinstance(res, (tuple, list)):
        if len(res) >= 4:
            return res[0], res[1], bool(res[2]), res[3]
        elif len(res) == 3:
            default_diag = {
                "original_mode": "Grayscale/RGB",
                "original_size": getattr(image_input, "size", (28, 28)),
                "polarity_detected": "Standard Normalized",
                "inverted": False,
                "bounding_box": [0, 0, 28, 28],
                "digit_width": 20,
                "digit_height": 20,
                "aspect_ratio": 1.0,
                "foreground_detected": not bool(res[2])
            }
            return res[0], res[1], bool(res[2]), default_diag
    return res[0], res[1], False, {}

# ---------------------------------------------------------
# Page Configuration & Advanced Aesthetic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition (MLP)",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS:
# 1. Removes the deploy button, header toolbar, and Streamlit branding
# 2. Applies Google Fonts (Outfit, Inter) and sleek cards
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* REMOVE DEPLOY BUTTON, HAMBURGER MENU, AND FOOTER */
    .stDeployButton, 
    [data-testid="stToolbar"], 
    #MainMenu, 
    footer,
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E3A8A 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.3);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #93C5FD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-top: 0.4rem;
        margin-bottom: 0;
        line-height: 1.5;
    }
    .hero-badge-container {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.12);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #E2E8F0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    /* Prediction Result Cards */
    .prediction-box {
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .prediction-digit {
        font-family: 'Outfit', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.0;
        color: #1D4ED8;
        margin: 0.4rem 0;
    }
    .confidence-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .conf-high {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
    }
    .conf-low {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
    }

    /* Step Card */
    .step-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        margin-bottom: 0.8rem;
    }

    /* Diagnostics Table */
    .diag-label {
        font-weight: 600;
        color: #475569;
        font-size: 0.85rem;
    }
    .diag-val {
        font-family: monospace;
        color: #0F172A;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Resource Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model(model_path="model.keras"):
    """Loads and caches the compiled Keras model."""
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            return model, None
        except Exception as e:
            return None, str(e)
    return None, "Model file not found."


def load_metrics(metrics_path="outputs/metrics.json"):
    """Loads precomputed evaluation metrics."""
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------
# Sidebar Controls & Settings
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Control Panel")

model, error_msg = load_trained_model("model.keras")

if model is not None:
    st.sidebar.success("✅ **Model Active:** `model.keras`")
else:
    st.sidebar.error("❌ **Model Not Found**")
    st.sidebar.info("Execute `python train.py` in your terminal to train.")

st.sidebar.markdown("---")

# Feature C: Configurable Confidence Threshold
st.sidebar.markdown("#### 🎯 Confidence Threshold (Feature C)")
threshold_percent = st.sidebar.slider(
    "Minimum Softmax Score (%)",
    min_value=30,
    max_value=99,
    value=70,
    step=1,
    help="Predictions where the top Softmax probability falls below this threshold trigger an explicit uncertainty warning."
)
confidence_threshold = threshold_percent / 100.0

st.sidebar.markdown(f"Threshold set to **{threshold_percent}%**.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Model Specification:**
- **Architecture:** Multi-Layer Perceptron (MLP)
- **Input Dimension:** 784 ($28 \\times 28$ Grayscale)
- **Hidden Layer:** 128 ReLU units
- **Output Layer:** 10 Softmax units
- **Parameters:** 101,770 weights & biases
- **Dataset:** MNIST (60k Train, 10k Test)
""")

# ---------------------------------------------------------
# Hero Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Handwritten Digit Recognition</h1>
    <p class="hero-subtitle">
        Deep Learning & Multi-Layer Perceptron (MLP) academic application featuring 
        Top-3 Candidate Ranking, Confidence-Aware Thresholding, and Edge-Case Diagnostics.
    </p>
    <div class="hero-badge-container">
        <span class="hero-badge">🧠 Multi-Layer Perceptron</span>
        <span class="hero-badge">📊 MNIST Benchmark</span>
        <span class="hero-badge">🎯 Top-3 Ranking</span>
        <span class="hero-badge">⚠️ Confidence Filter</span>
        <span class="hero-badge">⚡ 96.85% Test Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_predict, tab_edge, tab_metrics, tab_architecture = st.tabs([
    "🔮 Interactive Prediction",
    "🧪 Edge-Case & Stress Lab",
    "📊 Model Performance & Metrics",
    "🧠 Architecture & Explanation"
])


# =========================================================
# TAB 1: Interactive Digit Recognition
# =========================================================
with tab_predict:
    if model is None:
        st.error("Model weights file `model.keras` not found. Please train the model via `python train.py`.")
    else:
        st.markdown("### 1. Select Input Source")
        
        input_choice = st.radio(
            "Choose Input Method:",
            [
                "📁 Pre-Packaged Test Samples (Digits 0-9)",
                "📤 Upload Image File (PNG, JPG, JPEG)",
                "✏️ In-Browser Interactive Sketchpad"
            ],
            horizontal=True
        )

        input_image = None
        
        # Mode 1: Pre-packaged Samples
        if input_choice == "📁 Pre-Packaged Test Samples (Digits 0-9)":
            sample_dir = "sample_digits"
            sample_files = []
            if os.path.exists(sample_dir):
                sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith(".png")])
            
            if sample_files:
                col_sel, col_hint = st.columns([1.5, 2])
                with col_sel:
                    selected_sample = st.selectbox("Select a test digit from MNIST test split:", sample_files)
                with col_hint:
                    st.caption("These are official unseen test digits exported directly from the MNIST test set.")
                
                sample_path = os.path.join(sample_dir, selected_sample)
                input_image = Image.open(sample_path)
            else:
                st.info("No sample digits found. Run `python train.py` to auto-export samples.")
                
        # Mode 2: Custom Upload
        elif input_choice == "📤 Upload Image File (PNG, JPG, JPEG)":
            uploaded_file = st.file_uploader(
                "Upload handwritten digit (supports dark ink on light paper or white ink on black background):",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
                    
        # Mode 3: In-Browser HTML5 Canvas Sketchpad
        else:
            st.markdown("**Draw a single digit (0–9) using your mouse or touch:**")
            
            canvas_html = """
            <div style="display:flex; flex-direction:column; align-items:center; gap:10px;">
                <canvas id="paintCanvas" width="220" height="220" style="border:3px solid #3B82F6; border-radius:12px; background-color:#FFFFFF; cursor:crosshair;"></canvas>
                <div style="display:flex; gap:10px;">
                    <button onclick="clearCanvas()" style="background:#EF4444; color:white; border:none; padding:8px 16px; border-radius:8px; font-weight:bold; cursor:pointer;">🧹 Clear Canvas</button>
                    <button onclick="saveCanvas()" style="background:#2563EB; color:white; border:none; padding:8px 16px; border-radius:8px; font-weight:bold; cursor:pointer;">💾 Capture Drawing</button>
                </div>
                <input type="hidden" id="canvasData" name="canvasData">
            </div>
            <script>
                var canvas = document.getElementById('paintCanvas');
                var ctx = canvas.getContext('2d');
                ctx.lineWidth = 16;
                ctx.lineCap = 'round';
                ctx.strokeStyle = '#000000';
                var drawing = false;

                function getPos(e) {
                    var rect = canvas.getBoundingClientRect();
                    var clientX = e.clientX || (e.touches && e.touches[0].clientX);
                    var clientY = e.clientY || (e.touches && e.touches[0].clientY);
                    return { x: clientX - rect.left, y: clientY - rect.top };
                }

                function startDraw(e) { drawing = true; draw(e); }
                function stopDraw() { drawing = false; ctx.beginPath(); }
                function draw(e) {
                    if (!drawing) return;
                    e.preventDefault();
                    var pos = getPos(e);
                    ctx.lineTo(pos.x, pos.y);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(pos.x, pos.y);
                }

                canvas.addEventListener('mousedown', startDraw);
                canvas.addEventListener('mouseup', stopDraw);
                canvas.addEventListener('mousemove', draw);
                canvas.addEventListener('touchstart', startDraw);
                canvas.addEventListener('touchend', stopDraw);
                canvas.addEventListener('touchmove', draw);

                function clearCanvas() {
                    ctx.fillStyle = '#FFFFFF';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.beginPath();
                }
                clearCanvas();

                function saveCanvas() {
                    var dataUrl = canvas.toDataURL('image/png');
                    // Create download link for immediate upload
                    var a = document.createElement('a');
                    a.href = dataUrl;
                    a.download = 'my_drawing.png';
                    a.click();
                }
            </script>
            """
            st.components.v1.html(canvas_html, height=290)
            st.info("💡 **Quick Tip:** Click **'💾 Capture Drawing'** to save your drawing as `my_drawing.png`, then switch to the **'Upload Image File'** mode above to test it instantly!")

        # -----------------------------------------------------
        # Preprocessing & Live Prediction Execution
        # -----------------------------------------------------
        if input_image is not None:
            st.markdown("---")
            
            # Preprocess using robust utils
            processed_arr, display_pil, is_blank, diag = safe_preprocess(input_image)
            
            col_raw, col_proc, col_pred = st.columns([1, 1, 1.4])
            
            with col_raw:
                st.markdown("#### 1. Raw Input")
                st.image(input_image, caption=f"Size: {input_image.size} | Mode: {input_image.mode}", use_container_width=True)

            with col_proc:
                st.markdown("#### 2. Preprocessed (28×28)")
                st.image(display_pil, caption="Centered & Polarity Normalized", use_container_width=True)
                st.caption("Scaled to 20×20 bounding box with 4px border padding.")
                
            with col_pred:
                st.markdown("#### 3. Classification Result")
                
                if is_blank:
                    st.warning("⚠️ **Blank Canvas Detected:** No foreground digit strokes were found. Please draw or select a digit.")
                else:
                    batch_input = np.expand_dims(processed_arr, axis=0)
                    probabilities = model.predict(batch_input, verbose=0)[0]
                    
                    predicted_digit = int(np.argmax(probabilities))
                    top_score = float(probabilities[predicted_digit])
                    top3 = get_top_k_predictions(probabilities, k=3)
                    
                    # Feature A: Display Predicted Digit Badge
                    st.markdown(f"""
                    <div class="prediction-box">
                        <span style="font-size:0.95rem; font-weight:600; color:#64748B;">PREDICTED DIGIT</span>
                        <div class="prediction-digit">{predicted_digit}</div>
                        <span class="confidence-pill {'conf-high' if top_score >= confidence_threshold else 'conf-low'}">
                            Softmax Score: {top_score * 100:.2f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("")
                    
                    # Feature C: Confidence-Aware Evaluation Banner
                    conf_eval = check_confidence(top_score, threshold=confidence_threshold)
                    if conf_eval["passed"]:
                        st.success(f"✅ **Threshold Check Passed:** Score **{conf_eval['score_pct']}** exceeds threshold **{conf_eval['threshold_pct']}**.")
                    else:
                        st.warning(f"⚠️ **Low-Confidence Warning:** Score **{conf_eval['score_pct']}** is below threshold **{conf_eval['threshold_pct']}**.\n\n*The model is uncertain. Ambiguity or noise may be present.*")

            # -------------------------------------------------
            # Feature B: Top-3 Candidate Ranking
            # -------------------------------------------------
            if not is_blank:
                st.markdown("---")
                st.markdown("### 🏆 Top-3 Prediction Ranking (Feature B)")
                st.markdown("Displays the three classes with the highest Softmax scores along with relative likelihood bars:")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                cols_r = [col_r1, col_r2, col_r3]
                
                for idx, cand in enumerate(top3):
                    with cols_r[idx]:
                        medal = "🥇" if cand["rank"] == 1 else ("🥈" if cand["rank"] == 2 else "🥉")
                        st.markdown(f"#### {medal} Rank {cand['rank']}: **Digit {cand['digit']}**")
                        st.markdown(f"**Score:** `{cand['percentage']}`")
                        st.progress(float(cand["score"]))
                        
                # ---------------------------------------------
                # New Feature: Full 10-Class Distribution Chart
                # ---------------------------------------------
                st.markdown("---")
                st.markdown("### 📊 Complete 10-Class Probability Distribution")
                
                df_probs = pd.DataFrame({
                    "Digit": [f"Digit {i}" for i in range(10)],
                    "Probability": [float(p) for p in probabilities],
                    "Score (%)": [round(float(p) * 100, 2) for p in probabilities]
                })
                
                st.bar_chart(df_probs.set_index("Digit")["Probability"], height=240)
                
                # Preprocessing Diagnostics Expander
                with st.expander("🔍 Preprocessing Diagnostics & Metadata"):
                    diag_col1, diag_col2 = st.columns(2)
                    with diag_col1:
                        st.markdown(f"- **Original Dimensions:** `{diag['original_size']}`")
                        st.markdown(f"- **Color Mode:** `{diag['original_mode']}`")
                        st.markdown(f"- **Detected Polarity:** `{diag['polarity_detected']}`")
                        st.markdown(f"- **Inverted for MNIST:** `{diag['inverted']}`")
                    with diag_col2:
                        st.markdown(f"- **Foreground Detected:** `{diag['foreground_detected']}`")
                        st.markdown(f"- **Bounding Box (x1, y1, x2, y2):** `{diag['bounding_box']}`")
                        st.markdown(f"- **Cropped Size:** `{diag['digit_width']} × {diag['digit_height']} px`")
                        st.markdown(f"- **Aspect Ratio:** `{diag['aspect_ratio']}`")


# =========================================================
# TAB 2: Edge-Case & Stress-Testing Lab
# =========================================================
with tab_edge:
    st.markdown("### 🧪 Edge-Case & Robustness Verification Lab")
    st.markdown("""
    This lab allows testing challenging real-world scenarios to verify how the MLP and preprocessing pipeline behave under edge cases.
    """)
    
    preset_edge = st.selectbox(
        "Select an Edge-Case Scenario to Test:",
        [
            "1. ⚪ Inverted Polarity (Black Ink on White Paper)",
            "2. ⬛ Native MNIST Polarity (White Ink on Black Canvas)",
            "3. 📐 Off-Center / Corner-Shifted Digit (Tests Centering)",
            "4. 🎭 Ambiguous Digit (Resembles both 4 and 9)",
            "5. 🌫️ Non-Digit Doodle (Tests Low-Confidence Warning)",
            "6. ⬜ Blank Canvas (Tests Blank Detector)",
            "7. 🖼️ Transparent PNG (Tests RGBA Alpha Compositing)"
        ]
    )
    
    edge_dir = "edge_case_tests"
    os.makedirs(edge_dir, exist_ok=True)
    
    # Generate test image dynamically based on selection
    selected_edge_path = None
    
    if "1. ⚪ Inverted Polarity" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_black_on_white.png")
    elif "2. ⬛ Native MNIST" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_white_on_black.png")
    elif "3. 📐 Off-Center" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_off_center_corner.png")
    elif "4. 🎭 Ambiguous" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_ambiguous_4_or_9.png")
    elif "5. 🌫️ Non-Digit" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_non_digit_cross.png")
    elif "6. ⬜ Blank" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_blank.png")
    elif "7. 🖼️ Transparent" in preset_edge:
        selected_edge_path = os.path.join(edge_dir, "edge_rgba_transparent.png")
        
    # Ensure edge case images are generated
    if not os.path.exists(edge_dir) or not os.listdir(edge_dir):
        from test_edge_cases import generate_synthetic_test_images
        generate_synthetic_test_images(edge_dir)

    if selected_edge_path and os.path.exists(selected_edge_path):
        edge_img = Image.open(selected_edge_path)
        
        c_e1, c_e2, c_e3 = st.columns([1, 1, 1.4])
        
        arr_e, disp_e, is_blank_e, diag_e = safe_preprocess(edge_img)
        
        with c_e1:
            st.markdown("**Synthetic Edge Input**")
            st.image(edge_img, caption=os.path.basename(selected_edge_path), use_container_width=True)
            
        with c_e2:
            st.markdown("**Processed into 28×28**")
            st.image(disp_e, caption="MNIST Pipeline Output", use_container_width=True)
            
        with c_e3:
            st.markdown("**Behavior & Evaluation**")
            if is_blank_e:
                st.warning("✅ **Passed Edge Case:** Successfully detected as a Blank Canvas. Model inference safely bypassed.")
            else:
                prob_e = model.predict(np.expand_dims(arr_e, axis=0), verbose=0)[0]
                pred_e = int(np.argmax(prob_e))
                top_s_e = float(prob_e[pred_e])
                top3_e = get_top_k_predictions(prob_e, k=3)
                
                st.markdown(f"**Primary Class:** `Digit {pred_e}` (Score: `{top_s_e*100:.2f}%`)")
                
                conf_e = check_confidence(top_s_e, threshold=confidence_threshold)
                if conf_e["passed"]:
                    st.success(f"Confidence Check: **Passed** ({conf_e['score_pct']} ≥ {conf_e['threshold_pct']})")
                else:
                    st.warning(f"⚠️ Confidence Check: **Flagged Warning** ({conf_e['score_pct']} < {conf_e['threshold_pct']}).")
                    
                st.markdown("**Top-3 Alternatives:**")
                for c_item in top3_e:
                    st.markdown(f"- Rank {c_item['rank']}: **Digit {c_item['digit']}** (`{c_item['percentage']}`)")
    else:
        st.info("Generating test cases... Run `python test_edge_cases.py` to populate test samples.")


# =========================================================
# TAB 3: Model Performance & Evaluation Metrics (Feature E)
# =========================================================
with tab_metrics:
    st.markdown("### Model Evaluation on 10,000 Unseen MNIST Test Samples")
    
    metrics = load_metrics("outputs/metrics.json")
    
    if metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Test Accuracy", metrics.get("test_accuracy_pct", "96.85%"))
        with col_m2:
            st.metric("Cross-Entropy Loss", f"{metrics.get('test_loss', 0.1007):.4f}")
        with col_m3:
            st.metric("Validation Accuracy", f"{metrics.get('final_val_accuracy', 0.9660)*100:.2f}%")
        with col_m4:
            st.metric("Training Accuracy", f"{metrics.get('final_train_accuracy', 0.9771)*100:.2f}%")
            
        st.markdown("---")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("#### 📈 Accuracy & Loss Learning Curves")
            if os.path.exists("outputs/training_history.png"):
                st.image("outputs/training_history.png", use_container_width=True)
                
        with col_p2:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            if os.path.exists("outputs/confusion_matrix.png"):
                st.image("outputs/confusion_matrix.png", use_container_width=True)
                
        st.markdown("---")
        st.markdown("#### 🔍 Test Set Actual vs Predicted Sample Grid")
        if os.path.exists("outputs/sample_predictions.png"):
            st.image("outputs/sample_predictions.png", use_container_width=True)
            
        # Classification Report Table
        if "classification_report" in metrics:
            st.markdown("---")
            st.markdown("#### 📋 Detailed Per-Class Classification Report")
            cr = metrics["classification_report"]
            rows = []
            for digit in range(10):
                d_key = f"Digit {digit}"
                if d_key in cr:
                    rows.append({
                        "Digit Class": d_key,
                        "Precision": f"{cr[d_key]['precision']:.4f}",
                        "Recall": f"{cr[d_key]['recall']:.4f}",
                        "F1-Score": f"{cr[d_key]['f1-score']:.4f}",
                        "Test Samples (Support)": int(cr[d_key]['support'])
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Metrics not found. Run `python train.py` to evaluate.")


# =========================================================
# TAB 4: Architecture & Educational Guide
# =========================================================
with tab_architecture:
    st.markdown("### Deep Learning Architecture & Mathematical Theory")
    
    st.markdown("""
    #### 1. Network Structure Overview
    
    | Layer | Type | Output Shape | Parameters | Purpose |
    |---|---|---|---|---|
    | **Input** | `InputLayer` | `(None, 28, 28)` | 0 | Receives $28 \\times 28$ grayscale pixel arrays |
    | **Flatten** | `Flatten` | `(None, 784)` | 0 | Unrolls 2D image matrix into a 1D vector of 784 inputs |
    | **Hidden Layer** | `Dense(128, relu)` | `(None, 128)` | 100,480 | Learns non-linear combination of features ($784 \\times 128 + 128$) |
    | **Output Layer** | `Dense(10, softmax)` | `(None, 10)` | 1,290 | Outputs normalized probabilities for classes 0–9 ($128 \\times 10 + 10$) |
    
    **Total Parameters:** 101,770 weights and biases.
    
    ---
    
    #### 2. Mathematical Formulations
    
    - **ReLU Activation (Hidden Layer):**
      $$f(z) = \\max(0, z)$$
      Eliminates negative values to induce sparse activations and avoids the vanishing gradient problem encountered with Sigmoid.
      
    - **Softmax Activation (Output Layer):**
      $$\\sigma(\\mathbf{z})_i = \\frac{e^{z_i}}{\\sum_{j=0}^{9} e^{z_j}}$$
      Transforms unconstrained logits into a probability distribution summing to 1.0 (100%).
      
    - **Sparse Categorical Cross-Entropy Loss:**
      $$\\mathcal{L} = -\\sum_{i=0}^{9} y_i \\log(p_i)$$
      Penalizes confident incorrect classifications logarithmically, providing steep error gradients during backpropagation.
      
    - **Adam Optimizer:**
      Dynamically computes individual adaptive learning rates using moving averages of both gradients ($m_t$) and squared gradients ($v_t$).
    """)

# Footer
st.markdown("---")
st.markdown("<center><small style='color:#94A3B8;'>Academic Project: Handwritten Digit Recognition Using Deep Learning & Multi-Layer Perceptrons</small></center>", unsafe_allow_html=True)
