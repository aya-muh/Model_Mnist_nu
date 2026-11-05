import os
import io
import time
import base64
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB maximum upload size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Locate and load the trained XGBoost model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, 'best_model_xgboost_mnist.pkl'),
    os.path.join(BASE_DIR, '..', 'best_model_xgboost_mnist.pkl')
]

model = None
feature_names = None

for path in MODEL_CANDIDATES:
    if os.path.exists(path):
        try:
            print(f"[MODEL] Loading XGBoost model from: {path}")
            model = joblib.load(path)
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = [f"{r}x{c}" for r in range(1, 29) for c in range(1, 29)]
            print(f"[MODEL] Successfully loaded model with {len(feature_names)} features.")
            break
        except Exception as e:
            print(f"[MODEL ERROR] Failed loading model from {path}: {e}")

if model is None:
    print("[MODEL WARNING] Model file best_model_xgboost_mnist.pkl not found during initialization.")


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes: bytes):
    """
    Enhanced Optical Preprocessing Pipeline for MNIST Digit Classification:
    1. Open and composite any transparent (alpha) channels onto a neutral white background.
    2. Convert image to single-channel Grayscale ('L').
    3. Automatic background polarity detection: if average pixel intensity > 128 (or border mean > 128),
       invert the image so the digit strokes are bright (white) on a dark (black) background.
    4. Find the tight bounding box of the drawn digit using adaptive thresholding.
    5. Crop the digit and scale it to fit inside a 20x20 pixel bounding box while strictly preserving
       aspect ratio (standard NIST/MNIST specification).
    6. Center the 20x20 normalized digit inside a standard 28x28 frame with realistic 4-pixel padding,
       and refine positioning via center-of-mass alignment.
    7. Normalize values to [0.0, 1.0] (divide by 255.0) and flatten into a 784-element array.
    8. Generate a Base64 preview of the preprocessed 28x28 bitmap for real-time visual inspection.
    """
    image = Image.open(io.BytesIO(image_bytes))

    # 1. Handle transparency (e.g. Canvas PNG export or transparent files)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        image = bg.convert('RGB')

    # a. Convert to Grayscale ('L')
    gray_image = image.convert('L')
    arr = np.array(gray_image, dtype=np.float32)

    # b. Background polarity detection:
    # If average pixel intensity > 128 or border pixels average > 128, invert image
    overall_mean = float(np.mean(arr))
    border_pixels = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    border_mean = float(np.mean(border_pixels))
    inverted = False

    if overall_mean > 128.0 or border_mean > 128.0:
        gray_image = ImageOps.invert(gray_image)
        arr = np.array(gray_image, dtype=np.float32)
        inverted = True

    # c. Find bounding box of the drawn digit, crop it, and center it inside 28x28 frame
    # Use adaptive threshold to isolate stroke ink from paper noise
    noise_floor = 25.0
    p75 = float(np.percentile(arr, 75)) if arr.size > 0 else 0.0
    threshold = max(noise_floor, p75 * 0.35)

    stroke_mask = arr > threshold

    if np.any(stroke_mask):
        coords = np.argwhere(stroke_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # Slice upper bound is exclusive

        crop = arr[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape

        # Scale to fit inside 20x20 box preserving aspect ratio (MNIST standard)
        if crop_w > crop_h:
            new_w = 20
            new_h = max(1, int(round((crop_h / crop_w) * 20.0)))
        else:
            new_h = 20
            new_w = max(1, int(round((crop_w / crop_h) * 20.0)))

        cropped_img = Image.fromarray(crop.astype(np.uint8))
        resized_digit = cropped_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized_arr = np.array(resized_digit, dtype=np.float32)

        # Place onto 28x28 black canvas (geometric centering)
        canvas_28 = np.zeros((28, 28), dtype=np.float32)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

        # Center-of-mass refinement (MNIST translation alignment)
        total_mass = np.sum(canvas_28)
        if total_mass > 0:
            y_indices, x_indices = np.indices((28, 28))
            cy = np.sum(y_indices * canvas_28) / total_mass
            cx = np.sum(x_indices * canvas_28) / total_mass
            shift_x = int(np.clip(round(13.5 - cx), -3, 3))
            shift_y = int(np.clip(round(13.5 - cy), -3, 3))

            if shift_x != 0 or shift_y != 0:
                shifted = np.zeros((28, 28), dtype=np.float32)
                src_y_start = max(0, -shift_y)
                src_y_end = min(28, 28 - shift_y)
                dst_y_start = max(0, shift_y)
                dst_y_end = min(28, 28 + shift_y)

                src_x_start = max(0, -shift_x)
                src_x_end = min(28, 28 - shift_x)
                dst_x_start = max(0, shift_x)
                dst_x_end = min(28, 28 + shift_x)

                shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = canvas_28[src_y_start:src_y_end, src_x_start:src_x_end]
                canvas_28 = shifted
    else:
        # Fallback for empty or blank input
        canvas_28 = np.array(gray_image.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.float32)

    # d. Normalize values to [0.0, 1.0] (divide by 255.0) and flatten into 784-element array
    normalized_array = np.clip(canvas_28 / 255.0, 0.0, 1.0)
    flattened_features = normalized_array.flatten()

    # Generate Base64 preview of the preprocessed 28x28 matrix
    preview_io = io.BytesIO()
    preview_img = Image.fromarray((normalized_array * 255.0).astype(np.uint8))
    preview_img.save(preview_io, format='PNG')
    preview_base64 = "data:image/png;base64," + base64.b64encode(preview_io.getvalue()).decode('utf-8')

    return flattened_features, preview_base64, inverted, overall_mean


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ONLINE" if model is not None else "MODEL_UNAVAILABLE",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": "Classification model is not loaded.",
            "error_ar": "نموذج التصنيف غير محمل على الخادم."
        }), 500

    image_bytes = None
    input_source = None
    start_time = time.time()

    # Case 1: Direct File Upload
    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected for verification.",
                "error_ar": "لم يتم اختيار أي ملف للفحص."
            }), 400

        if not allowed_file(uploaded_file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported file format. Supported formats: PNG, JPG, JPEG, WEBP.",
                "error_ar": "صيغة الملف غير مدعومة. الصيغ المدعومة: PNG, JPG, JPEG, WEBP."
            }), 400

        try:
            image_bytes = uploaded_file.read()
            input_source = f"File: {uploaded_file.filename}"
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed reading uploaded image: {str(e)}",
                "error_ar": "فشل في قراءة بيانات الصورة المرفوعة."
            }), 400

    # Case 2: Interactive Drawing Canvas (Base64 payload) or Image URL (JSON)
    else:
        req_json = request.get_json(silent=True) or {}

        # 2a: Interactive Canvas Data
        if 'canvas_data' in req_json or 'image_data' in req_json:
            raw_b64 = req_json.get('canvas_data') or req_json.get('image_data')
            if not raw_b64 or not raw_b64.strip():
                return jsonify({
                    "success": False,
                    "error": "Canvas stroke data is empty.",
                    "error_ar": "لوحة الرسم فارغة، يرجى كتابة رقم أولاً."
                }), 400

            try:
                if ',' in raw_b64:
                    raw_b64 = raw_b64.split(',', 1)[1]
                image_bytes = base64.b64decode(raw_b64)
                input_source = "Interactive Canvas / لوحة رسم تفاعلية"
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Failed decoding canvas image data: {str(e)}",
                    "error_ar": "فشل فك تشفير بيانات لوحة الرسم."
                }), 400

        # 2b: Image URL
        elif 'image_url' in req_json or 'image_url' in request.form:
            image_url = (req_json.get('image_url') or request.form.get('image_url') or '').strip()

            if not image_url:
                return jsonify({
                    "success": False,
                    "error": "No image URL provided.",
                    "error_ar": "يرجى تقديم رابط صورة صالح."
                }), 400

            if not (image_url.startswith('http://') or image_url.startswith('https://')):
                return jsonify({
                    "success": False,
                    "error": "Invalid URL protocol. Must start with http:// or https://",
                    "error_ar": "بروتوكول الرابط غير صالح. يجب أن يبدأ بـ http:// أو https://"
                }), 400

            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) DigitRecognitionService/1.1'
                }
                resp = requests.get(image_url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    return jsonify({
                        "success": False,
                        "error": f"Failed to download image from URL. Server returned HTTP {resp.status_code}.",
                        "error_ar": f"فشل تحميل الصورة من الرابط. استجاب الخادم برمز {resp.status_code}."
                    }), 400

                image_bytes = resp.content
                input_source = f"URL: {image_url}"
            except requests.exceptions.Timeout:
                return jsonify({
                    "success": False,
                    "error": "Connection timed out while fetching image from the provided URL.",
                    "error_ar": "انتهت مهلة الاتصال أثناء محاولة جلب الصورة من الرابط."
                }), 400
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Network error fetching image: {str(e)}",
                    "error_ar": "خطأ في الشبكة أثناء جلب الصورة من الرابط."
                }), 400

        else:
            return jsonify({
                "success": False,
                "error": "No image input detected (file, canvas, or URL required).",
                "error_ar": "لم يتم استلام أي بيانات مدخلة (مطلوب ملف أو رسم أو رابط)."
            }), 400

    # Execute Preprocessing Pipeline
    try:
        flattened_features, preview_base64, inverted, overall_mean = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Image processing failed: {str(e)}",
            "error_ar": "فشلت معالجة الصورة. تأكد من إدخال مستند صالح."
        }), 400

    # Model Inference with DataFrame matching feature names
    try:
        df_input = pd.DataFrame([flattened_features], columns=feature_names)
        predicted_digit = int(model.predict(df_input)[0])

        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_input)[0]
            confidence = float(np.max(probabilities) * 100.0)
        else:
            probabilities = np.zeros(10)
            probabilities[predicted_digit] = 1.0
            confidence = 100.0

        # Build formal probability distribution list
        distribution = []
        for d in range(10):
            prob_val = float(probabilities[d])
            distribution.append({
                "digit": d,
                "probability": round(prob_val, 6),
                "percentage": round(prob_val * 100.0, 2)
            })

        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": {str(d): round(float(probabilities[d]), 6) for d in range(10)},
            "distribution": distribution,
            "inverted": inverted,
            "background_shade": "Light Media (Inverted to Dark)" if inverted else "Dark Background (Standard MNIST)",
            "pixel_mean": round(overall_mean, 2),
            "preview_image": preview_base64,
            "input_source": input_source,
            "inference_time_ms": inference_time_ms
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Inference execution failed: {str(e)}",
            "error_ar": "فشل تنفيذ عملية التنبؤ داخل النموذج."
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
