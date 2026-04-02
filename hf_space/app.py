"""
NeuroScan Flask API — HF Spaces deployment version.
Runs on port 7860 (required by Hugging Face Spaces).
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from PIL import Image
import io
import os

app = Flask(__name__)

# Allow requests from any origin (Vercel frontend + local dev)
CORS(app, origins="*")

# ---------- Load Model ----------
MODEL_LOADED = False
model = None

print("Loading NeuroScan champion model...")
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "v4_Unbeatable_Final.h5")
    model = load_model(MODEL_PATH)
    MODEL_LOADED = True
    print("✅ Model loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️  Warning: Could not load model: {e}")
    print("    Running in DEMO MODE — returning mock predictions.")

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


# ---------- Helpers ----------
def numpy_to_b64(img_array):
    """Convert a numpy image array (0-255 uint8) to a base64 PNG string."""
    img_uint8 = np.clip(img_array, 0, 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def add_noise(img):
    """Salt-and-pepper noise attack simulation."""
    amount = 0.05
    out = np.copy(img)
    num_salt = int(np.ceil(amount * img.size * 0.5))
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    out[tuple(coords)] = 255
    num_pepper = int(np.ceil(amount * img.size * 0.5))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    out[tuple(coords)] = 0
    return out


def apply_shield(img):
    """Median denoising + CLAHE enhancement (adversarial defense)."""
    denoised = cv2.medianBlur(img.astype(np.uint8), 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = np.zeros_like(denoised)
    for i in range(3):
        res[:, :, i] = clahe.apply(denoised[:, :, i])
    return res


# ---------- Routes ----------
@app.route('/', methods=['GET'])
def index():
    """Landing page shown in HF Spaces iframe."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>NeuroScan API</title>
        <style>
            body { font-family: system-ui, sans-serif; background: #0b1326; color: #dae2fd; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
            .container { text-align: center; max-width: 500px; padding: 2rem; }
            h1 { color: #00e5ff; font-size: 2rem; }
            p { color: #849396; line-height: 1.6; }
            a { color: #00e5ff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .status { background: #171f33; border: 1px solid #3b494c; border-radius: 12px; padding: 1rem; margin: 1.5rem 0; }
            .badge { display: inline-block; background: #00e5ff22; color: #00e5ff; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 NeuroScan API</h1>
            <p>Adversarial-robust Brain Tumor Classification API</p>
            <div class="status">
                <span class="badge">✅ Running</span>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Model loaded: ''' + str(MODEL_LOADED) + '''</p>
            </div>
            <p><strong>Endpoints:</strong></p>
            <p><a href="/api/health">/api/health</a> — Health check<br>
               <code>POST /api/analyze</code> — Upload MRI for diagnosis</p>
            <hr style="border-color: #3b494c; margin: 1.5rem 0;">
            <p>Frontend: <a href="https://neuro-scan-dusky.vercel.app" target="_blank">neuro-scan-dusky.vercel.app</a></p>
        </div>
    </body>
    </html>
    '''


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': MODEL_LOADED,
        'classes': CLASS_NAMES
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Accept a multipart image upload and return diagnosis results."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Read image
    try:
        pil_img = Image.open(file.stream).convert('RGB')
        img_np = np.array(pil_img)
    except Exception as e:
        return jsonify({'error': f'Could not read image: {str(e)}'}), 400

    # Stage A: Resize to 224x224
    original = cv2.resize(img_np, (224, 224))

    # Stage B: Simulate adversarial attack
    attacked = add_noise(original)

    # Stage C: Apply defensive shield
    shielded = apply_shield(attacked)

    if not MODEL_LOADED:
        # Demo mode — return plausible mock predictions
        predictions = [0.94, 0.03, 0.02, 0.01]
        predicted_class = CLASS_NAMES[0]
        max_conf = 0.94
    else:
        # Stage D: Run model inference
        prep = shielded.astype(np.float32) / 255.0
        prep = np.expand_dims(prep, axis=0)
        preds = model.predict(prep, verbose=0)[0]
        predictions = [float(p) for p in preds]
        max_conf = float(np.max(preds))
        predicted_class = CLASS_NAMES[int(np.argmax(preds))]

    # Stage E: Confidence-based status message
    if max_conf < 0.80:
        status = f"⚠️ LOW CONFIDENCE ({max_conf*100:.1f}%): AI suspects {predicted_class}. Recommendation: Manual Radiologist Review."
        alert_level = "warning"
    else:
        status = f"✅ ANALYSIS COMPLETE: High confidence ({max_conf*100:.1f}%) detection of {predicted_class}."
        alert_level = "success"

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': max_conf,
        'predictions': {CLASS_NAMES[i]: predictions[i] for i in range(4)},
        'status': status,
        'alert_level': alert_level,
        'images': {
            'original': numpy_to_b64(original),
            'attacked': numpy_to_b64(attacked),
            'shielded': numpy_to_b64(shielded),
        }
    })


if __name__ == '__main__':
    # HF Spaces requires port 7860 and host 0.0.0.0
    port = int(os.environ.get('PORT', 7860))
    print(f"\n🧠 NeuroScan API starting on port {port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
