import os

from flask import Flask, jsonify, request
from model.image_classification import preprocess_image, predict_class

app = Flask(__name__)

ALLOWED_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "webp"})
UNSUPPORTED_FILE_TYPE_ERROR = "Unsupported file type"


def get_file_extension(filename):
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lstrip(".").lower()


def is_allowed_file(filename):
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


@app.route('/api/v1/health', methods=['GET'])
def health_api():
    return jsonify({"status": "ok"}), 200

@app.route('/api/v1/predict', methods=['POST'])
def predict_api():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files['file']
        if not is_allowed_file(uploaded_file.filename):
            return jsonify({"error": UNSUPPORTED_FILE_TYPE_ERROR}), 400
        
        input_tensor = preprocess_image(uploaded_file.stream)
        pred, acc = predict_class(input_tensor)

        return jsonify({
            "breed": pred,
            "confidence": acc
        }), 200
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 