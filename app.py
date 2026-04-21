from flask import Flask, jsonify, request
from model.image_classification import preprocess_image, predict_class

app = Flask(__name__)

@app.route('/api/v1/predict', methods=['POST'])
def predict_api():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        input_tensor = preprocess_image(request.files['file'].stream)
        pred, acc = predict_class(input_tensor)

        return jsonify({
            "breed": pred,
            "confidence": acc
        }), 200
 
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 