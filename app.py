from flask import Flask, render_template, request
from model.image_classification import preprocess_image, predict_class

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("index.html")
 

@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            input_tensor = preprocess_image(request.files['file'].stream)
            pred, acc = predict_class(input_tensor)
            output = f"I'm {acc*100:.0f}% sure it's {pred}"
            return render_template("result.html", predictions=output)
 
    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)
 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)