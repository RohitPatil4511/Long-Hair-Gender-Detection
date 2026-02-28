from flask import Flask, render_template, request
import cv2
import os

from face_analysis import predict_age_gender

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    frame = cv2.imread(filepath)

    age, gender = predict_age_gender(frame)

    if age is None:
        age = "No Face Detected"
        gender = "No Face Detected"

    return render_template(
        "result.html",
        age=age,
        gender=gender,
        image=filepath
    )


if __name__ == "__main__":
   app.run(host="127.0.0.1", port=8000, debug=True)