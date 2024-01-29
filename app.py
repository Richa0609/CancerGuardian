from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model("cnn.h5")
class_names = ["colon_aca", "colon_n"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded image from the form
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(image_path)

            # Preprocess the image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Make a prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_names[predicted_class]
            confidence_percentage = prediction[0][predicted_class] * 100

            return render_template("index.html", prediction=predicted_label, confidence=confidence_percentage, image_path=image_path)

    return render_template("index.html", prediction=None, confidence=None, image_path=None)

if __name__ == "__main__":
    app.run(debug=True)
