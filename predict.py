from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Set your secret key here
app.config['UPLOAD_FOLDER'] = 'static/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
model_path = 'D:\ML projects\milestone_3\saved model\cnn7.keras'  # Replace with your actual model path
model = load_model(model_path)
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the project information page
@app.route('/project')
def project():
    return render_template('project.html')

# Route for handling the upload and prediction
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            prediction, confidence = predict_file(file_path)

            return render_template('predict.html', prediction=prediction, confidence=confidence, image_path=file_path)
        else:
            flash('Error: Only PNG, JPG, and JPEG file formats are allowed.')
            return redirect(request.referrer or url_for('home'))  # Redirect to previous page or home page

    # Handle GET requests to /upload (if needed)
    return redirect(url_for('home'))  # Redirect to home page for GET requests

# Prediction function
def predict_file(file_path):
    try:
        test_image = load_img(file_path, target_size=(256, 256))  # Match target size with training image size
        test_image = img_to_array(test_image)  # Convert image to np array
        test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D

        result = model.predict(test_image)  # Predict the class probabilities
        pred = np.argmax(result, axis=1)[0]  # Get the predicted class index
        prediction = class_names[pred]  # Get the predicted class label
        confidence = round(np.max(result) * 100, 2)  # Calculate confidence percentage

        return prediction, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for the predict page (if needed separately)
@app.route('/predict')
def predict_page():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)