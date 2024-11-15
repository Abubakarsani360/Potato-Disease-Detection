
# 🥔 Potato Disease Classification Web App

This repository hosts a Flask-based web application for real-time **Potato Disease Classification**. The application allows users to upload images of potato leaves and classifies them into one of three categories using a **Convolutional Neural Network (CNN)** trained on potato leaf images.  

**Categories:**
1. `Potato___Early_blight`
2. `Potato___Late_blight`
3. `Potato___healthy`

---

## 🧠 Model Overview

The model used in this application is a **Convolutional Neural Network (CNN)** trained on an image dataset of potato leaves. Here's an overview of the model and training pipeline:

### **Model Architecture**
- **Input Layer**:
  - Accepts images resized to `(256x256x3)` dimensions (RGB channels).
- **Convolutional Layers**:
  - Extract spatial features from input images using filters.
  - Multiple layers with ReLU activation and MaxPooling for feature reduction.
- **Dense Layers**:
  - Fully connected layers to aggregate extracted features.
  - Final output layer uses Softmax activation for multi-class classification.
- **Optimizer**: Adam (adaptive gradient optimization).
- **Loss Function**: Sparse Categorical Cross-Entropy.
- **Metrics**: Accuracy.

### **Training Dataset**
The dataset consists of images from three categories:
- **Early Blight**: Fungal infection causing brown lesions.
- **Late Blight**: Fungal infection causing black lesions and wilting.
- **Healthy**: Leaves without any infections.

### **Preprocessing Pipeline**
1. **Data Augmentation**:
   - Horizontal/vertical flips, rotations, and scaling to increase dataset diversity.
2. **Normalization**:
   - Pixel values scaled to the range `[0, 1]`.

### **Model Training**
- **Framework**: TensorFlow/Keras
- **Training Environment**: NVIDIA GPU (CUDA-enabled)
- **Hyperparameters**:
  - Batch size: 32
  - Epochs: 20
  - Learning rate: 0.001
- **Validation Accuracy**: ~95%
- **Saved Model**: Stored as `cnn7.keras` for deployment.

---

## 🛠️ Technical Implementation

### **Technology Stack**
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, Bootstrap
- **Model Inference**: TensorFlow/Keras
- **File Handling**: Secure file uploads using `werkzeug`.

### **Prediction Pipeline**
1. **Image Upload**: Users upload images in `PNG`, `JPG`, or `JPEG` formats.
2. **Preprocessing**:
   - Resize the image to `(256x256x3)`.
   - Convert the image to a NumPy array.
3. **Model Inference**:
   - Predict the class probabilities for the uploaded image.
   - Map the predicted index to the corresponding class label.
4. **Output**:
   - Display the predicted class with a confidence score.

### **Folder Structure**
```
project/
├── app.py                  # Flask application script
├── static/                 # Directory for uploaded images
├── templates/              # HTML templates
│   ├── home.html           # Home page
│   ├── about.html          # About page
│   ├── project.html        # Project details
│   └── predict.html        # Result display page
├── saved model/
│   └── cnn7.keras          # Pretrained Keras model
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

---

## 🚀 Getting Started

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/potato-disease-classification.git
cd potato-disease-classification
```

### **2. Install Dependencies**
Ensure Python 3.8+ is installed and run:
```bash
pip install -r requirements.txt
```

### **3. Run the Flask App**
```bash
python app.py
```

### **4. Access the App**
Open your browser and navigate to `http://127.0.0.1:5000`.

---

## 🖼️ Application Features

### **Home Page**
- Upload images for classification.
- Navigate to "About" and "Project" pages.

### **About Page**
- Learn about the purpose and scope of the application.

### **Project Page**
- Get insights into the technical details, including the model and dataset.

### **Prediction Page**
- View the classification results and confidence scores for uploaded images.

---

## ⚙️ Deployment Notes

### **Local Deployment**
- Ensure the `cnn7.keras` model file is placed in the `saved model/` directory.
- Use `debug=True` for development and debugging.

### **Production Deployment**
- Use a WSGI server like **Gunicorn** for deployment:
  ```bash
  gunicorn -w 4 app:app
  ```
- Configure cloud hosting platforms such as AWS, Heroku, or GCP.

---

## 📊 Performance Metrics
- **Accuracy**: Achieved 95% on validation data.
- **Inference Time**: ~100ms per image on a GPU-enabled environment.

---

## 🔮 Future Improvements
1. **Multi-Device Support**:
   - Optimize the model for deployment on mobile and embedded systems.
2. **Enhanced Model**:
   - Experiment with transfer learning for improved accuracy.
3. **Additional Features**:
   - Add real-time video feed analysis.
   - Expand to detect other crop diseases.

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request for any improvements or fixes.

---

Feel free to use, enhance, and share this application. Happy coding! 🚀
