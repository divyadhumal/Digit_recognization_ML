# 🧠 Handwritten Digit Recognition using CNN (MNIST)

## 📌 Overview
This project focuses on classifying handwritten digits (0–9) using the MNIST dataset.  
A Convolutional Neural Network (CNN) is built using Keras and TensorFlow to accurately recognize digit images.

---

## 📊 Dataset
- **Dataset:** MNIST  
- **Total samples:** 70,000  
- **Training data:** 60,000  
- **Testing data:** 10,000  
- **Image size:** 28 × 28 pixels  
- **Color format:** Grayscale  
- **Classes:** 0–9  

---

## ⚙️ Tech Stack
- Python  
- NumPy  
- Matplotlib  
- TensorFlow  
- Keras  

---

## 🔍 Workflow
1. Load MNIST dataset  
2. Reshape images to (28, 28, 1)  
3. Normalize pixel values (0–1)  
4. Build CNN model  
5. Train model on training data  
6. Validate model performance  
7. Test model on unseen data  

---

## 🤖 Model Architecture
- Conv2D (32 filters, 3×3, ReLU)  
- MaxPooling2D (2×2)  
- Conv2D (64 filters, 3×3, ReLU)  
- MaxPooling2D (2×2)  
- Flatten  
- Dropout (0.25)  
- Dense (10 neurons, Softmax)  

---

## 📈 Results
- Achieved high accuracy on test dataset  
- Efficient in recognizing handwritten digits  

---

## ⚠️ Features
- EarlyStopping used to avoid overfitting  
- ModelCheckpoint used to save best model  
- Model saved in `.h5` format  

---

## 🚀 Future Improvements
- Apply data augmentation  
- Build deeper CNN architecture  
- Deploy as web application  

---
 
