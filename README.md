# 🖋️ Handwritten Digit Predictor

Welcome to the **Handwritten Digit Predictor** repository! This project uses a **K-Nearest Neighbors (KNN)** algorithm to recognize digits drawn by the user. It combines **machine learning** with a **Flask-based web interface**, allowing users to interactively test handwritten digit predictions.

## 📌 Project Overview

This project aims to:

- 🧠 Train a **KNN model** to classify handwritten digits using image data.
- 🎨 Provide a **web interface** where users can draw digits.
- 🔍 Predict digits in real-time with visual feedback.

## 📂 Dataset

The model is trained on the **Sklearn Digits Dataset**, which contains:

- 8x8 grayscale images of handwritten digits (0–9)
- Corresponding labels for each image

## 🛠️ Tech Stack

- **Programming Language**: Python 🐍
- **Libraries Used**:
  - `Flask` – to create the web application
  - `NumPy` – for numerical operations
  - `OpenCV` – for image preprocessing
  - `scikit-learn` – for training and saving the KNN model

## 📜 Project Structure

1. 🧪 **Model Training** (`Model_selection.ipynb`) – Trains and evaluates the KNN model  
2. 🧠 **Model Deployment** – Saved as `knn_model.pkl`  
3. 🌐 **Web App** (`app.py`) – Flask app to take user input and make predictions  
4. 🖼️ **Frontend** – HTML/JS canvas for drawing digits (`templates/index.html`)

## 🚀 How to Run

1. Clone this repository:
   ```sh
   git clone https://github.com/DhruvBavaliya13/handwritten-digit-predictor.git
   cd handwritten-digit-predictor
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```sh
   python app.py
   ```
4. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## ✍️ How It Works

- Use your mouse to draw a digit on the canvas
- Click **Predict**
- The KNN model processes the input and returns the predicted digit

## 📸 Sample Interface

![Screenshot 2025-04-09 102347](https://github.com/user-attachments/assets/2e1127a0-9acf-4018-a01b-b1bcc25b0d7b)

## 🤝 Contributing

Feel free to **fork** this repository and submit a pull request if you have ideas to improve the model or the UI!

## 📬 Contact

For any queries or collaboration opportunities, reach out via:  
📧 Email: [drbavaliya13@gmail.com](mailto:drbavaliya13@gmail.com)

---

⭐ If you find this project useful, don’t forget to **star** this repository! ⭐
