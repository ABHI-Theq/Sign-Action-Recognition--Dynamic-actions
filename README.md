## 🧠 Sign Language Action Recognition

This project performs **real-time sign language action recognition** using **MediaPipe Holistic**, **OpenCV**, and a deep learning model (GRU or LSTM). It detects human body landmarks (face, pose, and hands) and trains a sequential model to classify different sign actions.

---

### 📽️ Demo Preview

Real-time webcam feed is processed to recognize hand signs and display the predicted action live.

---

## ✨ Features

* Real-time video feed with OpenCV
* Landmark detection using MediaPipe Holistic
* Keypoint extraction (pose, left/right hand, face)
* Custom dataset creation and training
* GRU-based (or optionally LSTM-based) action classification
* Model checkpointing and early stopping
* Live predictions from webcam

---

## 🧰 Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Matplotlib
* TensorFlow / Keras

---

## 🗂️ Project Structure

```
📁 SignLangRecognition/
🔹 SIgnLang.ipynb       # Main Jupyter Notebook
🔹 data/                # Contains collected keypoint data
🔹 models/              # Saved model weights
🔹 README.md
```

---

## 🧱 How It Works

### 1. MediaPipe Detection Pipeline

```python
def mediapipe_detection(image, model):
    ...
```

Converts BGR to RGB, disables write, applies detection, and returns the result.

### 2. Keypoint Extraction

Each frame’s landmarks (hands, face, pose) are flattened into a NumPy array and saved.

### 3. Data Collection

Actions are captured over multiple frames (`sequence_length`), and saved as `.npy` sequences for training.

```python
actions = np.array(['hello', 'thanks', 'iloveyou'])
```

### 4. Model Building (GRU/LSTM)

```python
model = Sequential()
model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
...
model.add(Dense(actions.shape[0], activation='softmax'))
```

Trained using `categorical_crossentropy` and `Adam` optimizer.

### 5. Callbacks

* **TensorBoard** for visualization
* **ModelCheckpoint** to save best model on validation loss
* **EarlyStopping** to prevent overfitting

### 6. Real-Time Inference

Live webcam feed is used to collect a buffer of frames, pass through the model, and display the predicted class.

---

## 🧪 Training & Evaluation

* Dataset split: 70% training, 30% testing
* Validation during training
* Evaluation using classification accuracy

---

## 📦 Installation

### Requirements

```bash
pip install opencv-python mediapipe numpy matplotlib tensorflow
```

---

## 🎮 Run the Project

### 1. Run Notebook

Launch `SIgnLang.ipynb` and go through all the cells step-by-step to:

* Collect data
* Train the model
* Start real-time inference

---

## 📊 Future Work

* Add more complex gestures or multi-hand interactions
* Deploy on browser or mobile app
* Improve accuracy with Transformer-based architectures

---

## 👤 Author

**Abhishek Sharma**
Email: [abhi03085e@gmail.com](mailto:abhi03085e@gmail.com)
