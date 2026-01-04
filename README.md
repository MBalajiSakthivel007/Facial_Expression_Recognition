Facial Expression Recognition using CNN
📌 Project Overview

Facial Expression Recognition (FER) is a key component of Human–Computer Interaction (HCI) systems that enables machines to understand human emotions. This project implements a CNN-based facial expression recognition system enhanced with Digital Image Processing techniques to improve performance under real-world conditions such as lighting variation and noise.

❓ Why This Project?

Understanding facial expressions is essential for building intelligent, emotion-aware systems. This project was developed to gain hands-on experience in:

Image preprocessing and enhancement

Deep learning-based emotion classification

Handling real-world challenges in facial analysis

🎯 Objectives

Detect human faces from images

Classify facial expressions into emotion categories

Improve recognition accuracy using preprocessing techniques

🧠 Technical Approach
Deep Learning

Convolutional Neural Network (CNN)

Implemented using TensorFlow and Keras

Digital Image Processing Techniques

Histogram Equalization (HE)

Contrast Limited Adaptive Histogram Equalization (CLAHE)

Local Binary Patterns (LBP)

Canny Edge Detection

Principal Component Analysis (PCA)

These techniques enhance contrast, extract facial features, and reduce noise before classification.

🧩 Model Details

CNN model trained on grayscale facial images

Input image size: 48 × 48

Multiple convolution and max-pooling layers used

Fully connected layers for classification

Softmax output layer for emotion prediction

⚙️ Model Architecture

Input layer (48×48 grayscale image)

Convolution + ReLU activation layers

Max Pooling layers

Fully Connected (Dense) layers

Softmax output layer

📂 Dataset Details

Facial expression image dataset (grayscale)

Images resized and normalized before training

Face detection performed using Haar Cascade Classifier

📊 Performance & Results

Emotion prediction with confidence score

Example output:

Happy: 63.1%

Tested on unseen facial images for generalization


Sample Output

<img width="2484" height="1220" alt="image" src="https://github.com/user-attachments/assets/56d34a65-6421-41ef-9756-56bb39de9949" />

🌍 Real-World Applications

Emotion-aware virtual assistants

Online learning engagement analysis

Mental health and stress monitoring

Customer sentiment analysis

Smart surveillance systems

⚠️ Limitations

Performance may reduce under extreme lighting conditions

Limited number of emotion classes

Accuracy depends on face alignment and image quality

## 🛠️ Skills & Tools Used

| Category | Technologies |
|----------|-------------|
| Programming Language | Python |
| Deep Learning | Convolutional Neural Networks (CNN) |
| Frameworks | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Image Processing | HE, CLAHE, LBP, Canny Edge Detection |
| Model Development | Model Training and Evaluation |
| Version Control | Git & GitHub |


▶️ How to Run the Project

To download the project source code from GitHub to your local system, run the following command:
```bash
git clone https://github.com/your-username/Facial_Expression_Recognition.git
cd Facial_Expression_Recognition
```

To install all required libraries needed to run the project, execute the following command:
```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

To start the facial expression recognition system, run the following command:
```bash
python main.py
```

## 📁 Project Structure
```
Facial_Expression_Recognition/
│
├── main.py
├── model.h5
├── haarcascade_frontalface_default.xml
├── emotion-classification-cnn-using-keras.ipynb
├── README.md
├── Presentation.pdf
└── Facial_Expression_Recognition_Report.pdf
```

🚀 Future Enhancements

Real-time emotion detection using webcam

Improve accuracy using transfer learning

Support additional emotion classes

Deploy as a web or mobile application

👨‍💻 Author

Balaji Sakthivel



