from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# ============================================
# LOAD MODELS AND CONFIGURATION
# ============================================
face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
cap = cv2.VideoCapture(0)

# ============================================
# DIP PREPROCESSING FUNCTIONS
# ============================================
def apply_histogram_equalization(gray_img):
    equalized = cv2.equalizeHist(gray_img)
    return equalized

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)
    return enhanced

def extract_lbp_features(gray_img):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    lbp = np.uint8((lbp / lbp.max()) * 255)
    return lbp

def apply_edge_detection(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    return edges

def apply_bilateral_filter(gray_img):
    filtered = cv2.bilateralFilter(gray_img, 9, 75, 75)
    return filtered

def preprocess_face_dip(roi_gray):
    denoised = apply_bilateral_filter(roi_gray)
    enhanced = apply_clahe(denoised)
    return enhanced

# ============================================
# MAIN LOOP
# ============================================
print("Press 'q' to quit")
print("Press '1' to toggle DIP preprocessing ON/OFF")
print("Press '2' to show LBP features")
print("Press '3' to show Edge detection")

use_dip = True
show_lbp = False
show_edges = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # ✅ Define default status text in case no faces are detected
    status_text = 'No Face Detected'

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_original = roi_gray.copy()

        # Apply DIP preprocessing if enabled
        if use_dip:
            roi_gray_processed = preprocess_face_dip(roi_gray)
            status_text = 'DIP: ON (CLAHE + Bilateral)'
        else:
            roi_gray_processed = roi_gray
            status_text = 'DIP: OFF (Original)'

        # Show extra visualizations
        if show_lbp:
            lbp_img = extract_lbp_features(roi_gray_processed)
            cv2.imshow('LBP Features', cv2.resize(lbp_img, (200, 200)))

        if show_edges:
            edges_img = apply_edge_detection(roi_gray_processed)
            cv2.imshow('Edge Detection', cv2.resize(edges_img, (200, 200)))

        comparison = np.hstack([
            cv2.resize(roi_original, (100, 100)),
            cv2.resize(roi_gray_processed, (100, 100))
        ])
        cv2.imshow('Before (Left) vs After (Right)', comparison)

        roi_gray_resized = cv2.resize(roi_gray_processed, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray_resized]) != 0:
            roi = roi_gray_resized.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            confidence = prediction.max() * 100

            display_text = f'{label}: {confidence:.1f}%'
            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ✅ Display status text regardless of detection
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow('Emotion Detector with DIP Preprocessing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        use_dip = not use_dip
        print(f"DIP Preprocessing: {'ON' if use_dip else 'OFF'}")
    elif key == ord('2'):
        show_lbp = not show_lbp
        if not show_lbp:
            cv2.destroyWindow('LBP Features')
    elif key == ord('3'):
        show_edges = not show_edges
        if not show_edges:
            cv2.destroyWindow('Edge Detection')

cap.release()
cv2.destroyAllWindows()