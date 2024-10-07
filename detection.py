import cv2

def detect_smile_and_head_movement(img):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load pre-trained classifiers for smile and face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    smile_detected = False
    head_movement_detected = False

    for (x, y, w, h) in faces:
        # Detect smile within the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # Detect smile (at least one smile found)
        if len(smiles) > 0:
            smile_detected = True

        # Basic head movement detection (using position changes)
        if abs(x) > 10 or abs(y) > 10:  # Adjust the threshold as needed
            head_movement_detected = True

    # Return True if both smile and head movement are detected
    return smile_detected and head_movement_detected
