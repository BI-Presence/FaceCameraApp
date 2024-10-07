from flask import Flask, Response, render_template_string
import cv2
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
import os
import pickle
from datetime import datetime
import webbrowser
from detection import detect_smile_and_head_movement  # Import detection function

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("D:\\Magang\\Bank Indonesia\\ProjectApp\\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-7356b-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendacerealtime-7356b.appspot.com"
})

bucket = storage.bucket()

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background image
imgBackground = cv2.imread("D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\Resources\\BI - Presence.png")

# Import mode images
folderModePath = 'D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\Resources\\Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load encoding file
print("Loading Encode File")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentsIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []
studentInfo = None
image_cache = {}
web_opened = False

def generate_frames():
    global modeType, counter, id, imgStudent, studentInfo, image_cache, web_opened, imgBackground

    while True:
        if web_opened:
            break

        success, img = cap.read()
        if not success:
            break
        
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[162:162 + 480, 55:55 + 640] = img

        if len(imgModeList) > 0:
            imgModeResized = cv2.resize(imgModeList[modeType], (414, 633))
            imgBackground[44:44 + 633, 808:808 + 414] = imgModeResized

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                id = studentsIds[matchIndex]
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                # Check for smile and head movement before proceeding
                if detect_smile_and_head_movement(img):
                    counter += 1
                    modeType = 1

                if id not in image_cache:
                    studentInfo = db.reference(f'worker/{id}').get()
                    print(f"Fetching image for ID: {id}")
                    imgStudent = None
                    for ext in ['jpg', 'png']:
                        blob = bucket.blob(f'D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\Images/{id}.{ext}')
                        if blob.exists():
                            try:
                                array = np.frombuffer(blob.download_as_string(), np.uint8)
                                imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)
                                
                                if studentInfo:
                                    datetimeObject = datetime.strptime(studentInfo['last_attendace_time'], "%Y-%m-%d %H:%M:%S")
                                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                                    if secondsElapsed > 10:
                                        ref = db.reference(f'worker/{id}')
                                        studentInfo['total_attendace'] += 1
                                        ref.child('total_attendace').set(studentInfo['total_attendace'])
                                        ref.child('last_attendace_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                    else:
                                        modeType = 3
                                        counter = 0
                                        imgModeResized = cv2.resize(imgModeList[modeType], (414, 633))
                                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeResized
                            except Exception as e:
                                print(f"Error decoding image: {e}")
                            break
                    image_cache[id] = (imgStudent, studentInfo)
                else:
                    imgStudent, studentInfo = image_cache[id]

        if studentInfo:
            if 10 < counter < 20:
                modeType = 2
                webbrowser.open("https://www.bi.go.id/id/default.aspx")
                web_opened = True

            if counter <= 10:
                if 'total_attendace' in studentInfo:
                    cv2.putText(imgBackground, str(studentInfo['total_attendace']), (851, 108),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                if 'name' in studentInfo:
                    cv2.putText(imgBackground, str(studentInfo['name']), (900, 375),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                if 'divisi' in studentInfo:
                    cv2.putText(imgBackground, str(studentInfo['divisi']), (1006, 535),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                if 'standing' in studentInfo:
                    cv2.putText(imgBackground, str(studentInfo['standing']), (920, 628),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                if 'year' in studentInfo:
                    cv2.putText(imgBackground, str(studentInfo['year']), (1030, 630),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                if 'starting_year' in studentInfo:
                    cv2.putText(imgBackground, str(studentInfo['starting_year']), (1130, 630),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                cv2.putText(imgBackground, str(id), (1006, 444),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        

                if imgStudent is not None:
                    imgStudentResized = cv2.resize(imgStudent, (216, 216))
                    imgBackground[110:110 + 216, 909:909 + 216] = imgStudentResized
            
            counter += 1

            if counter >= 20:
                counter = 0
                modeType = 0
                studentInfo = None
                imgStudent = []
                imgModeResized = cv2.resize(imgModeList[modeType], (414, 633))
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeResized

        ret, buffer = cv2.imencode('.jpg', imgBackground)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Face Recognition System</title>
            <style>
                body {
                    margin: 0;
                    padding: 30px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    background-color: #ffffff;
                }
                .video-container {
                    width: 100%;
                    max-width: 1280px;
                    height: 0;
                    padding-bottom: 56.25%; /* 16:9 Aspect Ratio */
                    position: relative;
                    overflow: hidden;
                }
                .video-container img {
                    position: absolute;
                    top: 0px;
                    bottom: 30px;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }
            </style>
        </head>
        <body>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Face Recognition Feed">
            </div>
        </body>
        </html>
    ''')

if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)