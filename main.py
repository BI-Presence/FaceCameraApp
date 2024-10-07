import cv2
import os
import numpy as np
import pickle
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime
import webbrowser  # Tambahkan impor untuk webbrowser

# Inisialisasi Firebase
cred = credentials.Certificate("D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-7356b-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendacerealtime-7356b.appspot.com"
})

bucket = storage.bucket()

# Coba gunakan kamera dengan index 0 jika 1 tidak tersedia
cap = cv2.VideoCapture(0)

# Set resolusi kamera (lebar = 640, tinggi = 480)
cap.set(3, 640)  # 3 adalah ID untuk lebar (width)
cap.set(4, 480)  # 4 adalah ID untuk tinggi (height)

# Pastikan path ini benar dan file tersedia
imgBackground = cv2.imread("D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\Resources\\BI - Presence.png")

# Importing the mode images into a list
folderModePath = 'D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\Resources\\Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

# Load gambar dari folder
for path in modePathList:
    full_path = os.path.join(folderModePath, path)
    imgMode = cv2.imread(full_path)
    if imgMode is not None:
        imgModeList.append(imgMode)
    else:
        print(f"Failed to load image at {full_path}")

# Load the encoding file
print("Loading Encode File")
with open('EncodeFile.p', 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown, studentsIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgStudent = []
studentInfo = None  # Inisialisasi awal untuk studentInfo

# Cache untuk menyimpan informasi tentang gambar yang sudah diambil/dicoba
image_cache = {}

# Flag untuk menghentikan deteksi setelah web dibuka
web_opened = False

while True:
    if web_opened:  # Jika web sudah dibuka, hentikan loop
        break

    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img

    # Resize image dari imgModeList agar sesuai dengan area background
    if len(imgModeList) > 0:
        imgModeResized = cv2.resize(imgModeList[modeType], (414, 633))  # Sesuaikan ukuran ke (414, 633)
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeResized  # Masukkan gambar yang telah di-resize

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

            counter += 1
            modeType = 1

            studentInfo = db.reference(f'worker/{id}').get()

            if not isinstance(studentInfo, dict):
                print(f"Data tidak valid untuk ID: {id}")
                studentInfo = None

            if id not in image_cache:
                print(f"Fetching image for ID: {id}")
                imgStudent = None
                extensions = ['jpg', 'png']

                for ext in extensions:
                    blob = bucket.blob(f'D:\\Magang\\Bank Indonesia\\ProjectFaceRecognition\\Images/{id}.{ext}')
                    
                    if blob.exists():
                        try:
                            array = np.frombuffer(blob.download_as_string(), np.uint8)
                            imgStudent = cv2.imdecode(array, cv2.IMREAD_COLOR)

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
                    else:
                        print(f"Gambar dengan ekstensi .{ext} tidak ditemukan untuk ID: {id}")

                image_cache[id] = imgStudent
            else:
                imgStudent = image_cache[id]
                if imgStudent is None:
                    print(f"Cached: Gambar tidak ditemukan untuk ID: {id}")
                else:
                    print(f"Cached: Gambar ditemukan untuk ID: {id}")

            if studentInfo is not None:
                print(studentInfo)
            else:
                print(f"No data found for ID: {id}")

    if studentInfo is not None:
        if 10 < counter < 20:
            modeType = 2

            # Membuka web BI dan menghentikan deteksi wajah
            webbrowser.open("https://www.bi.go.id/id/default.aspx")  
            web_opened = True  # Set flag agar loop berhenti setelah web terbuka

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



    # Tampilkan gambar dengan overlay dari kamera
    cv2.imshow("Face Attendance", imgBackground)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepas kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
