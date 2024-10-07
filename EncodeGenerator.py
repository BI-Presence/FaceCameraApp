import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Inisialisasi Firebase
cred = credentials.Certificate("D:\Magang\Bank Indonesia\ProjectFaceRecognition\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendacerealtime-7356b-default-rtdb.firebaseio.com/",
    'storageBucket': "faceattendacerealtime-7356b.appspot.com"
})

# Importing student images
folderPath = 'D:\Magang\Bank Indonesia\ProjectFaceRecognition\Images'
PathList = os.listdir(folderPath)
print(PathList)
imgList = []
studentsIds = []

# Loop melalui file gambar, ambil ID dari nama file (tanpa ekstensi)
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentsIds.append(os.path.splitext(path)[0])  # Ambil nama file tanpa ekstensi sebagai ID

    # Upload gambar ke Firebase storage
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print("Student IDs:", studentsIds)

# Fungsi untuk menemukan encoding wajah
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        
        # Cek apakah wajah terdeteksi
        if len(encodes) > 0:
            encode = encodes[0]
            encodeList.append(encode)
        else:
            print("No face found in image, skipping this image.")
    
    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)

# Pastikan bahwa setidaknya ada satu encoding yang ditemukan
if len(encodeListKnown) == 0:
    raise ValueError("No faces found in any images.")

print("Encoding Complete")

# Gabungkan encoding dan student IDs
encodeListKnownIds = [encodeListKnown, studentsIds]

# Simpan hasil encoding ke file
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownIds, file)

print("File Saved")
