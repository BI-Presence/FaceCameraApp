import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("D:\Magang\Bank Indonesia\ProjectFaceRecognition\serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://faceattendacerealtime-7356b-default-rtdb.firebaseio.com/"
})

ref = db.reference('worker')

data = {
    "123456":
    {
        "name" : "Aliando Syarief",
        "divisi" : "DPPP",
        "starting_year" : 2018,
        "total_attendace": 6,
        "standing" : "G",
        "year" : 4,
        "last_attendace_time" : "2024-09-27 08:04:34"
    },
     "234567":
    {
        "name" : "Zara Adhisty",
        "divisi" : "DPID",
        "starting_year" : 2019,
        "total_attendace": 6,
        "standing" : "G",
        "year" : 4,
        "last_attendace_time" : "2024-09-27 08:01:34"
    },
     "345678":
    {
        "name" : "Roslina Puspita",
        "divisi" : "DPRD",
        "starting_year" : 2017,
        "total_attendace": 6,
        "standing" : "G",
        "year" : 4,
        "last_attendace_time" : "2024-09-27 07:04:34"
    }
}

for key, value in data.items():
    ref.child(key).set(value)
