import os
import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognitionworkers-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognitionworkers.appspot.com"
})



#importando as imagens dos trabalhadores 
folderPath = 'Images'
pathList = os.listdir(folderPath)
#print(pathList)
imgList = []
workersId = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    workersId.append(os.path.splitext(path)[0])
    
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)
#print(workersId)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       encode = face_recognition.face_encodings(img)[0]
       encodeList.append(encode)
    return encodeList


    
print("Encoding iniciado...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, workersId]
#print(encodeListKnown)
#print(encodeListKnownWithIds)
print("Encodings completo.")

file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("arquivo salvo")    