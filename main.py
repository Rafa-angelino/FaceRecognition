from datetime import datetime
import os
import cv2 
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognitionworkers-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognitionworkers.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

imgBackground = cv2.imread('Resources/background.png')

#importando as imagens mode dentro de uma lista 
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

#importando o arquivo encoding
print("Carregando arquivo encoding")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds =  pickle.load(file)
file.close()
encodeListKnown, workersId = encodeListKnownWithIds
#print(workersId)
print("arquivo encode carregado")


modeType = 0
counter = 0
id=-1
imgWorker = []
while True:
    success, img = cap.read()
    
    imgS = cv2.resize(img,(0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
    
    
    
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    
    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            #print('Matches', matches)
            #print('faceDis', faceDis)

            #pega o index do rosto detectado no array faceDis
            matchIndex = np.argmin(faceDis)
            #print("matchIndex: ", matchIndex)
            
            #se o valor do index estiver na lista matches ele retorna 'rosto detectado'
            if matches[matchIndex]:
                #print(workersId[matchIndex])
                
                #desenha o retangulo no rosto
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = workersId[matchIndex]
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275,400))
                    cv2.imshow("Reconhecimento de face", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1
        
        if counter !=0:
            if counter == 1:
                #puxa os dados do database realtime
                workerInfo = db.reference(f'Workers/{id}').get()
                
                #pega a imagem do storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgWorker = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                
                #Atualizando a data de trabalho comparecido
                datetimeObject = datetime.strptime(workerInfo['last_attendance_time'],
                                                    "%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                #print(secondsElapsed)
                
                if secondsElapsed > 30:
                
                    ref = db.reference(f'Workers/{id}')
                    workerInfo['total_attendance']+= 1
                    ref.child('total_attendance').set(workerInfo['total_attendance']) #atualiza os valor referente ao dia do trabalho comparecido no banco de dados
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                
            if modeType != 3:
                    
                if 10<counter<20:
                    modeType = 2
                
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                    
                if counter <= 10:    
                    cv2.putText(imgBackground, str(workerInfo['total_attendance']),(861,125),
                                cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
                    
                    cv2.putText(imgBackground, str(workerInfo['Major']),(1006,550),
                                cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)    
                    cv2.putText(imgBackground, str(id),(1006,493),
                                cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
                    cv2.putText(imgBackground, str(workerInfo['standing']),(910,625),
                                cv2.FONT_HERSHEY_COMPLEX,0.6,(100,100,100),1)
                    cv2.putText(imgBackground, str(workerInfo['year']),(1025,625),
                                cv2.FONT_HERSHEY_COMPLEX,0.6,(100,100,100),1)
                    cv2.putText(imgBackground, str(workerInfo['starting_year']),(1125,625),
                                cv2.FONT_HERSHEY_COMPLEX,0.6,(100,100,100),1)
                    
                    (w,h), _ = cv2.getTextSize(workerInfo['name'], cv2.FONT_HERSHEY_COMPLEX,1,1)
                    offset = (414-w)//2
                    cv2.putText(imgBackground, str(workerInfo['name']),(808+offset,445),
                                cv2.FONT_HERSHEY_COMPLEX,1,(50,50,50),1)
                    
                    #print(imgBackground)
                    imgBackground[175:175 + 216, 909:909 + 216]  = imgWorker
                
            counter += 1
            if counter>=20:
                counter=0
                modeType =0
                workerInfo = []
                imgWorker = []
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0                
    
    cv2.imshow("webcam", img)
    cv2.imshow('Rosto reconhecido', imgBackground)
    cv2.waitKey(1)