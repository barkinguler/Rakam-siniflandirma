import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img /255.0
    
    return img

cap = cv2.VideoCapture(0)

model = load_model('model1.h5')


while True:
    
    success, frame = cap.read()

    bolge = frame[0:300, 0:250]



    img = np.asarray(bolge)
    img = cv2.resize(img, (32,32))
    img = preProcess(img)
    
    img = img.reshape(1,32,32,1)
    
    # predict
    classIndex = int(model.predict_classes(img))
    
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex, probVal)
    
    if probVal > 0.7:
        cv2.putText(frame, str(classIndex)+ "   "+ str(probVal), (50,50),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),1)

    cv2.imshow("Rakam Siniflandirma",bolge)

    if cv2.waitKey(1) & 0xFF == ord("q"): break    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    