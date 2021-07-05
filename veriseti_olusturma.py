import cv2
import numpy as np

cap = cv2.VideoCapture(0)
a=3000
while(a<4000):
    ret, frame = cap.read()

    bolge = frame[0:300, 0:250]
    img=cv2.resize(bolge, (32, 32))


    cv2.imshow('frame', bolge)

    cv2.imwrite('dataset/8/'+str(a)+'.jpg',img)

    a=a+1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

