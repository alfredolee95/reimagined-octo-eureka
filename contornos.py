import cv2
import numpy as np
cap =cv2.VideoCapture(0)
areaProm=[]
avgVal=10
ret=True
while ret==True:
    _, frame = cap.read()
    lower_color =np.array([50, 00, 00])
    upper_color= np.array([90, 255, 80])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    blurred_frame = cv2.GaussianBlur(mask, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred_frame, cv2.CV_64F)
    canny = cv2.Canny(blurred_frame, 50, 300)
    _, ctns, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, ctns, -1, (0,0,255), 2)
    print('Número de contornos encontrados: ', len(ctns))

    #texto = 'Contornos encontrados: '+ str(len(ctns))
    #cv2.putText(frame, texto, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 0, 0), 1)
    cv2.imshow('Imagezn', frame)
    cv2.imshow('Imagen', laplacian)
    key = cv2.waitKey(1)
    if (key == 27):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()