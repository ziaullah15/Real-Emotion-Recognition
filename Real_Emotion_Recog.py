import cv2
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

while True:
    ret,frame = cap.read()
    if frame is None:
         raise ValueError('Unable to get a frame!')
    result = DeepFace.analyze(frame, actions = ['emotion'])
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    
    for(x,y,a,b) in faces:
        cv2.rectangle(frame, (x,y), (x+a , y+b), (0,255,0), 2)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'], (50,50), font, 1, (0,0,255),2, cv2.LINE_4);
    cv2.imshow('Demo video' , frame)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
    
