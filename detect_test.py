import cv2
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
maskClassifier = Classifier('keras_model.h5', 'labels.txt')


while True:
    _, img = cap.read()

    prediction = maskClassifier.getPrediction(img)
  
    print(prediction)

    cv2.imshow("Image", img)
    
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


cap.release()

cv2.destroyAllWindows()


