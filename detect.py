import cv2
import os
import cvzone
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
maskClassifier = Classifier('keras_model.h5', 'labels.txt')

#Import all the waste iamges
imgWasteList = []
pathFolderWaste = "Resources/."
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    imgBackground = cv2.imread("Resources/background1.png")

    prediction = maskClassifier.getPrediction(img)
    print(prediction)
    classID = prediction[1]

    if prediction:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1], (909,127))


    #imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[0], [909, 127])

    imgBackground[148:148 + 340, 159:159 +454] = imgResize
    #Displays
    #cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    
    
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


cap.release()

cv2.destroyAllWindows()

