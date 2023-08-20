import cv2
import os
import cvzone
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
maskClassifier = Classifier('keras_model.h5', 'labels.txt')
imgArrow = cv2.imread("Resources/arrow.png", cv2.IMREAD_UNCHANGED)

#Import all the waste iamges
imgWasteList = []
pathFolderWaste = "Resources/Waste"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

#Import all the waste iamges
imgBinsList = []
pathFolderWaste = "Resources/Bins"
pathList = os.listdir(pathFolderWaste)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste, path), cv2.IMREAD_UNCHANGED))

while True:
    _, img = cap.read()
    imgResize = cv2.resize(img, (454, 340))

    imgBackground = cv2.imread("Resources/background.png")

    prediction = maskClassifier.getPrediction(img)
  
    classID = prediction[1]
    print(classID)

    if classID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1], (909,127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978,320))
       
    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[0], (895,374))


    imgBackground[148:148 + 340, 159:159 +454] = imgResize
    #Displays
    #cv2.imshow("Image", img)
    cv2.imshow("Output", imgBackground)
    
    
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


cap.release()

cv2.destroyAllWindows()


    #imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[0], [909, 127])