import numpy as np
import cv2
import dlib
import math
import os
from imutils import resize
from imutils import face_utils
from imutils import paths
import sys
import time

# Gets absolute path to resource for PyInstaller
def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Finds the tip of the nose and retrns the coordinates
def prediceFaceLandmarks(gray, startX, startY, endX, endY):
    shape = predictor(gray[startY:endY, startX:endX], rect)
    shape = face_utils.shape_to_np(shape)
    i=0
    avgx = avgy = 0
    tx = ty = 0
    for (xp, yp) in shape:
            xp = xp + startX
            yp = yp + startY
            i = i + 1
            # Checking Landmark numbers for nose
            if i>=28 and i<=36:
                    avgx = avgx + xp
                    avgy = avgy + yp
                    cv2.circle(frame, (xp, yp), 1, (0, 0, 255), -1)
            # Checking Landmark numbers for tip of nose
            if i==31:
                tx = xp
                ty = yp
    # Average position of nose can also be taken
    #avgx = (int)(avgx/9)
    #avgy = (int)(avgy/9)
    avgx = tx
    avgy = ty
    cv2.circle(frame, (avgx, avgy), 1, (0, 255, 255), -1)
    return avgx, avgy

# Generates Folder Structure for output
def classifyAngles(inFrame, yaw, pitch, currentOutputPath, ctr):
    
    if not os.path.exists(currentOutputPath + "/Y+15/"):
        os.mkdir(currentOutputPath + "/Y+15/")
    if not os.path.exists(currentOutputPath + "/Y+30/"):
        os.mkdir(currentOutputPath + "/Y+30/")
    if not os.path.exists(currentOutputPath + "/Y+45/"):
        os.mkdir(currentOutputPath + "/Y+45/")
    if not os.path.exists(currentOutputPath + "/Y-15/"):
        os.mkdir(currentOutputPath + "/Y-15/")
    if not os.path.exists(currentOutputPath + "/Y-30/"):
        os.mkdir(currentOutputPath + "/Y-30/")
    if not os.path.exists(currentOutputPath + "/Y-45/"):
        os.mkdir(currentOutputPath + "/Y-45/")
    if not os.path.exists(currentOutputPath + "/P+15/"):
        os.mkdir(currentOutputPath + "/P+15/")
    if not os.path.exists(currentOutputPath + "/P-15/"):
        os.mkdir(currentOutputPath + "/P-15/")
    if not os.path.exists(currentOutputPath + "/Others/"):
        os.mkdir(currentOutputPath + "/Others/")

    if yaw >= -45 and yaw <= 45 or pitch >= -15 and pitch <= 15:
        if yaw >= 0 and yaw <= 15:
            cv2.imwrite(currentOutputPath + "/Y+15/" + str(ctr) + ".jpg", inFrame)
        elif yaw > 15 and yaw <= 30:
            cv2.imwrite(currentOutputPath + "/Y+30/" + str(ctr) + ".jpg", inFrame)
        elif yaw > 30 and yaw <= 45:
            cv2.imwrite(currentOutputPath + "/Y+45/" + str(ctr) + ".jpg", inFrame)
        elif yaw < 0 and yaw >= -15:
            cv2.imwrite(currentOutputPath + "/Y-15/" + str(ctr) + ".jpg", inFrame)
        elif yaw < -15 and yaw >= -30:
            cv2.imwrite(currentOutputPath + "/Y-30/" + str(ctr) + ".jpg", inFrame)
        elif yaw < -30 and yaw >= -45:
            cv2.imwrite(currentOutputPath + "/Y-45/" + str(ctr) + ".jpg", inFrame)

        if pitch >= 0 and pitch <= 15:
            cv2.imwrite(currentOutputPath + "/P+15/" + str(ctr) + ".jpg", inFrame)
        elif pitch < 0 and pitch >= -15:
            cv2.imwrite(currentOutputPath + "/P-15/" + str(ctr) + ".jpg", inFrame)
    else:
        cv2.imwrite(currentOutputPath + "/Others/" + str(ctr) + ".jpg", inFrame)

# Path for Model Structure
prototxtLocation = resource_path("deploy.prototxt.txt")
# Path for pre trained weight values 
caffeModel = resource_path("res10_300x300_ssd_iter_140000.caffemodel")
# Path for Facial Landmarks Predictor
predictor = dlib.shape_predictor(resource_path("shape_predictor_68_face_landmarks.dat"))
detector = dlib.get_frontal_face_detector()
thresholdConfidence = 0.5

inputPath = input("Enter Input Images Path : ")
inputPath = inputPath.replace('\\','/')

outputPath = input("Enter Output Images Path : ")
outputPath = outputPath.replace('\\','/')

processingWidth = int(input("Enter width of images to be used for processing (E.g. 300) : "))

faceFlag = False
noseFlag = False

# Load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxtLocation, caffeModel)

# initialize the video stream and allow the cammera sensor to warmup
# print("[INFO] starting video stream...")
#cap = cv2.VideoCapture(0)

# Count to store number of images
ctr = 0

# Create output path if not already present
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Loop over the images in the input directory
for imagePath in paths.list_images(inputPath):
        imagePath = imagePath.replace('\\','/')
        inputDirectory = inputPath[inputPath.rfind("/")+1:] + "/" + imagePath.replace(inputPath + "/", "")
        currentOutputPath = outputPath + "/" + inputDirectory[:inputDirectory.rfind("/")]
        print("\n" + imagePath)
        print(currentOutputPath)
        if not os.path.exists(currentOutputPath):
            os.makedirs(currentOutputPath)
        # Grab the image and resize it for proper detection
        # Realtime video input can also be used
        #ret, inFrame = cap.read()
        inFrame = cv2.imread(imagePath, 1)
        frame = resize(inFrame,width=processingWidth)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        box = None
        conf = 0

        # loop over the detections
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < thresholdConfidence:
                        continue
                
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                if confidence > conf:
                    conf = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        # Consider only the face with maximum confidence to eliminate false positives
        if conf >= thresholdConfidence:
                (startX, startY, endX, endY) = box.astype("int")
                    
                rectCx = (int)((startX + endX)/2)
                rectCy = (int)((startY + endY)/2)

                y = startY - 10 if startY - 10 > 10 else startY + 10
                # Placing rectangle over face
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                
                rects = detector(gray[startY:endY, startX:endX], 1)

                if len(rects) > 0:
                        noseFlag = True
                else:
                        noseFlag = False
                for (i, rect) in enumerate(rects):
                    if i>0:
                        break
                    avgx, avgy = prediceFaceLandmarks(gray, startX, startY, endX, endY)
                
                # Finding angles only when structure of nose can be estimated
                if noseFlag:
                        yaw = math.degrees(math.asin(((endX - startX) / 2 if abs(rectCx - avgx) > (endX - startX) / 2  else rectCx - avgx) * 2 / (endX - startX)))
                        pitch = math.degrees(math.asin((rectCy - avgy) * 2 / (endY - startY))) + (endY - startY) * 0.05
                        print("yaw = {:.2f}, ".format(yaw) + "pitch = {:.2f}".format(pitch))
                        
                        classifyAngles(inFrame, yaw, pitch, currentOutputPath, ctr)

                        if avgx >= rectCx-((int)((endX - startX) * 0.15)) and avgx <= rectCx+((int)((endX - startX) * 0.15)):
                                text = "Frontal {:.2f}%".format(conf * 100)
                        else:
                                text = "Non Frontal {:.2f}%".format(conf * 100)
                else:
                        cv2.imwrite(currentOutputPath + "/Others/" + str(ctr) + ".jpg", inFrame)
                        text = "Non Frontal {:.2f}%".format(conf * 100)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        ctr = ctr + 1

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break
print("Number of Images Processed = " + str(ctr))
# do a bit of cleanup
#cap.release()
cv2.destroyAllWindows()

input("\nPress ENTER to continue...")
print("\nDeveloped by Ashish Ranjan, e-mail: whoashishranjan@gmail.com")
time.sleep(3)
