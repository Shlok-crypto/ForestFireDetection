import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imutils
import VideoSaving as vs
import os
import smtplib
import datetime
from twilio.rest import Client

#TrackBar
def temps(x):
    pass

cv.namedWindow("bars")   # Default Values for Fire
cv.createTrackbar("up_hue","bars", 42, 180, temps)
cv.createTrackbar("low_hue","bars", 0, 180, temps)
cv.createTrackbar("up_saturation","bars", 255, 255, temps)
cv.createTrackbar("low_saturation","bars", 0, 255, temps)
cv.createTrackbar("up_value","bars", 255, 255, temps)
cv.createTrackbar("low_value","bars", 255, 225, temps)


"""
    Initialing the Frame
"""
#capture = cv.VideoCapture("FireVideo.mp4")
capture = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc('m','p','4','v')
                                                       #W   #H
outframe = vs.VideoFileSaving("FireDetection",Winsize=(432,768),fourcc=fourcc)
dir = r"C:/Users/Linh/Desktop/True_Fire/ROI"
counter = 0

while True:
# Reading Video Frame
    _, frame = capture.read()

#resize the Framea
    w = int(frame.shape[1]*(60/100))
    h = int(frame.shape[0]*(60/100))
    dimensions = (w,h)
    frame = (cv.resize(frame,dimensions,interpolation= cv.INTER_NEAREST))
    #frame = imutils.resize(frame,width=640)
    FrameCopy = frame.copy()

    #print(frame.shape)
    #frame = cv.flip(frame,1)

# BGR TO HSV
    hsvFrame = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

#Getting the HSV values for masking the Fire
    upper_hue = cv.getTrackbarPos("up_hue", "bars")
    upper_saturation = cv.getTrackbarPos("up_saturation", "bars")
    upper_value = cv.getTrackbarPos("up_value", "bars")
    lower_value = cv.getTrackbarPos("low_value", "bars")
    lower_hue = cv.getTrackbarPos("low_hue", "bars")
    lower_saturation = cv.getTrackbarPos("low_saturation", "bars")

   # thresh = cv.getTrackbarPos("thresh", "bars")

    lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
    upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

# Creating mask of the ROI(fire)
    mask = cv.inRange(hsvFrame, lower_hsv, upper_hsv)
    blur = cv.GaussianBlur(frame,(7,7),0)
    mask = cv.dilate(mask, None, iterations=11)

# Find Contours
    contour, hierarchies = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    for c in contour:
        if  cv.contourArea(c) < 500:
            continue
        hull = cv.convexHull(c)
        x,y,w,h = cv.boundingRect(hull)
        #cv.drawContours(mask,[hull],-1, (255,255,255),-1) # Fill the CONTOUR WHITE

    # Saving the ROI(Fire) into folder
        #cv.imwrite(dir + "/" + str(counter) + ".jpg", FrameCopy[y: y + w, x: x + h])

        cv.drawContours(frame, c, -1, (0, 255, 0), 3)  # DRAW CONTOURS
        cv.drawContours(frame, [hull], 0, (0, 0, 255), 2) # DRAW TIGHT FITTING BOUNDARY
        cv.putText(frame, "Fire Detected", (50, 20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

    # Convert masks(gray) to BGR (2 Channel to 3 Channel )
    mask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)

    # Extract the Fire From the Frame
    tempFrame = cv.bitwise_and(FrameCopy,mask)

    # Saving the Fire onto the folder
    #cv.imwrite(dir + "/" + str(counter) + ".jpg", tempFrame)
    counter += 1

    cv.imshow("Original Frame",frame)
    cv.imshow("Mask",mask)
    cv.imshow("Fire",tempFrame)
#
    if cv.waitKey(1) == ord('q'):
            break

# capture.release()
cv.destroyAllWindows()


def FireInImages():
    # Contains all the Images
    imageList = []
    dir = 'FireImages'

    for i in os.listdir(dir):
        img = cv.imread(dir + '/' + i)
        imageList.append(img)
    print("[INFO] All Images Appended into List")

    # Process the IMAGESq
    #counter = 366
    fire = False
    aveW = 0
    aveH = 0
    for j,i in enumerate(imageList):
        img = i

        # resize the Frame
        w = int(img.shape[1] * (80 / 100))
        h = int(img.shape[0] * (80 / 100))
        dimensions = (w, h)
        img = (cv.resize(img, dimensions, interpolation=cv.INTER_NEAREST))
        imgCopy = img.copy()
        # BGR TO HSV
        hsvFrame = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        lower_hsv = np.array([lower_hue, lower_saturation, lower_value])
        upper_hsv = np.array([upper_hue, upper_saturation, upper_value])

        # Creating mask of the ROI(fire)
        mask = cv.inRange(hsvFrame, lower_hsv, upper_hsv)
        mask = cv.dilate(mask, None, iterations=11)

    # Extract the Fire From the Frame

        # Find Contours
        contour, hierarchies = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        for c in contour:
            if cv.contourArea(c) < 500:
                continue

            hull = cv.convexHull(c)
            x, y, w, h = cv.boundingRect(hull)
            cv.drawContours(mask, [hull], -1, (255, 255, 255), -1)  # Fill the CONTOUR WHITE
            cv.drawContours(imgCopy, [hull], -1, (0,255,0), 2)  # Draw the CONTOUR on Images
            
            account_sid = '###############################'
            auth_token = '################################'
            client = Client(account_sid, auth_token)

            message = client.messages.create(
                body='Testing message 2 with image',
                media_url=['https://media.istockphoto.com/id/1381637603/photo/mountain-landscape.jpg?s=1024x1024&w=is&k=20&c=C9JwCd6nvW_0hmfolDgi5uq2yAqeNWwyqLgZdODGsEQ='],
                from_='+123456789', # add send number
                to='+123456789'     # add reciver number
            )

        # Saving the ROI(Fire) into folder
            #counter += 1

        fireSegmented = cv.bitwise_and(img, img, mask=mask) # Fire Segmented

        cv.imshow(f"{str(j)}Original",img)
        #cv.imshow(f"{str(j+1.4)}rriginal",cv.bilateralFilter(img, 10, 20, 20))
        cv.imshow(f"{str(j+1.2)}FireDetected",imgCopy)
        cv.imshow("Segmentaion",fireSegmented)
        
        #cv.imshow("Mask", mask)

        # Colormask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        # FireSegmentaion = cv.bitwise_and(img, Colormask)

        cv.waitKey(0)
FireInImages()
