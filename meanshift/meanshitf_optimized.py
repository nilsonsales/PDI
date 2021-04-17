# -*- coding: utf-8 -*-
"""

@author: Nilson
"""

#%%
import numpy as np
import cv2 as cv


#%%

# function to display the coordinates of
# of the points clicked on the image 
position_list = []

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        position_list.append((x, y))


#%%

video_path = 'dog.mp4'

vidcap = cv.VideoCapture(video_path)
ret, frame = vidcap.read()

# clear the position list
position_list.clear()

if ret == True:
    #while 0xFF & cv.waitKey(1) != ord('q'):
    # displaying image
    cv.imshow("Select the object to track", frame)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv.setMouseCallback('Select the object to track', click_event)

    cv.waitKey(0)
    cv.destroyAllWindows()


# Defining the rectangle from the mouse clicks
x, y, w, h = position_list[0][0], position_list[0][1], (position_list[1][0]-position_list[0][0]), (position_list[1][1]-position_list[0][1])

track_window = (x, y, w, h)
print(track_window)


# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while(vidcap.isOpened()):
    ret, frame = vidcap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        
        # Draw it on image
        x,y,w,h = track_window

        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
        cv.imshow('img2', img2)

        key = cv.waitKey(30)
        if key == ord('q'):
            break
    else:
        break

cv.destroyAllWindows()
cv.waitKey(1)