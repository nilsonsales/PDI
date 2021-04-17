# -*- coding: utf-8 -*-
"""

@author: Nilson
"""

#%%
import numpy as np
import cv2 as cv

#%%
# Meanshift

cap = cv.VideoCapture('dog.mp4') 
# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = 180, 23, 40, 50 # focus on the girl's face
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640,  360))


while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply meanshift to get the new location
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window

        img2 = cv.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2', img2)

        # write the video frames
        out.write(img2)
        key = cv.waitKey(30)

        if key == ord('q'):
            cv.destroyAllWindows()
            break
    else:
        break

# save the video
out.release()
cv.destroyAllWindows()
cv.waitKey(1)


#%%
# Camshift

cap = cv.VideoCapture('dog.mp4') 
# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = 400, 150, 100, 160  # focus on the dog
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output2.avi', fourcc, 20.0, (640,  360))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        # apply camshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2', img2)
        
        # write the video frames
        out.write(img2)
        key = cv.waitKey(30)

        if key == ord('q'):
            cv.destroyAllWindows()
            break
    else:
        break

# save the video
out.release()
cv.destroyAllWindows()
cv.waitKey(1)