# -*- coding: utf-8 -*-
"""

@author: Nilson
"""

#%%

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "img/"

#%%

img = cv2.imread(os.path.join(path,"ic1.jpg"), cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

#%%

print("Loading images...")

imgs = []
for image_name in os.listdir(path):
    curr_img = cv2.imread(path + image_name)
    imgs.append(curr_img)

#%%

print("Stitching...")

stitcher = cv2.Stitcher_create()
(status, result) = stitcher.stitch(imgs)

result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

plt.imshow(result_rgb)
plt.show()


#%%

cv2.imwrite('ic.jpg', result)

# %%
