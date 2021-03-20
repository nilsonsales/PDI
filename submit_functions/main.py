# -*- coding: utf-8 -*-
"""
@author: Nilson Sales
"""

#%%
# Importing libraries

from imutils import paths
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "img/"

# %%

def subtraction(img1, img2):
    result = cv2.absdiff(img1, img2)

    plt.subplot(131); plt.title("IMG 1"); plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(132); plt.title("IMG 2"); plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.subplot(133); plt.title("Difference"); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

#%%

def logical_operations(img1, img2): 
    AND = img1 & img2
    OR = img1 | img2
    NOT = ~img1
    XOR = img1 ^ img2

    plt.subplot(221); plt.title("AND"); plt.imshow(cv2.cvtColor(AND, cv2.COLOR_BGR2RGB))
    plt.subplot(222); plt.title("OR"); plt.imshow(cv2.cvtColor(OR, cv2.COLOR_BGR2RGB))
    plt.subplot(223); plt.title("NOT"); plt.imshow(cv2.cvtColor(NOT, cv2.COLOR_BGR2RGB))
    plt.subplot(224); plt.title("XOR"); plt.imshow(cv2.cvtColor(XOR, cv2.COLOR_BGR2RGB))

    plt.show()

# %%

def union(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    result = cv2.max(img1, img2)

    plt.subplot(131); plt.title("IMG 1"); plt.imshow(img1, cmap='gray')
    plt.subplot(132); plt.title("IMG 2"); plt.imshow(img2, cmap='gray')
    plt.subplot(133); plt.title("Union"); plt.imshow(result, cmap='gray')
    plt.show()

#%%

img1 = cv2.imread(os.path.join(path,"hand1.jpg"), cv2.IMREAD_COLOR)
img2 = cv2.imread(os.path.join(path,"hand2.jpg"), cv2.IMREAD_COLOR)

subtraction(img1, img2)
logical_operations(img1, img2)
union(img1, img2)


#%%

img1 = cv2.imread(os.path.join(path,"astronaut.png"), cv2.IMREAD_COLOR)
img2 = cv2.imread(os.path.join(path,"lena.png"), cv2.IMREAD_COLOR)

subtraction(img1, img2)
logical_operations(img1, img2)
union(img1, img2)
