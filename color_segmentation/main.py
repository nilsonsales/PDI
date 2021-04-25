'''
author: Nilson Sales
'''
#%%
# import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

#%%
fish = cv2.imread("clownfish.jpg")
fish = cv2.cvtColor(fish, cv2.COLOR_BGR2RGB)

plt.imshow(fish)
plt.show()

#%%
# Make coloroured 3D scatterplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# split the image and set up the 3D plot
r, g, b = cv2.split(fish)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

# reshaping and normalization 
pixel_colors = fish.reshape((np.shape(fish)[0]*np.shape(fish)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

#%%
# Visualizing in HSV Color Space
hsv_fish = cv2.cvtColor(fish, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(hsv_fish)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

#%%
# Picking Out a Range
light_orange = (1, 180, 125)
dark_orange = (19, 255, 255)

# Threshold the clownfish
mask = cv2.inRange(hsv_fish, light_orange, dark_orange)
result = cv2.bitwise_and(fish, fish, mask=mask)

plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Original + Mask")
plt.show()

#%%
# Getting the white/blue colours as well
light_white = (0, 0, 180)
dark_white = (145, 45, 255)

mask_white = cv2.inRange(hsv_fish, light_white, dark_white)
result_white = cv2.bitwise_and(fish, fish, mask=mask_white)

plt.subplot(1, 2, 1)
plt.imshow(mask_white, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result_white)
plt.show()

#%%
# Combining the two masks
final_mask = mask + mask_white

final_result = cv2.bitwise_and(fish, fish, mask=final_mask)

plt.subplot(1, 2, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(final_result)
plt.show()

#%%
# Apply a blur filter

blur = cv2.GaussianBlur(final_result, (7, 7), 0)
plt.imshow(blur)
plt.show()

#%%
# Save image

cv2.imwrite("clownfish-segmented.jpg", cv2.cvtColor(blur, cv2.COLOR_RGB2BGR))