import cv2
import matplotlib.pyplot as plt

from subpixel_edges import subpixel_edges
import numpy as np
np.bool = bool
# (optional) 
# help(subpixel_edges) 

img = cv2.imread("a.jpg")
img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
edges = subpixel_edges(img_gray, 25, 0, 2)

plt.imshow(img)
plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
plt.show()