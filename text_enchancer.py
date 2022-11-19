import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('images\\paper.jpg', 0)

ret, thresh1 = cv2.threshold(image, 170, 255, cv2.THRESH_BINARY)

show_diff = np.hstack((image, thresh1))

plt.imshow(show_diff, cmap="gray")
plt.show()