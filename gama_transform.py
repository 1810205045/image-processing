import cv2
from matplotlib import pyplot as plt
import numpy as np


def gama_transform(image, value):
    s = np.array(255*(image/255)**value)
    return s.astype(np.uint8)


image = cv2.imread('images\\resim.jpg', 0)
print(image.shape, image.dtype)

gama_adjusted_image = gama_transform(image, 2.8)

show_diff = np.vstack((image, gama_adjusted_image))

plt.imshow(show_diff, cmap="gray")
plt.show()