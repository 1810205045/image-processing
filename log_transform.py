import cv2
from matplotlib import pyplot as plt
import numpy as np


def log_transform(r, c):
    r = r.astype(float)
    s = c*np.log(1 + r)
    return s.astype(np.uint8)


def re_scale(image):
    s = image.astype(float)
    s -= np.min(s)
    s /= np.max(s)
    return (s*255).astype(np.uint8)


image = cv2.imread('images\\resim.jpg', 0)
print(image.shape, image.dtype)

log_image = log_transform(image, c=1)

print("Before rescale")
print("Log min: ", np.min(log_image))
print("Log max: ", np.max(log_image))

log_image = re_scale(log_image)

print("After rescale")
print("Log min: ", np.min(log_image))
print("Log max: ", np.max(log_image))

show_diff = np.vstack((image, log_image))

print("Original image: ", image.shape)
print("Log image: ", log_image.shape)
print("Stacked image: ", show_diff.shape)

plt.imshow(show_diff, cmap="gray")
plt.show()