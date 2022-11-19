import cv2
from matplotlib import pyplot as plt
import numpy as np


def re_scale(image):
    s = image.astype(float)
    s -= np.min(s)
    s /= np.max(s)
    return (s*255).astype(np.uint8)


def image_compress(image, bit_planes):
    compressed_image = np.zeros_like(image)
    for bp in bit_planes:
        compressed_image += slice_bit_plane(image, bp)
    return compressed_image

def slice_bit_plane(image, bit_plane):
    sliced_image = np.full_like(image, bit_plane)
    return np.bitwise_and(image, sliced_image)


image = cv2.imread('images\\resim.jpg', 0)

plane = [8, 16, 24]
sliced_images = []

for bit_p in plane:
    bit_img = slice_bit_plane(image, bit_p)
    bit_img2 = re_scale(bit_img)
    sliced_images.append(bit_img2)

row1 = np.hstack((image, sliced_images[0]))
row2 = np.hstack((sliced_images[1], sliced_images[2]))

table = np.vstack((row1, row2))

plt.imshow(table, cmap="gray")
plt.show()