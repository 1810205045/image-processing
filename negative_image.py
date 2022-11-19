import cv2
from matplotlib import pyplot as plt
import numpy as np

image = cv2.imread('images\\resim.jpg', 0)

l_value = np.max(image)
convert2negative = l_value - image

cv2.imshow("image1", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

show_diff = np.hstack((image, convert2negative))

plt.imshow(show_diff, cmap="gray")
plt.show()
