from scipy.datasets import face
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_opening,
    binary_erosion,
)
import matplotlib.pyplot as plt
import numpy as np

arr = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)

image = np.load("wires6.npy.txt")

labeled = label(image)

for lbl in range(1, labeled.max() + 1):
    count = label(binary_erosion(labeled == lbl)).max()
    
    if count > 1:
        print(f"{lbl} провод разделяется {count} раз(а)")
    else:
        print(f"{lbl} провод не разделен")
    
plt.imshow(label(image))
plt.show()