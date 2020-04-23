import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import zoom

ori = '/data/datasets/ADNI_NoBack/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.npy'
MRI = '../ADNIP_NoBack/ADNI_002_S_0295_MR_MPR__GradWarp__B1_Correction__N3__Scaled_2_Br_20081001114556321_S13408_I118671.npy'
MRI = np.load(MRI)
ori = np.load(ori)

print(MRI.shape)

plt.subplot(1, 3, 1)
plt.imshow(ori[80, :, :], cmap='gray', vmin=-1, vmax=2.5)
plt.title('1.5T')
plt.subplot(1, 3, 2)
plt.imshow(MRI[80, :, :], cmap='gray', vmin=-1, vmax=2.5)
plt.title('1.5T*')
plt.subplot(1, 3, 3)
plt.imshow(MRI[80, :, :]-ori[80, :, :], cmap='gray', vmin=-1, vmax=2.5)
plt.title('mask')
plt.show()