import matplotlib.pyplot as plt
import numpy as np
import cv2


mapp = np.load('testData/spatialMasking9.npy')
# mapp = np.load('spatialMasking9.npy')
# mapp = cv2.GaussianBlur(mapp, (11, 11), 1)
cm2 = plt.cm.get_cmap('jet')
plt.imshow(mapp,vmax = 1000, cmap = cm2)
plt.colorbar()
# plt.savefig('spatialMasking5.png', dpi=300)
plt.show()

# plt.imshow(mapp2,vmax = 600, cmap = cm2)
# plt.show()

# mapp = np.load('testData/temporalMasking.npy')
# # mapp2 = np.load('spatialMasking9.npy')
# mapp = cv2.GaussianBlur(mapp, (201, 201), 1)
# cm2 = plt.cm.get_cmap('jet')
# plt.imshow(mapp,vmax = 50, cmap = cm2)
# # plt.colorbar()
# plt.savefig('temporalMasking.png', dpi=300)
# plt.show()

