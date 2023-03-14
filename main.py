import numpy as np
import matplotlib.pyplot as plt
import cv2

flat_chess = cv2.imread('../DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2RGB)
# plt.imshow(flat_chess)
# plt.show()

gray_flat_chess = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_flat_chess, cmap='gray')
# plt.show()

real_chess = cv2.imread('../DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess, cv2.COLOR_BGR2RGB)
# plt.imshow(real_chess)
# plt.show()

gray_real_chess = cv2.imread('../DATA/real_chessboard.jpg')
gray_real_chess = cv2.cvtColor(gray_real_chess, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_real_chess, cmap='gray')
# plt.show()

"""
anywhere in the cornerharris image we have a 1% of max value, this a corner detected
we then set that pixel to red
cant detect corners on the edge, because nothing to compare to on outside, could add custom edge
"""
# use gray image to determine the corners
# gray = np.float32(gray_flat_chess)
# dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
# dst = cv2.dilate(dst, None) #does have anything to do with cornerHarris, just improves results

# merge the corners into the real chessboard image
# flat_chess[dst>0.01*dst.max()] = [255,0,0]
# plt.imshow(flat_chess)
# plt.show()

"""more complex image"""
# # use gray image to determine the corners
# gray = np.float32(gray_real_chess)
# dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
# dst = cv2.dilate(dst, None) #does have anything to do with cornerHarris, just improves results
#
# # merge the corners into the real chessboard image
# real_chess[dst>0.01*dst.max()] = [255,0,0]
# plt.imshow(real_chess)
# plt.show()

"""Shi-tomasi algorithm"""
# second parameter is number of corners to detect, you can put -1 to detect all corners
# corners = cv2.goodFeaturesToTrack(gray_flat_chess, 64, 0.01, 10)
# corners = np.int0(corners) # convert to integers
# # flatten array, since shi-tomasi doesnt autmatically draw points
# for i in corners:
#     x,y = i.ravel()
#     cv2.circle(flat_chess, (x,y), 3, (255,0,0), -1)
#
# plt.imshow(flat_chess)
# plt.show()

# second parameter is number of corners to detect, you can put -1 to detect all corners
corners = cv2.goodFeaturesToTrack(gray_real_chess, 256, 0.01, 10)
corners = np.int0(corners) # convert to integers
# flatten array, since shi-tomasi doesnt autmatically draw points
for i in corners:
    x,y = i.ravel()
    cv2.circle(real_chess, (x,y), 3, (255,0,0), -1)

plt.imshow(real_chess)
plt.show()