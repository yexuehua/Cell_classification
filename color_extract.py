#import SimpleITK as sitk
import cv2
import numpy as np

path = r"C:\Users\212774000\Desktop\vuebox\US_test.png"
img = cv2.imread(path)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img_h = img_hsv

# lower_red = np.array([160, 60, 60])
# upper_red = np.array([180, 255, 255])
#
# lower_red2 = np.array([0, 60, 60])
# upper_red2 = np.array([10, 255, 255])

# mask_r = cv2.inRange(img_hsv, lower_red, upper_red)
# mask_r2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

lower_blue = np.array([100, 60, 60])
upper_blue = np.array([124, 255, 255])
mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

# mask = mask_r + mask_r2
mask = mask_blue

kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
mask_e = cv2.erode(mask, kernel_1)
mask_d = cv2.dilate(mask_e, kernel_2)
cv2.imshow("red",mask_d)
cv2.waitKey()
cv2.imwrite(r"C:\Users\212774000\Desktop\vuebox\blue.png",mask_d)