import cv2
from matplotlib import pyplot as plt
# 读取图片
img = cv2.imread("circle.png")
# 转成灰度图片
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化
ret, img = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)

# 显示图像
plt.imshow(img)
plt.show()
cv2.imwrite("black_white.jpg", img)

# # 填空缺
# # 原图取补得到MASK图像
# mask = 255 - img
# # 构造Marker图像
# marker = np.zeros_like(img)
# marker[0, :] = 255
# marker[-1, :] = 255
# marker[:, 0] = 255
# marker[:, -1] = 255
# # 形态学重建
# SE = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3))
# while True:
#     marker_pre = marker
#     dilation = cv2.dilate(marker, kernel=SE)
#     marker = np.min((dilation, mask), axis=0)
#     if (marker_pre == marker).all():
#         break
# dst = 255 - marker
# dst[dst == 255] = 255
# cv2.imshow("res", dst)
# cv2.imwrite('../image/bus/binary.jpg', dst)
