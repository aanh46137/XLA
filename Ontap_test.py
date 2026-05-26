import cv2
import numpy as np
import matplotlib.pyplot as plt

#Câu 1: Lược đồ Histogram
def histogram(img):
    h = [0]*256
    for x in img:
        for y in x:
            h[y - 1] += 1
    x = np.arange(0,256)
    plt.bar(x,h)
    plt.show()

img = cv2.imread('anhtest.png', cv2.IMREAD_GRAYSCALE)
histogram(img)

#Câu 2: Tăng độ sáng
c = int(input('Nhập độ sáng muốn tăng: '))
img_bright = np.clip(img.astype(int) + c, 0, 255).astype(np.uint8)

# img_bright = np.copy(img).astype(int)
# img_bright += c
#
# for x in range(len(img_bright)):
#     for y in range(len(img_bright[0])):
#         if img_bright[x][y] >= 255:
#             img_bright[x][y] = 255
#         if img_bright[x][y] <= 0:
#             img_bright[x][y] = 0


#Câu 3: Phép đóng và mở
# Tạo phần tử cấu trúc B (Ma trận vuông toàn số 1)
kernel = np.ones((5, 5), np.uint8)

# kernel = np.array([[1, 0, 0, 0, 1],
#                    [0, 1, 0, 1, 0],
#                    [0, 0, 1, 0, 0],
#                    [0, 1, 0, 1, 0],
#                    [1, 0, 0, 0, 1]], dtype=np.uint8)


# Áp dụng phép Mở
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Áp dụng phép Đóng
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#Câu 4: Phát hiện biên
def detect_edges(gray_image):
    # 1. Phương pháp Canny
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blur, threshold1=50, threshold2=150)

    # 2. Phương pháp Sobel
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Hướng ngang
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Hướng dọc
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    # 3. Phương pháp Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.abs(laplacian))

    return canny, sobel, laplacian

canny, sobel, laplacian = detect_edges(img)

plt.figure(figsize=(18, 9))

# 1. Ảnh gốc
plt.subplot(2, 4, 1)
plt.title("Ảnh gốc")
plt.imshow(img, cmap='gray')
plt.axis('off')

# 2. Ảnh tăng độ sáng
plt.subplot(2, 4, 2)
plt.title("Ảnh tăng độ sáng")
plt.imshow(img_bright, cmap='gray')
plt.axis('off')

# 3. Phép mở
plt.subplot(2, 4, 3)
plt.title("Phép Mở (Opening)")
plt.imshow(opening, cmap='gray')
plt.axis('off')

# 4. Phép đóng
plt.subplot(2, 4, 4)
plt.title("Phép Đóng (Closing)")
plt.imshow(closing, cmap='gray')
plt.axis('off')

# 5. Biên Canny
plt.subplot(2, 4, 5)
plt.title("Biên Canny")
plt.imshow(canny, cmap='gray')
plt.axis('off')

# 6. Biên Sobel
plt.subplot(2, 4, 6)
plt.title("Biên Sobel")
plt.imshow(sobel, cmap='gray')
plt.axis('off')

# 7. Biên Laplacian
plt.subplot(2, 4, 7)
plt.title("Biên Laplacian")
plt.imshow(laplacian, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
