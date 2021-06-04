import cv2
from matplotlib import pyplot as plt

from denoise import sharping

# 히스토그램 구하기
def CalcGrayHist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist

# 히스토그램 그리기
def ShowHistogram(hist):
    plt.hist(hist.ravel(), 256, [0, 256])
    plt.show()

# 히스토그램 중간 값
def GetMedianByHist(hist):
    maxIndex = 0
    minIndex = 0
    max = 0
    min = 0
    medianValue = 0

    for i in range(255):
        if (max < hist[i, 0]):
            max = hist[i, 0]
            maxIndex = i

        if (min > hist[i, 0]):
            min = hist[i, 0]
            minIndex = i

    if (maxIndex > minIndex):
        medianValue = (maxIndex - minIndex) / 2

    else:
        medianValue = (minIndex - maxIndex) / 2

    return medianValue

img = cv2.imread("sample (6).jpg")
img = sharping(img)

cv2.imwrite("sharp.jpg",img)