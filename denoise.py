import cv2
import numpy as np

# 양방향 필터
# img = 변환할 이미지 , count = 노이즈 제거 양방향 필터 적용할 횟수
def _bilateralFilter(img, count=1):
    if (count <= 0):
        return img

    result_img = img
    for i in range(count):
        result_img = cv2.bilateralFilter(img, -1, 10, 5)

    return result_img

# 미디안 필터
# img = 변환할 이미지, count = 적용 횟수, _ksize = 필터 사이즈
def _medianFilter(img, count=1, _ksize=3):
    if(count <=0):
        return img
    
    result_img = img
    for i in range(count):
        result_img = cv2.medianBlur(result_img, _ksize)
    
    return result_img


def _GaussianBlur(img, count=1, ksize=(0, 0), sigmaX=1):
    return cv2.GaussianBlur(img, ksize, sigmaX)

def test_denoiser(img):
    image = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    sharpen = cv2.GaussianBlur(image, (0,0), 3)

    result = cv2.addWeighted(image, 1.5, sharpen, -0.5, 0)

    return result

def sharping(img):
    sharpe_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    """
    sharpe_filter = np.array([[-1, -1, -1, -1, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, 2, 9, 2, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, -1, -1, -1, -1]]) / 9.0
    """
    # 샤프닝 필터
    img = cv2.filter2D(img,-1,sharpe_filter)

    return img