import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
import time

from denoise import _bilateralFilter
from denoise import test_denoiser
from denoise import _medianFilter
from denoise import sharping
from denoise import _GaussianBlur

MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"


def preprocess_image(image_path):
    image = tf.image.decode_image(tf.io.read_file(image_path))
    # PNG 파일일 경우 Alpha 채널을 제거 한다.
    # 3채널 이미지
    if image.shape[-1] == 4:
        image = image[..., :-1]

    image = tf.cast(image, tf.float32)

    # [batch_size, height, width, channel]
    return tf.expand_dims(image, 0)

"""
IMAGE_PATH = "sample (1).jpg"
tf_image = preprocess_image(IMAGE_PATH)

# 3차원 uint8(0~255) 이미지로 변환
cv_image = tf.squeeze(tf_image)
cv_image = tf.cast(tf.clip_by_value(cv_image, 0, 255), tf.uint8).numpy()
cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

# 노이즈 제거
# cv_image = _bilateralFilter(cv_image, 20)
cv_image = test_denoiser(cv_image)

# 4차원 float32 이미지로 변환
tf_image = tf.expand_dims(cv_image, 0)
tf_image = tf.cast(tf_image, tf.float32)

print("모델 로딩...")
# 초해상도 변환 모델 해상도 4배 증가
sr_model = hub.load(MODEL_PATH)

print("이미지 변환중....")
sr_result_image = sr_model(tf_image)
print("이미지 변환 완료!")

# 3차원 uint8(0~255) 이미지로 변환
sr_result_image = tf.squeeze(sr_result_image)
sr_result_image = tf.cast(tf.clip_by_value(sr_result_image, 0, 255), tf.uint8).numpy()

# sr_result_image = _bilateralFilter(sr_result_image, 20)
sr_result_image = test_denoiser(sr_result_image)
"""

def test_sr(image_path, model):
    tf_image = preprocess_image(image_path)

    # 3차원 uint8(0~255) 이미지로 변환
    cv_image = tf.squeeze(tf_image)
    cv_image = tf.cast(tf.clip_by_value(cv_image, 0, 255), tf.uint8).numpy()
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # 4차원 float32 이미지로 변환
    tf_image = tf.expand_dims(cv_image, 0)
    tf_image = tf.cast(tf_image, tf.float32)

    # 초해상도 변환 모델 해상도 4배 증가
    sr_model = model

    print("이미지 변환중....")
    sr_result_image = sr_model(tf_image)
    print("이미지 변환 완료!")

    # 3차원 uint8(0~255) 이미지로 변환
    sr_result_image = tf.squeeze(sr_result_image)
    sr_result_image = tf.cast(tf.clip_by_value(sr_result_image, 0, 255), tf.uint8).numpy()

    sr_result_image = test_denoiser(sr_result_image)
    
    return sr_result_image

model = hub.load(MODEL_PATH)

_start = time.time()
for i in range(9):
    start = time.time()
    IMAGE_PATH = "sample (%d).jpg"%(i+1)
    RESULT_PATH = "sr_testDenoiser (%d).jpg"%(i+1)
    result = test_sr(IMAGE_PATH, model)
    cv2.imwrite(RESULT_PATH, result)
    print(RESULT_PATH + " -> 저장 완료")
    print("걸린 시간 : %f"%(time.time() - start))

print("끝! (%f)"%(time.time() - _start))