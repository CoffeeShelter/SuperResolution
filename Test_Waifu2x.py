from denoise import Waifu2x
import urllib.request

for i in range(9):
    image_path = "samples/sample (%d).jpg"%(i + 1)
    result_path = "waifu2x/waifu2x_result (%d).jpg"%(i + 1)
    image_url = Waifu2x(image_path,'2616c350-2e60-41e0-8d95-1294368a1652')
    urllib.request.urlretrieve(image_url, result_path)
    print(result_path + " -> 저장 완료")

print("끝!")