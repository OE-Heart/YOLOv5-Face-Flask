import cv2
import requests

# 将图像以jpg编码，并转换为字节流
def get_img_bytes(img):  
    img_str = cv2.imencode('.jpg',img)[1].tobytes() if img is not None else None
    return img_str

def main():  
    img = cv2.imread("images/test.jpg")
    bFrame = get_img_bytes(img)
    request_input = {'image': bFrame}
    result = requests.post('http://127.0.0.1:7000/detect_one', files=request_input).json()
    
    if result['success']:
        id = result['id']
        category = result['category']
        points = result['points']
        print(id)
        print(category)
        print(points)
    else:
        print("failed")
    

if __name__ == "__main__":
    main()