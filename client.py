import cv2
import requests

def main():  
    img = cv2.imread("images/2.jpg")
    img_str = cv2.imencode('.jpg',img)[1].tobytes() # 将图像以jpg编码，并转换为字节流
    request_input = {'image': img_str}
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