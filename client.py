import cv2
import requests

# 将图像以jpg编码，并转换为字节流
def get_img_bytes(img):  
    img_str = cv2.imencode('.jpg',img)[1].tobytes() if img is not None else None
    return img_str

# 定义工具方法，在原始图像上画框
def plot_one_box(x, img, color=None, label="person", line_thickness=None):
    """ 画框,引自 YoLov5 工程.
    参数: 
        x:      框， [x1,y1,x2,y2]
        img:    opencv图像
        color:  设置矩形框的颜色, 比如 (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def main():  
    img = cv2.imread("images/test.jpg")
    bFrame = get_img_bytes(img)
    request_input = {'image': bFrame}
    result = requests.post('http://127.0.0.1:7000/infer', files=request_input).json()
    
    if result['success']:
        cv2.imwrite('result.jpg', result["img"])
        # boxs = result["box"] 
        # confs = result["conf"]
        # ids = result["classid"]
    else:
        print("failed")
    

if __name__ == "__main__":
    main()