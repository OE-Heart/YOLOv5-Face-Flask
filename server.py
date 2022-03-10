import numpy as np
import time
import cv2
from flask import Flask, jsonify, request
from model import YOLOv5

app = Flask(__name__)
det = YOLOv5()

@app.route("/infer", methods=["POST"])
def predict():
    result = {"success": False}
    if request.method == "POST":
        if request.files.get("image") is not None:
            try:
                # 得到客户端传输的图像          
                start = time.time()      
                input_image = request.files["image"].read()
                imBytes = np.frombuffer(input_image, np.uint8)
                iImage = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
                
                # 执行推理
                outs = det.infer(iImage)
                print("duration: ",time.time()-start)
 
                if (outs is None) and (len(outs) < 0):
                    result["success"] = False
                # 将结果保存为json格式
                # result["box"] = outs[0].tolist()
                # result["conf"] = outs[1].tolist()
                # result["classid"] = outs[2].tolist()
                result["img"] = outs[0].tolist()
                result['success'] = True
            
            except Exception:
                print(Exception)
        
    return jsonify(result)

if __name__ == "__main__":
    print(("* Loading yolov5 model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='127.0.0.1', port=7000)