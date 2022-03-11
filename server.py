import numpy as np
import time
import cv2
from flask import Flask, jsonify, request
from model import YOLOv5_face

app = Flask(__name__)
face_det = YOLOv5_face()

@app.route("/detect_face_one", methods=["POST"])
def detect_face_one():
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
                outs = face_det.infer(iImage)
                print("duration: ",time.time()-start)
 
                if (outs is None) and (len(outs) < 0):
                    result["success"] = False
                else:
                    # 将结果保存为json格式
                    result['id'] = outs[0]
                    result['category'] = outs[1]
                    result['points'] = outs[2]
                    result['success'] = True
            except Exception:
                print(Exception)
    
    return jsonify(result)

@app.route("/detect_batch", methods=["POST"])
def detect_batch():
    result = {"success": False}

    return jsonify(result)


if __name__ == "__main__":
    print(("* Loading yolov5 model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(host='127.0.0.1', port=7000)