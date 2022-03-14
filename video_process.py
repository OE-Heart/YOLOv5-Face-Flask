import cv2
import time
from model import YOLOv5_face

face_det = YOLOv5_face()

fcap = cv2.VideoCapture('video/demo.mp4')

# 获取视频帧的宽
w = fcap.get(cv2.CAP_PROP_FRAME_WIDTH)
# 获取视频帧的高
h = fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# 获取视频帧的帧率
fps = fcap.get(cv2.CAP_PROP_FPS)

# 获取VideoWriter类实例
writer = cv2.VideoWriter('video/output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (int(w), int(h)))

# 判断是否正确获取VideoCapture类实例
while fcap.isOpened():

    # 获取帧画面
    success, frame = fcap.read()

    cnt = 0
    start = time.time()
    while success:
        success, frame = fcap.read()
        # do something in here
        frame = face_det.infer(frame)
        # cv2.imwrite('video/result' + str(cnt) + '.jpg', frame)
        # 保存帧数据
        writer.write(frame)

        cnt += 1
        if (cnt % 100 == 0):
            print(cnt, "done")
    
end = time.time()
print('time cost', end - start,'s')
    
# 释放VideoCapture资源
fcap.release()
# 释放VideoWriter资源
writer.release()