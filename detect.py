import cv2
import torch
from yolov5 import detect  # yolov5 detection 함수

# video 파일 읽어오기
cap = cv2.VideoCapture('video.mp4')

# 프레임 크기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 프레임 레이트
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# 모델 초기화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)

# 큐 초기화
frame_queue = queue.Queue()
detection_queue = queue.Queue()
results_queue = queue.Queue()

# frame_queue에 프레임 추가
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # frame_queue에 프레임 추가
    frame_queue.put(frame)

    # detection 수행 후 detection_queue에 추가
    detections = detect(model, device, frame)
    detection_queue.put(detections)

# 작업이 끝나면 큐를 닫음
cap.release()
frame_queue.join()
detection_queue.join()



"""
main() 함수에서 frame_queue와 detection_queue에 프레임과 
detection을 추가하는 부분. 
OpenCV를 사용하여 비디오 파일에서 프레임을 읽어오고, 
YOLOv5를 사용하여 detection을 수행
cv2.VideoCapture() 비디오 파일을 읽어와 프레임을 가져온 후,
frame_queue에 추가, 이후 detect() 함수를 사용하여 
YOLOv5를 이용하여 detection을 수행하고, 
detection_queue에 추가.
이렇게 frame_queue와 detection_queue에 프레임과 detection을 추가하면, 
main() 함수에서 멀티스레딩
"""