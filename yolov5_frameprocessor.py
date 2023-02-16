import cv2
import torch
from yolov5 import detect  # yolov5 detection 함수
from multiprocessing import Process, Queue

class FrameProcessor(Process):
    def __init__(self, frame_queue, detection_queue, results_queue, device):
        super(FrameProcessor, self).__init__()
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.results_queue = results_queue
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(self.device)

    def run(self):
        while True:
            # 프레임 큐에서 프레임 가져오기
            frame = self.frame_queue.get()

            # 작업이 끝나면 큐를 닫음
            if frame is None:
                self.detection_queue.put(None)
                self.frame_queue.task_done()
                break

            # YOLOv5를 사용하여 detection 수행
            detections = detect(self.model, self.device, frame)

            # detection_queue에 결과 추가
            self.detection_queue.put(detections)

            # 결과 큐에서 결과 가져오기
            result = self.results_queue.get()

            # 작업이 끝나면 큐를 닫음
            if result is None:
                self.frame_queue.task_done()
                self.detection_queue.task_done()
                break

            # 결과 처리
            # ...

            # 큐에 task가 처리됨을 알림
            self.frame_queue.task_done()
            self.detection_queue.task_done()

if __name__ == '__main__':
    # video 파일 읽어오기
    cap = cv2.VideoCapture('video.mp4')

    # 프레임 크기
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 프레임 레이트
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # 모델 초기화
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 큐 초기화
    frame_queue = Queue(maxsize=100)
    detection_queue = Queue(maxsize=100)
    results_queue = Queue(maxsize=100)

    # 프레임 프로세서 시작
    frame_processor = FrameProcessor(frame_queue, detection_queue, results_queue, device)
    frame_processor.start()

    # frame_queue에 프레임 추가
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame_queue에 프레임 추가
        frame_queue.put(frame)

    # 작업이 끝나면 큐를 닫음
    frame_queue.put(None)

    # detection_queue에 detection 추가
    while True:
        detections = detection_queue.get()
        if detections is None:
            detection_queue.task_done()
            break

        # detection 처리
        # ...

        # 결과 큐에


"""
FrameProcessor 클래스는 프레임을 처리하는 데 사용되는 멀티프로세스 작업을 수행
이 클래스는 multiprocessing.Process 클래스를 상속하며, run() 메서드를 구현
run() 메서드는 클래스를 인스턴스화하고 start() 메서드를 호출할 때 실행

frame_queue: 프레임을 담는 큐
detection_queue: 검출된 결과를 담는 큐
results_queue: 처리된 결과를 담는 큐
device: YOLOv5 모델을 실행할 디바이스 (CPU 또는 GPU)
model: YOLOv5 모델 객체
run() 메서드에서는 다음과 같은 작업을 수행

frame_queue에서 프레임을 가져오고,
가져온 프레임이 None이면, detection_queue에도 None을 넣고, 큐를 닫음
가져온 프레임을 detect() 함수를 사용하여 추론
검출된 결과를 detection_queue에 넣음
results_queue에서 처리된 결과
None일 경우, 큐를 닫음
결과를 처리합니다.
frame_queue, detection_queue, results_queue에 task가 처리
"""