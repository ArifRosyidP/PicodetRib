import cv2
import numpy as np
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
import paddle.vision.models
# Load the Paddle Detection configuration and model
cfg = load_config('configs/picodet/picodet_l_640_coco_lcnet.yml')
trainer = Trainer(cfg, mode='test')
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_l_640_coco_lcnet/best_model.pdparams')
trainer.model.training = False

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = trainer.predict(images=[frame], draw_threshold=0.5)

    for result in results:
        for box in result['bbox']:
            x, y, w, h = box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # Display the frame with object detection
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
