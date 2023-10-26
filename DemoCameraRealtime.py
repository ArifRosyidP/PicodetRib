import os
import time
import cv2
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
import datetime
import collections

# Load the model
cfg = load_config('configs/picodet/picodet_l_640_coco_lcnet.yml')
trainer = Trainer(cfg, mode='test')
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_l_640_coco_lcnet/best_model.pdparams')
trainer.model.training = False

# Start video capture
cap = cv2.VideoCapture(0)

# Create a separate directory for images with potholes
pothole_dir = 'output/potholes/'
if not os.path.exists(pothole_dir):
    os.makedirs(pothole_dir)

# Initialize deque for calculating moving average of FPS
fps_deque = collections.deque(maxlen=20)
start_time = time.time()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Save the captured frame to a file
    cv2.imwrite('output/temp.jpg', frame)

    # Perform prediction on the captured frame
    results = trainer.predict(
        images=['output/temp.jpg'],
        draw_threshold=0.7,
        output_dir='output/')

    # Check if 'pothole' is in the detected classes and save the image if it is
    for result in results:
        if 'pothole' in result['pred_class_names']:
            # Generate a unique filename based on the current time
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.jpg'
            # Read the processed image with bounding boxes
            processed_img = cv2.imread('output/temp.jpg')
            # Save the captured frame to a file
            cv2.imwrite(pothole_dir + filename, processed_img)

    # Calculate FPS using moving average and display it on the frame
    fps_deque.append(time.time() - start_time)
    start_time = time.time()
    fps = len(fps_deque)/sum(fps_deque)
    
    # Convert the fps to string so we can display it on frame
    fps_str = "FPS:"+str(int(fps))

    # Display the resulting frame with bounding boxes and FPS
    img = cv2.imread('output/temp.jpg')

    # Put fps on the processed image
    cv2.putText(img, fps_str, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()