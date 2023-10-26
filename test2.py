import cv2
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer

cfg = load_config('configs/picodet/picodet_l_640_coco_lcnet.yml')
trainer = Trainer(cfg, mode='test')
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_l_640_coco_lcnet/best_model.pdparams')
trainer.model.training = False

# Start video capture
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Save the captured frame to a file
    cv2.imwrite('temp.jpg', frame)

    # Perform prediction on the captured frame
    trainer.predict(
        images=['temp.jpg'],
        draw_threshold=0.5,
        output_dir='output/')

    # Display the resulting frame with bounding boxes
    img = cv2.imread('output/temp.jpg')
    cv2.imshow('frame', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()