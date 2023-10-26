from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Muat dataset COCO dan prediksi model dalam format COCO
coco_gt = COCO("dataset/pothole_coco/annotations/instance_validPothole.json")
coco_dt = coco_gt.loadRes("evaluation/bbox.json")

# Buat objek COCOeval
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# Evaluasi model pada semua IoU dari 0.5 sampai 0.95
coco_eval.params.iouThrs = [0.5, 0.95]

# Lakukan evaluasi
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Dapatkan mAP
mAP = coco_eval.stats[0]  # mAP @[ IoU=0.50:0.95 | area= all | maxDets=100 ]
print("Mean Average Precision (mAP):", mAP)