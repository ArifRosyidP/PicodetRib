import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer

cfg = load_config('configs/picodet/picodet_l_640_coco_lcnet.yml')
trainer = Trainer(cfg, mode='test')
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_l_640_coco_lcnet/best_model.pdparams')
trainer.model.training = False

trainer.predict(
    images=['dataset/pothole_coco/testImage/img-36_jpg.rf.47f0fb502327ec69d5a041581727a149.jpg'],
    draw_threshold=0.5,
    output_dir='output/')