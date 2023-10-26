import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer

cfg = load_config('configs/picodet/picodet_l_640_coco_lcnet.yml')
trainer = Trainer(cfg, mode='test')
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_l_640_coco_lcnet/best_model.pdparams')
trainer.model.training = False

hasil = trainer.predict(
    images=['demo/car.jpg'],
    draw_threshold=0.5,
    output_dir='output/')

print(hasil)