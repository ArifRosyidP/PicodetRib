import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer

cfg = load_config('configs/picodet/picodet_s_320_coco_lcnet_udtiri.yml')
trainer = Trainer(cfg, mode='test')
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_s_320_coco_lcnet_udtiri/best_model_epoch_209.pdparams')
trainer.model.training = False

hasil = trainer.predict(
    images=['demo/lubang2.jpg'],
    draw_threshold=0.5,
    output_dir='output/')

print(hasil)