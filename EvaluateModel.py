import paddle
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.engine import Trainer
from ppdet.metrics import COCOMetric
import json

# Muat konfigurasi dan buat Trainer
cfg = load_config('configs/picodet/picodet_l_640_coco_lcnet.yml')
trainer = Trainer(cfg, mode='eval')

# Nonaktifkan normalisasi fusi dan mode penyebaran, dan muat bobot model
trainer.model.fuse_norm = False
trainer.model.deploy = False
trainer.load_weights('model/picodet_l_640_coco_lcnet/best_model.pdparams')

# Atur model ke mode evaluasi
trainer.evaluate()

