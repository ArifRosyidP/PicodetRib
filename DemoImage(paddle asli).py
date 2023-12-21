import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer

cfg = load_config('configs/picodet/picodet_s_320_coco_lcnet_udtiri.yml')
trainer = Trainer(cfg, mode='test')
trainer.load_weights('model/picodet_s_320_coco_lcnet_udtiri/best_model_epoch_251.pdparams')

trainer.predict(
    images= "dataset/udtiri/validImage/",
    draw_threshold=0.5,
    save_results=True,
    output_dir='output/testingGaes')

#print(hasil)