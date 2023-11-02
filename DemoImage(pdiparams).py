import paddle

# Muat model
model = paddle.jit.load("inference_model/picodet_s_320_coco_lcnet_udtiri")

# Pastikan model dalam mode evaluasi
model.eval()

# Sekarang Anda dapat menggunakan 'model' untuk inferensi pada data Anda