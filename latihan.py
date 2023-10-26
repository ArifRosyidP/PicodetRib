import paddle

# Muat model
model_state_dict = paddle.load("model/picodet_l_640_coco_lcnet/best_model.pdparams")

# Buat model Anda, misalnya sebuah model Linear
model = paddle.nn.Linear(10, 1)

# Terapkan bobot model
model.set_state_dict(model_state_dict)

# Sekarang Anda dapat menggunakan model untuk inferensi
# Misalnya, jika Anda memiliki beberapa data input:
input_data = paddle.randn([10])
output_data = model(input_data)