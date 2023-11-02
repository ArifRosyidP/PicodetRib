from onnx2keras import onnx_to_keras
import keras
import onnx

onnx_model = onnx.load('modelONNX/picodet_s_320_udtiri.onnx')
input_all = [node.name for node in onnx_model.graph.input]
k_model = onnx_to_keras(onnx_model, input_all)

keras.models.save_model(k_model,'modelKeras/picodet_s_320_udtiri.h5',overwrite=True,include_optimizer=True)