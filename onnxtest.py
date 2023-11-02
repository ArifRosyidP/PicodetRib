import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession('modelONNX/picodet_s_320_udtiri.onnx')

outputs = ort_session.run(
    None,
    {'input': np.random.randn(60, 2, 320, 320).astype(np.float32)}
)
print(outputs)