from full import Yolo_preprocessor, Yolo_postprocessor
import torch
import numpy as np
import onnxruntime
import onnx

if __name__ == "__main__":
    preprocessor = Yolo_preprocessor()
    postprocesso = Yolo_postprocessor()

    dummy_input = np.zeros([640, 480, 4])


    dummy_output = preprocessor(dummy_input)
    

    # Exprot preprocessor onnx
    torch.onnx.export(
        preprocessor,
        dummy_input,
        './models/yolo_480_640_float32_pre.onnx',
        verbose = True,
        do_constant_folding=True,
        opset_version = 15,
        input_names=['input'],
        output_names=['output']
    )