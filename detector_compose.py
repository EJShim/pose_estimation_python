from full import Yolo_preprocessor, Yolo_postprocessor
import torch
import numpy as np
import onnxruntime
import onnx
import cv2

if __name__ == "__main__":
    preprocessor = Yolo_preprocessor()
    postprocessor = Yolo_postprocessor()


    dummy_image = torch.tensor( cv2.imread('samples/DF0N5391.jpg'))

    # Exprot preprocessor onnx
    torch.onnx.export(
        preprocessor,
        dummy_image,
        './models/yolo_480_640_float32_pre.onnx',
        verbose = True,
        do_constant_folding=True,
        opset_version = 11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            "input" : {0 : "width", 1:"height", 2:"channels"}
        }
    )


    dummy_pre = preprocessor(dummy_image)
    detection_model = onnxruntime.InferenceSession('./models/yolo_480_640_float32.onnx')
    dummy_output = detection_model.run(["Identity"], {"input_1":dummy_pre.numpy()})[0]
    dummy_output = torch.tensor(dummy_output)

    dummy_output = torch.zeros([1, 18900, 85])
    dummy_image = torch.zeros([1000, 1000, 3]).to(torch.uint8)
    print("====================================================")
    torch.onnx.export(
        postprocessor,
        (dummy_output, dummy_image),
        './models/yolo_480_640_float32_post.onnx',
        verbose= True,
        do_constant_folding=True,
        opset_version=11,
        input_names=['input', 'original_image'],
        output_names=['boxes','scores'],
        dynamic_axes={
            "original_image" : {0 : "width", 1:"height", 2:"channels"}
        }
    )