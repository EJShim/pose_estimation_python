import sys
import cv2
import vtk
import onnxruntime
from vtk.util import numpy_support
import numpy as np


if __name__ == "__main__":
    #Set Default input paht
    input_path = "samples/DF0N5391.jpg"
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

    detector_model_path = './models/yolo_480_640_float32.onnx'
    pose_model_path = './models/mobile_human_pose_working_well_256x256.onnx'

    detection_model = onnxruntime.InferenceSession(detector_model_path)
    pose_model = onnxruntime.InferenceSession(pose_model_path)


    #Detection shape
    input_shape = detection_model.get_inputs()[0].shape
    # 480, 640, 3


    #Read input image
    image = cv2.imread(input_path)

    image_width = image.shape[0]
    image_height = image.shape[1]
    
    #Prepare for detection
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_input = cv2.resize(image, (640, 480))
    image_input = image_input / 255.0
    image_input = np.expand_dims(image_input, 0)
    image_input = image_input.astype(np.float32)

    print(detection_model.get_inputs()[0].name)
    print(detection_model.get_outputs()[0].name)
    
    
    output = detection_model.run(["Identity"], {"input_1":image_input})[0]

    #postprocess output
    conf_thres = 0.25

    # Filter boxes with low confidence
    output = output[output[:,4] > conf_thres]
    
    # Filter person class only
    classId = np.argmax(output[:,5:], axis=1)
    output = output[classId == 0]

    # FIXME : need to verify if width and height are
    boxes = output[:,:4]
    boxes[:, 0] *= image_width
    boxes[:, 1] *= image_height 
    boxes[:, 2] *= image_width  
    boxes[:, 3] *= image_height 



