import sys
import cv2
import vtk
import onnxruntime
from vtk.util import numpy_support
import torch
import numpy as np
import torchvision


class Yolo_preprocessor(torch.nn.Module):
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)        
        x = x.to(torch.float32)
        output = torch.nn.functional.interpolate(x, size=(480,640), mode='bilinear')
        output = output.permute(0, 2, 3, 1)
        output = output / 255.0

        return output

class Yolo_postprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conf_thres = 0.25
        self.iou_thres = 0.45        

    def xywh2xyxy(self, x, img_width, img_height):

        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        num_boxes= x.shape[0]
        zero_array = torch.zeros(num_boxes, dtype=int)
        width_array = torch.ones(num_boxes, dtype=int)*img_width
        height_array = torch.ones(num_boxes, dtype=int)*img_height
        
        y = x.clone()

        # y[:, 0] = x[:, 0] - x[:, 2] // 2 # top left x
        # y[:, 1] = x[:, 1] - x[:, 3] // 2 # top left y
        # y[:, 2] = x[:, 0] + x[:, 2] // 2 # bottom right x
        # y[:, 3] = x[:, 1] + x[:, 3] // 2  # bottom right y
        
        y[:, 0] = x[:, 0] - torch.div(x[:, 2] , 2, rounding_mode='trunc') # top left x
        y[:, 1] = x[:, 1] - torch.div(x[:, 3] , 2, rounding_mode='trunc') # top left y
        y[:, 2] = x[:, 0] + torch.div(x[:, 2] , 2, rounding_mode='trunc') # bottom right x
        y[:, 3] = x[:, 1] + torch.div(x[:, 3] , 2, rounding_mode='trunc')  # bottom right y


        y = y.to(int)

        y[:, 0] = torch.max(torch.stack((y[:, 0], zero_array)),axis=0).values  # top left x
        y[:, 1] = torch.max(torch.stack((y[:, 1], zero_array)),axis=0).values  # top left y
        y[:, 2] = torch.min(torch.stack((y[:, 2], width_array)),axis=0).values  # bottom right x
        y[:, 3] = torch.min(torch.stack((y[:, 3], height_array)),axis=0).values  # bottom right y

        return y

    
    def forward(self, x, original_image):            
        output = x[0]

        # Filter boxes with low confidence
        output = output[output[:,4] > self.conf_thres]

        
        # output = torch.tensor(output)
        classId = torch.argmax(output[:,5:], axis=1)        
        output = output[classId == 0]
        
        boxes = output[:,:4]
        image_width = original_image.size(0)
        image_height = original_image.size(1)
        boxes[:, 0] *= image_height
        boxes[:, 1] *= image_width 
        boxes[:, 2] *= image_height  
        boxes[:, 3] *= image_width 
        scores = output[:,4]

        boxes = boxes[torch.logical_or(boxes[:,2] > 0, boxes[:,3] > 0)]

        # Keep boxes only with positive width and height
        boxes = self.xywh2xyxy(boxes, image_width, image_height).to(torch.float32)

        # NMS
        box_ids = torchvision.ops.nms(boxes, scores, self.iou_thres)
        scores = scores[box_ids]
        boxes = boxes[box_ids,:]

        return boxes, scores


if __name__ == "__main__":
    #Set Default input paht
    input_path = "samples/DF0N5391.jpg"
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

    # Get Detection model
    detection_preprocessing__model = onnxruntime.InferenceSession('./models/yolo_480_640_float32_pre.onnx')
    detection_model = onnxruntime.InferenceSession('./models/yolo_480_640_float32.onnx')
    detection_postprocessing_model = onnxruntime.InferenceSession('./models/yolo_480_640_float32_post.onnx')

    # Get Pose model    
    pose_model = onnxruntime.InferenceSession('./models/mobile_human_pose_working_well_256x256.onnx')

    #Postprocessors
    # detector_preprocessor = Yolo_preprocessor()
    # detection_postprocessor = Yolo_postprocessor()

    #Read input image
    image = cv2.imread(input_path)
    cv2.imshow("original image", image)
    
    
    # Run Detection
    # image_input = detector_preprocessor(torch.tensor(image))
    image_input = detection_preprocessing__model.run(["output"], {"input" : image})[0]
    output = detection_model.run(["Identity"], {"input_1":image_input})[0]

    #boxes, scores = detection_postprocessor(torch.tensor(output), torch.tensor(image))
    boxes, scores = detection_postprocessing_model.run(["boxes", "scores"], {"input":output,"original_image":image})
    

    #Draw xboxes
    for box, score in zip(boxes, scores):
        box_img = cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0), 3)
        cv2.putText(box_img, str(int(100*score)) + '%', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,191,0), 3, cv2.LINE_AA)

    cv2.imshow("box", box_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

