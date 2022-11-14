import sys
import cv2
import vtk
import onnxruntime
from vtk.util import numpy_support
import torch
import numpy as np


class Yolo_preprocessor(torch.nn.Module):
    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        output = torch.nn.functional.interpolate(x, size=(480,640))
        output = output.permute(0, 2, 3, 1)
        output = output / 255.0
        output = output.to(torch.float32)

        return output

class Yolo_postprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conf_thres = 0.25
        self.iou_thres = 0.45


    def compute_iou(self, box, boxes, box_area, boxes_area):
        # this is the iou of the box against all other boxes
        assert boxes.shape[0] == boxes_area.shape[0]

        # get all the origin-ys
        # push up all the lower origin-xs, while keeping the higher origin-xs
        ys1 = np.maximum(box[0], boxes[:, 0])

        # get all the origin-xs
        # push right all the lower origin-xs, while keeping higher origin-xs
        xs1 = np.maximum(box[1], boxes[:, 1])

        # get all the target-ys
        # pull down all the higher target-ys, while keeping lower origin-ys
        ys2 = np.minimum(box[2], boxes[:, 2])

        # get all the target-xs
        # pull left all the higher target-xs, while keeping lower target-xs
        xs2 = np.minimum(box[3], boxes[:, 3])

        # each intersection area is calculated by the
        # pulled target-x minus the pushed origin-x
        # multiplying
        # pulled target-y minus the pushed origin-y
        # we ignore areas where the intersection side would be negative
        # this is done by using maxing the side length by 0
        intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)

        # each union is then the box area
        # added to each other box area minusing their intersection calculated above
        unions = box_area + boxes_area - intersections

        # element wise division
        # if the intersection is 0, then their ratio is 0
        ious = intersections / unions
        return ious

    def non_max_suppression(self, boxes, scores, threshold):	

        assert boxes.shape[0] == scores.shape[0]

        # bottom-left origin
        ys1 = boxes[:, 0]
        xs1 = boxes[:, 1]

        # top-right target
        ys2 = boxes[:, 2]
        xs2 = boxes[:, 3]

        # box coordinate ranges are inclusive-inclusive
        areas = (ys2 - ys1) * (xs2 - xs1)
        scores_indexes = scores.argsort().tolist()
        boxes_keep_index = []

        while len(scores_indexes):
            index = scores_indexes.pop()
            boxes_keep_index.append(index)
            if not len(scores_indexes):
                break
            ious = self.compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                            areas[scores_indexes])
            filtered_indexes = set((ious > threshold).nonzero()[0])
            # if there are no more scores_index
            # then we should pop it
            scores_indexes = [
                v for (i, v) in enumerate(scores_indexes)
                if i not in filtered_indexes
            ]
        return np.array(boxes_keep_index)

    def xywh2xyxy(self, x, img_width, img_height):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        num_boxes= x.shape[0]
        zero_array = np.zeros(num_boxes, dtype=int)
        width_array = np.ones(num_boxes, dtype=int)*img_width
        height_array = np.ones(num_boxes, dtype=int)*img_height

        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] // 2 # top left x
        y[:, 1] = x[:, 1] - x[:, 3] // 2 # top left y
        y[:, 2] = x[:, 0] + x[:, 2] // 2 # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] // 2  # bottom right y

        y = y.astype(int)

        y[:, 0] = np.max(np.vstack((y[:, 0], zero_array)),axis=0)  # top left x
        y[:, 1] = np.max(np.vstack((y[:, 1], zero_array)),axis=0)  # top left y
        y[:, 2] = np.min(np.vstack((y[:, 2], width_array)),axis=0)  # bottom right x
        y[:, 3] = np.min(np.vstack((y[:, 3], height_array)),axis=0)  # bottom right y

        return y

    
    def forward(self, x, image_width, image_height):
        output = x[0]

        # Filter boxes with low confidence
        output = output[output[:,4] > self.conf_thres]
        
        classId = np.argmax(output[:,5:], axis=1)
        output = output[classId == 0]
        
        boxes = output[:,:4]
        boxes[:, 0] *= image_height
        boxes[:, 1] *= image_width 
        boxes[:, 2] *= image_height  
        boxes[:, 3] *= image_width 
        scores = output[:,4]
        
        boxes = boxes[np.logical_or(boxes[:,2] > 0, boxes[:,3] > 0)]

        # # Keep boxes only with positive width and height
        boxes = self.xywh2xyxy(boxes, image_width, image_height).astype(int)


        # Filter ids?
        box_ids = self.non_max_suppression(boxes, scores, self.iou_thres)
        scores = scores[box_ids]
        boxes = boxes[box_ids,:]

        return boxes, scores



if __name__ == "__main__":
    #Set Default input paht
    input_path = "samples/DF0N5391.jpg"
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

    detector_model_path = './models/yolo_480_640_float32.onnx'
    pose_model_path = './models/mobile_human_pose_working_well_256x256.onnx'

    detection_model = onnxruntime.InferenceSession(detector_model_path)
    pose_model = onnxruntime.InferenceSession(pose_model_path)

    #Postprocessors
    detector_preprocessor = Yolo_preprocessor()
    detection_postprocessor = Yolo_postprocessor()


    #Detection shape
    input_shape = detection_model.get_inputs()[0].shape
    # 480, 640, 3


    #Read input image
    image = cv2.imread(input_path)
    cv2.imshow("original image", image)
    
    # Get the origianl image size?
    image_width = image.shape[0]
    image_height = image.shape[1]
    
    # Run Detection
    image_input = detector_preprocessor(torch.tensor(image))
    output = detection_model.run(["Identity"], {"input_1":image_input.numpy()})[0]
    boxes, scores = detection_postprocessor(output, image_width, image_height)
    

    #Draw xboxes
    for box, score in zip(boxes, scores):
        box_img = cv2.rectangle(image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0), 3)
        cv2.putText(box_img, str(int(100*score)) + '%', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,191,0), 3, cv2.LINE_AA)

    cv2.imshow("box", box_img)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

