import time
import cv2
import numpy as np
import onnx
import onnxruntime
# from imread_from_url import imread_from_url


def non_max_suppression(boxes, scores, threshold):	

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
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                        areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
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

def xywh2xyxy(x, img_width, img_height):
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

    y = y.astype(np.int)

    y[:, 0] = np.max(np.vstack((y[:, 0], zero_array)),axis=0)  # top left x
    y[:, 1] = np.max(np.vstack((y[:, 1], zero_array)),axis=0)  # top left y
    y[:, 2] = np.min(np.vstack((y[:, 2], width_array)),axis=0)  # bottom right x
    y[:, 3] = np.min(np.vstack((y[:, 3], height_array)),axis=0)  # bottom right y

    return y


anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
class YoloV5s():

    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image):

        return self.detect_objects(image)

    def initialize_model(self, model_path):

        self.session = onnxruntime.InferenceSession(model_path)

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        output = self.inference(input_tensor)

        boxes, scores = self.process_output(output)

        return boxes, scores

    def prepare_input(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width, self.img_channels = img.shape

        img_input = cv2.resize(img, (self.input_width, self.input_height))
                
        img_input = img_input/ 255.0
        # img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, 0)     

        return img_input.astype(np.float32)

    def inference(self, input_tensor):

        return self.session.run(self.output_names, {self.input_name: input_tensor})[0]

    def process_output(self, output):


        output = np.squeeze(output)

        # Filter boxes with low confidence
        output = output[output[:,4] > self.conf_thres]

        # Filter person class only
        classId = np.argmax(output[:,5:], axis=1)
        output = output[classId == 0]

        boxes = output[:,:4]
        boxes[:, 0] *= self.img_width
        boxes[:, 1] *= self.img_height 
        boxes[:, 2] *= self.img_width  
        boxes[:, 3] *= self.img_height 

        # Keep boxes only with positive width and height
        boxes = boxes[np.logical_or(boxes[:,2] > 0, boxes[:,3] > 0)]

        scores = output[:,4]
        boxes = xywh2xyxy(boxes, self.img_width, self.img_height).astype(int)

        box_ids = non_max_suppression(boxes, scores, self.iou_thres)

        if box_ids.shape[0] == 0:
            return None, None

        scores = scores[box_ids]
        boxes = boxes[box_ids,:]

        return boxes, scores

    def getModel_input_details(self):

        self.input_name = self.session.get_inputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.channels = self.input_shape[3]

    def getModel_output_details(self):

        model_outputs = self.session.get_outputs()
        self.output_names = []
        self.output_names.append(model_outputs[0].name)

    @staticmethod
    def draw_detections(img, boxes, scores):

        if boxes is None:
            return img

        for box, score in zip(boxes, scores):
            img = cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0), 3)

            cv2.putText(img, str(int(100*score)) + '%', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,191,0), 3, cv2.LINE_AA)

        return img

# if __name__ == '__main__':

#     model_path='../models/model_float32.onnx'  
#     image = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Bruce_McCandless_II_during_EVA_in_1984.jpg/768px-Bruce_McCandless_II_during_EVA_in_1984.jpg")

#     object_detector = YoloV5s(model_path)

#     boxes, scores = object_detector(image)  

#     image = YoloV5s.draw_detections(image, boxes, scores)
   
#     cv2.namedWindow("Detected people", cv2.WINDOW_NORMAL)
#     cv2.imshow("Detected people", image)
#     cv2.waitKey(0)