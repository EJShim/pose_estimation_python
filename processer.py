import torch
import torchvision

class Postprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.joint_num = 21

    def forward(self, output):
        output_depth = 672//self.joint_num

        heatmaps = output.reshape((1, 21, 32*32*32))
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=2)
        heatmaps = heatmaps.reshape([1, 21, 32, 32, 32])    

        accu_x = heatmaps.sum(axis=(2,3))
        accu_y = heatmaps.sum(axis=(2,4))
        accu_z = heatmaps.sum(axis=(3,4))

        accu_x = accu_x * torch.arange(32)
        accu_y = accu_y * torch.arange(32)
        accu_z = accu_z * torch.arange(output_depth)

        accu_x = accu_x.sum(axis=2, keepdims=True)
        accu_y = accu_y.sum(axis=2, keepdims=True)
        accu_z = accu_z.sum(axis=2, keepdims=True)

        accu_x = accu_x/32
        accu_y = accu_y/32
        accu_z = accu_z/output_depth*2 - 1 

        coord_out = torch.cat([accu_x, accu_y, accu_z], axis=2)

        return coord_out[0]



class Preprocessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        

    def forward(self, raw : torch.Tensor) -> torch.Tensor:

        image = raw[:,:,:3]
        image = torch.permute(image, (2, 0, 1))
        image = image.unsqueeze(0)
        image = image.to(torch.float32)

        #Not needed..
        # image = torch.nn.functional.interpolate(image, size=(256,256))
        
        return image


class Yolo_preprocessor(torch.nn.Module):
    def forward(self, x):
        x = x[:,:,:3]
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

    preprocessor = Preprocessor()
    dummy_input = torch.zeros([256, 256, 4])

    #Export onnx
    torch.onnx.export(
        preprocessor,
        dummy_input,
        'preprocessor.onnx',
        verbose = True,
        do_constant_folding=True,
        opset_version = 11,
        input_names=['input'],
        output_names=['output']
    )

    postprocessor = Postprocessor()
    dummy_input = torch.zeros([1, 672, 32, 32]).to(torch.float32)

    #Export onnx
    torch.onnx.export(
        postprocessor,
        dummy_input,
        'postprocessor.onnx',
        verbose = True,
        do_constant_folding=True,
        opset_version = 11,
        input_names=['input'],
        output_names=['output']
    )