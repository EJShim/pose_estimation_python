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