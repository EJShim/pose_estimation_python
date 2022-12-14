import os, sys
root_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_path)
import onnxruntime
import numpy as np
import cv2
import vtk
from vtk.util import numpy_support
from scipy.special import softmax
from reference.utils import pixel2cam, crop_image, joint_num, draw_skeleton, draw_heatmap, colors, skeleton
from reference. yolo import YoloV5s

class MobileHumanPose():

    def __init__(self, model_path, focal_length = [1500, 1500], principal_points = [1280/2, 720/2]):

        self.focal_length = focal_length
        self.principal_points = principal_points

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image, bbox, abs_depth = 1.0):

        return self.estimate_pose(image, bbox, abs_depth)

    def initialize_model(self, model_path):

        self.session = onnxruntime.InferenceSession(model_path)

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def estimate_pose(self, image, bbox, abs_depth = 1000):

        input_tensor = self.prepare_input(image, bbox)

        output = self.inference(input_tensor)

        keypoints = self.process_output(output, abs_depth, bbox)

        return keypoints

    def prepare_input(self, image, bbox):

        img = crop_image(image, bbox)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.img_height, self.img_width, self.img_channels = img.shape
        principal_points = [self.img_width/2,  self.img_height/2]

        img_input = cv2.resize(img, (self.input_width,self.input_height))
        
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis,:,:,:]        

        return img_input.astype(np.float32)


    def inference(self, input_tensor):

        output = self.session.run(self.output_names, {self.input_name: input_tensor})[0]

        return np.squeeze(output)

    def process_output(self, output, abs_depth, bbox):  

        heatmaps = output.reshape((-1,joint_num, self.output_depth*self.output_height*self.output_width))
        heatmaps = softmax(heatmaps, 2)

        scores = np.squeeze(np.max(heatmaps, 2)) # Ref: https://github.com/mks0601/3DMPPE_POSENET_RELEASE/issues/47

        heatmaps = heatmaps.reshape((-1, joint_num, self.output_depth, self.output_height, self.output_width))
        
        accu_x = heatmaps.sum(axis=(2,3))
        accu_y = heatmaps.sum(axis=(2,4))
        accu_z = heatmaps.sum(axis=(3,4))

        accu_x = accu_x * np.arange(self.output_width, dtype=np.float32)
        accu_y = accu_y * np.arange(self.output_height, dtype=np.float32)
        accu_z = accu_z * np.arange(self.output_depth, dtype=np.float32)

        accu_x = accu_x.sum(axis=2, keepdims=True)
        accu_y = accu_y.sum(axis=2, keepdims=True)
        accu_z = accu_z.sum(axis=2, keepdims=True)

        scores2 = []
        for i in range(joint_num):
            scores2.append(heatmaps.sum(axis=2)[0, i, int(accu_y[0,i,0]), int(accu_x[0,i,0])])

        accu_x = accu_x/self.output_width
        accu_y = accu_y/self.output_height
        accu_z = accu_z/self.output_depth*2 - 1 

        coord_out = np.squeeze(np.concatenate((accu_x, accu_y, accu_z), axis=2))

        pose_2d = coord_out[:,:2]
        pose_2d[:,0] = pose_2d[:,0] * self.img_width + bbox[0]
        pose_2d[:,1] = pose_2d[:,1] * self.img_height + bbox[1]

        joint_depth = coord_out[:,2]*1000 + abs_depth

        pose_3d = pixel2cam(pose_2d, joint_depth, self.focal_length, self.principal_points)

        # Calculate the joint heatmap
        person_heatmap = cv2.resize(np.sqrt(heatmaps.sum(axis=(1,2))[0,:,:]), (self.img_width,self.img_height))

        return pose_2d, pose_3d, person_heatmap, scores

    def getModel_input_details(self):

        self.input_name = self.session.get_inputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.channels = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]


    def getModel_output_details(self):

        model_outputs = self.session.get_outputs()

        self.output_names = []
        self.output_names.append(model_outputs[0].name)

        self.output_shape = model_outputs[0].shape
        self.output_depth = self.output_shape[1]//joint_num
        self.output_height = self.output_shape[2]
        self.output_width = self.output_shape[3]


if __name__ == "__main__":

    input_path = "samples/DF0N5391.jpg"
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]

    draw_detections = True
    focal_length = [None, None]
    principal_points = [None, None]

    pose_model_path = './models/mobile_human_pose_working_well_256x256.onnx'
    pose_estimator = MobileHumanPose(pose_model_path, focal_length, principal_points)

    detector_model_path = './models/yolo_480_640_float32.onnx'
    person_detector = YoloV5s(detector_model_path, conf_thres=0.5, iou_thres=0.4)

    image = cv2.imread(input_path)
    

    # Detect people in the image
    boxes, scores = person_detector(image)

    det_img = image.copy()
    if draw_detections:
        det_img = person_detector.draw_detections(det_img, [boxes[0]], scores)
    

    # Simulate depth based on the bouding box area
    box = boxes[0]
    area = (box[2] - box[0]) * (box[3] - box[1])
    depth = 500 / (area / (image.shape[0] * image.shape[1])) + 500    

    # Run Pose Estimation
    keypoints, pose_3d, person_heatmap, scores = pose_estimator(image, box, depth)
    pose_img = image.copy()
    pose_img = draw_skeleton(pose_img, keypoints, box[:2], scores)
    
    # Add the person heatmap to the image heatmap
    heatmap_viz_img = image.copy()
    img_heatmap = np.empty(image.shape[:2])
    img_heatmap[box[1]:box[3],box[0]:box[2]] += person_heatmap 
    heatmap_viz_img = draw_heatmap(heatmap_viz_img, img_heatmap)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)



    # Add Joint
    points = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    point_color = vtk.vtkUnsignedCharArray()
    point_color.SetNumberOfComponents(3)
    for idx, pose in enumerate(pose_3d):        

        points.InsertNextPoint(pose)        
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, idx)
        verts.InsertNextCell(vertex)

        color = colors[idx]
        point_color.InsertNextTuple([color[0] * 255, color[1] * 255, color[2] * 255])
        
    # Add Skeelton
    lines = vtk.vtkCellArray()
    for sk in skeleton:        
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, sk[0])
        line.GetPointIds().SetId(1, sk[1])
        lines.InsertNextCell(line)    

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)    
    polydata.SetVerts(verts)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(point_color)

    mapper = vtk.vtkPolyDataMapper()
    # mapper.SetRadius(10)
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)    
    actor.GetProperty().SetLineWidth(5);

    
    ren.AddActor(actor)    

    ren.ResetCamera()
    renWin.Render()    

    a = cv2.hconcat([image, det_img])
    b = cv2.hconcat([heatmap_viz_img, pose_img ])
    final = cv2.vconcat([a,b])
    cv2.imshow("image", final)
    
    iren.Start()
    cv2.destroyAllWindows()
    