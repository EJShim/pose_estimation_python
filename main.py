import onnxruntime
import cv2
import vtk
import numpy as np

import torch
from processer import Preprocessor

joints_name = ('Head_top', 
                'Thorax', 
                'R_Shoulder', 
                'R_Elbow', 
                'R_Wrist', 
                'L_Shoulder', 
                'L_Elbow', 
                'L_Wrist', 
                'R_Hip', 
                'R_Knee', 
                'R_Ankle', 
                'L_Hip', 
                'L_Knee', 
                'L_Ankle', 
                'Pelvis', 
                'Spine', 
                'Head', 
                'R_Hand', 
                'L_Hand', 
                'R_Toe', 
                'L_Toe')
joint_num = 21

def sphereActor(pos = [0, 0, 0]):

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(pos[0], pos[1], pos[2])
    sphere.SetRadius(0.05)
    sphere.Update()
    polydata = sphere.GetOutput()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


if __name__ == "__main__":

    estimator = onnxruntime.InferenceSession('models/estimator.onnx')

    preprocessor = Preprocessor()

    #Read input image
    image = cv2.imread('samples/sample.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
    image = cv2.resize(image, [256, 256])
    image = image.astype(np.float32)

    #Process input
    input_tensor = image

    #prediction
    output = estimator.run(["post_processor/output"], {'pre_processor/input' : input_tensor})[0]
    

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)



    target = vtk.vtkPolyData()
    target.SetPoints(vtk.vtkPoints())
    target.SetVerts(vtk.vtkCellArray())

    for idx, pos in enumerate(output):        
        
        target.GetPoints().InsertNextPoint(pos[0], pos[1], pos[2])
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, idx)
        target.GetVerts().InsertNextCell(vertex)

        actor = sphereActor(pos)

        joint_name = joints_name[idx].lower() 
        if  idx == 19:
            actor.GetProperty().SetColor(1, 0, 0)


        ren.AddActor(actor)

    # asdf = sphereActor()
    # asdf.GetProperty().SetColor(1, 0, 0)
    # ren.AddActor(asdf)

    # Render Knight
    reader = vtk.vtkOBJReader()
    reader.SetFileName("samples/decimated-knight.obj")
    reader.Update()
    polydata = reader.GetOutput()
    
    source = vtk.vtkPolyData()
    source.SetPoints(vtk.vtkPoints())
    source.SetVerts(vtk.vtkCellArray())

    for a, idx in enumerate( [262,165,242,452,58,67,65,318,100,395,8,232,420,174,288,415,12,294,324,61,26]):
        pos = polydata.GetPoints().GetPoint(idx)
        source.GetPoints().InsertNextPoint(pos[0], pos[1], pos[2])
        vertex = vtk.vtkVertex()
        vertex.GetPointIds().SetId(0, a)
        source.GetVerts().InsertNextCell(vertex)

    print(source.GetNumberOfPoints()) 
    print(target.GetNumberOfPoints())

    reg = vtk.vtkLandmarkTransform()
    reg.SetSourceLandmarks(source.GetPoints())
    reg.SetTargetLandmarks(target.GetPoints())
    reg.SetModeToRigidBody()    
    reg.Update()

    trans = vtk.vtkTransform()
    trans.SetMatrix(reg.GetMatrix())
    
    trans_poly = vtk.vtkTransformPolyDataFilter()
    trans_poly.SetInputData(polydata)
    trans_poly.SetTransform(trans)
    trans_poly.Update()
    polydata = trans_poly.GetOutput()

    #Write obj
    writer = vtk.vtkOBJWriter()
    writer.SetInputData(polydata)
    writer.SetFileName("knight_transformed.obj")
    writer.Update()
    


    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1,1,0)
    ren.AddActor(actor)

    ren.ResetCamera()
    renWin.Render()
    
    #Visuaslize 2D Image
    image = image.astype(np.uint8)
    cv2.imshow('asdf', image)

    iren.Start()
    cv2.destroyAllWindows()