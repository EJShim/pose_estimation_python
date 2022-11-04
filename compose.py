# ONNX Compoase : https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md
import onnx

if __name__ == "__main__":
    estimator_path = 'saved_model_mobile_human_pose_working_well\mobile_human_pose_working_well_256x256.onnx'
    postprocessor_path = 'postprocessor.onnx'
    preprocessor_path = 'preprocessor.onnx'

    estimator = onnx.load(estimator_path)
    estimator = onnx.compose.add_prefix(estimator, prefix="estimator/")

    preprocessor = onnx.load(preprocessor_path)
    preprocessor = onnx.compose.add_prefix(preprocessor, prefix="pre_processor/")

    post_processor = onnx.load(postprocessor_path)
    post_processor = onnx.compose.add_prefix(post_processor, prefix="post_processor/")


    estimator = onnx.compose.merge_models(
        preprocessor, estimator, io_map=[("pre_processor/output", "estimator/input")]
    )

    estimator = onnx.compose.merge_models(
        estimator, post_processor, io_map=[("estimator/output", "post_processor/input")]
    )

    # Chagne input output name again
    # combined_model.graph.node[0].input[0] = 'input'
    # combined_model.graph.node[-1].output[-1] = 'output'

    # print(combined_model.graph.node[-2])
    # print(combined_model.graph.node[-1])

    onnx.save(estimator, 'estimator.onnx')