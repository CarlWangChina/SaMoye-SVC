import onnx
from onnx import helper, checker
from onnx import TensorProto,GraphProto
import onnx.utils
from onnx.tools import update_model_dims
import re
import argparse
import numpy as np
import os
from onnx import shape_inference
from onnx.tools import update_model_dims
import onnxruntime

def print_node_shapes(model):

    # Infer shapes
    inferred_model = shape_inference.infer_shapes(model)

    # Iterate over all nodes in the model
    for node in inferred_model.graph.node:
        # Iterate over all inputs of the node
        for input in node.input:
            # Find the corresponding value_info in the graph's input
            for value_info in inferred_model.graph.value_info:
                if value_info.name == input:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    if len(shape) > 1 and shape[0] == 1:
                        print(f"Node name: {node.name}, Input: {input}, shape: {shape}")
        
        # Iterate over all outputs of the node
        for output in node.output:
            # Find the corresponding value_info in the graph's output
            for value_info in inferred_model.graph.value_info:
                if value_info.name == output:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    if len(shape) > 1 and shape[0] == 1:
                        print(f"Node name: {node.name}, Output: {output}, shape: {shape}")                     

def modify_node_shapes(model):
    # Infer shapes
    inferred_model = shape_inference.infer_shapes(model)

    # Iterate over all nodes in the model
    for node in inferred_model.graph.node:
        # Iterate over all inputs of the node
        for input in node.input:
            # Find the corresponding value_info in the graph's input
            for value_info in inferred_model.graph.value_info:
                if value_info.name == input:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    if len(shape) > 1 and shape[0] == 1:
                        # print(f"Node name: {node.name}, Input: {input}, shape: {shape}")
                        # Set the first dimension to None
                        value_info.type.tensor_type.shape.dim[0].dim_value = -1
                        # value_info.type.tensor_type.shape.dim[0].dim_param = 'None'
        
        # Iterate over all outputs of the node
        for output in node.output:
            # Find the corresponding value_info in the graph's output
            for value_info in inferred_model.graph.value_info:
                if value_info.name == output:
                    shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                    if len(shape) > 1 and shape[0] == 1:
                        # print(f"Node name: {node.name}, Output: {output}, shape: {shape}")
                        # Set the first dimension to None
                        value_info.type.tensor_type.shape.dim[0].dim_value = 0
                        value_info.type.tensor_type.shape.dim[0].dim_param = 'None'
    return inferred_model

def modify_node_input(model, target_node_name, target_input, new_value):
    # Iterate over all nodes in the model
    for node in model.graph.node:
        # Check if the current node is the one we're looking for
        if node.name == target_node_name:
            # Check if the input we're looking for is in the node's inputs
            if target_input in node.input:
                # Modify the value
                for i,input in enumerate(node.input):
                    if input == target_input:
                        print(f"Original value: {input}")
    return model

def delete_ini_insert_constant(model,node_name, target_input_name, new_value):
    for node in model.graph.node:
        if node.name == node_name:
            # Check if 'target_input_name' is in the inputs of the node
            if target_input_name in node.input:
                print(f"The {target_input_name} parameter is used in the node '{node.name}' of the module '{node.op_type}'")
                print(f"The inputs of the node are: {node.input}")
                # Replace 'speedup' inputs with the new constant node
                # 获取speedup在node.input中的索引
                for i,input in enumerate(node.input):
                    if input == target_input_name:
                        node.input.pop(i)
                        for initializer in model.graph.initializer:
                            if initializer.name == target_input_name:
                                model.graph.initializer.remove(initializer)
                                print(f"Remove {target_input_name} from initializer")
                                break
                        new_tensor = helper.make_tensor(target_input_name, 7, [], [new_value])
                        new_node = helper.make_node('Constant', [], [target_input_name], value=new_tensor)
                        model.graph.node.insert(0, new_node)
                        node.input.append(target_input_name)
                print(f"The inputs of the node are: {node.input}")
            break
    return model

def change_init(model, node_name, new_name, target_input_name, new_value):
    for node in model.graph.node:
        if node.name == node_name:
            # Check if 'target_input_name' is in the inputs of the node
            if target_input_name in node.input:
                print(f"The {target_input_name} parameter is used in the node '{node.name}' of the module '{node.op_type}'")
                print(f"The inputs of the node are: {node.input}")
                # Replace 'speedup' inputs with the new constant node
                # 获取speedup在node.input中的索引
                for i,input in enumerate(node.input):
                    if input == target_input_name:
                        print(f'input is {input}')
                        node.input.pop(i)
                        new_initializer_name = target_input_name + new_name
                        new_initializer = helper.make_tensor(new_initializer_name, data_type=TensorProto.INT64, dims=[len(new_value)], vals=new_value)
                        if new_initializer_name not in [initializer.name for initializer in model.graph.initializer]:
                            model.graph.initializer.append(new_initializer)
                            print(f"Add {new_initializer_name} to initializer")
                        node.input.append(new_initializer_name)
                print(f"The inputs of the node are: {node.input}")
            break
    return model

def old_main():
    onnx_model_file = '/home/john/Muer_DS/DiffSinger/onnxrun/onnx_model/zhibin_pitch'

    onnx_model_name = '0913_zhibin_melodyencoder128x4_nobasepitch.linguistic.onnx'
    onnx_model_new_name = 'batch_size_none.linguistic.onnx'
    
    onnx_model_path = os.path.join(onnx_model_file, onnx_model_name)
    onnx_model_new_path = os.path.join(onnx_model_file, onnx_model_new_name)
    model = onnx.load(onnx_model_path)
    model = modify_node_shapes(model)
    for i in range(len(model.graph.input)):
        dim_proto = model.graph.input[i].type.tensor_type.shape.dim[0]
        dim_proto.dim_param = 'None'
    for i in range(len(model.graph.output)):
        dim_proto = model.graph.output[i].type.tensor_type.shape.dim[0]
        dim_proto.dim_param = 'None'
    onnx.checker.check_model(model)
    onnx.save(model, onnx_model_new_path)

    onnx_model_name = '0913_zhibin_melodyencoder128x4_nobasepitch.pitch.onnx'
    onnx_model_new_name = 'batch_size_none.pitch.onnx'
    
    onnx_model_path = os.path.join(onnx_model_file, onnx_model_name)
    onnx_model_new_path = os.path.join(onnx_model_file, onnx_model_new_name)
    
    model = onnx.load(onnx_model_path)
    model = modify_node_shapes(model)
    # model = modify_node_input(model, '/pre/smooth/Unsqueeze', '/lr/Constant_4_output_0', 1)
    model = change_init(model, '/pre/smooth/Unsqueeze', '_1', '/lr/Constant_4_output_0', [1])
    model = change_init(model, '/pre/smooth/Squeeze', '_2', '/lr/Constant_4_output_0', [1])
    # Create a new constant node for the 'speedup' value
    speedup_value = 100  # Replace this with your desired speedup value
    speedup_tensor = helper.make_tensor('speedup', 7, [], [speedup_value])
    speedup_node = helper.make_node('Constant', [], ['speedup'], value=speedup_tensor)

    # Add the new constant node to the model
    # model.graph.node.extend([speedup_node])
    # model.graph.node.insert(0, speedup_node)
    # Iterate over all nodes in the model
    # for node in model.graph.node:
    # # Check if 'speedup' is in the inputs of the node
    #     if 'speedup' in node.input:
    #         print(f"The 'speedup' parameter is used in the node '{node.name}' of the module '{node.op_type}'")
    #         print(f"The other inputs of the node are: {node.input}")
    #         # Replace 'speedup' inputs with the new constant node
    #         # 获取speedup在node.input中的索引
    #         for i,input in enumerate(node.input):
    #             if input == 'speedup':
    #                 node.input.pop(i)
    #                 node.input.append('speedup')
    #         print(f"The other inputs of the node are: {node.input}")
    
    
    # print_node_shapes(model)

    for i in range(len(model.graph.input)):
        # print(f"{i}  = {model.graph.input[i].name}")
        # print(f"{i}  = {model.graph.input[i].type.tensor_type.elem_type}")
        input_shape = model.graph.input[i].type.tensor_type.shape.dim
        # print("Original input shape:", input_shape)
        try:
            input_shape[0].dim_param = 'None'
            # print("New input shape:", input_shape)
        except:
            # 保存要删除的输入
            # removed_input = model.graph.input.pop(7)  # 删除第8个输入并保存其信息

            # 创建一个新的输入节点
            # new_input = onnx.helper.make_tensor_value_info(
            #     name=removed_input.name,  # 新输入的名称
            #     elem_type=removed_input.type.tensor_type.elem_type,
            #     shape=['None']  # 使用动态批处理形状
            # )

            # 将新的输入节点添加到模型中
            # model.graph.input.append(new_input)
            # print(f"New input shape: {model.graph.input}")
            pass
    for i in range(len(model.graph.output)):
        dim_proto = model.graph.output[i].type.tensor_type.shape.dim[0]
        dim_proto.dim_param = 'None'
    onnx.checker.check_model(model)
    onnx.save(model, onnx_model_new_path)

def new_linguistic_pitch_main():
    onnx_model_file = '/home/john/MuerSinger2/DiffSinger/onnxrun/onnx_model/baili_pitch'

    onnx_model_name = 'linguistic.onnx'
    onnx_model_new_name = 'batch_size_none.linguistic.onnx'
    
    onnx_model_path = os.path.join(onnx_model_file, onnx_model_name)
    onnx_model_new_path = os.path.join(onnx_model_file, onnx_model_new_name)

    model = onnx.load(onnx_model_path)
    
    pitchModel = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    input_details = pitchModel.get_inputs()
    for input_detail in input_details:
        print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
    output_details = pitchModel.get_outputs()
    for output_detail in output_details:
        print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
    
    input_dims = {
            "tokens": ['batch', 'n_tokens'],
            "word_div": ['batch', 'n_words'], # pitch 新增
            "word_dur": ['batch', 'n_words'] # pitch 新增
        }
    output_dims = {
            "encoder_out": ['batch', 'n_tokens', 256],
            "x_masks": ['batch', 'n_tokens']
        }
    variable_length_model = update_model_dims.update_inputs_outputs_dims(model, input_dims, output_dims)
    # need to check model after the input/output sizes are updated
    onnx.checker.check_model(variable_length_model)
    onnx.save(variable_length_model, onnx_model_new_path)

def new_pitch_main():
    onnx_model_file = '/home/john/MuerSinger2/DiffSinger/onnxrun/onnx_model/baili_pitch'
    
    onnx_model_name = 'pitch.onnx'
    onnx_model_new_name = 'batch_size_none.pitch.onnx'
    
    onnx_model_path = os.path.join(onnx_model_file, onnx_model_name)
    onnx_model_new_path = os.path.join(onnx_model_file, onnx_model_new_name)

    model = onnx.load(onnx_model_path)
    
    pitchModel = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
    input_details = pitchModel.get_inputs()
    for input_detail in input_details:
        print(f"Input Name: {input_detail.name}, Input Shape: {input_detail.shape}")
    output_details = pitchModel.get_outputs()
    for output_detail in output_details:
        print(f"Output Name: {output_detail.name}, Output Shape: {output_detail.shape}")
        
    input_dims = {
        "encoder_out": ['batch', 'n_tokens', 256],
        "ph_dur": ['batch', 'n_tokens'],
        "note_midi": ['batch', 'n_notes'],
        "note_rest": ['batch', 'n_notes'],
        "note_dur": ['batch', 'n_notes'],
        "pitch": ['batch', 'n_frames'],
        "expr": ['batch', 'n_frames'],
        "retake": ['batch', 'n_frames'], # pitch 新增
        "speedup": []
    }
    output_dims = {
        "pitch_pred": ['batch', 'n_frames']
    }
    variable_length_model = update_model_dims.update_inputs_outputs_dims(model, input_dims, output_dims)
    variable_length_model = change_init(variable_length_model, '/pre/smooth/Unsqueeze', '_new', '/lr/Constant_4_output_0', [1])
    variable_length_model = change_init(variable_length_model, '/pre/smooth/Squeeze', '_new', '/lr/Constant_4_output_0', [1])
    
    # need to check model after the input/output sizes are updated
    onnx.checker.check_model(variable_length_model)
    onnx.save(variable_length_model, onnx_model_new_path)

def new_linguistic_dur_main():
    onnx_model_file = '/home/john/MuerSinger2/DiffSinger/onnxrun/onnx_model/baili_dur'

    onnx_model_name = 'linguistic.onnx'
    onnx_model_new_name = 'batch_size_none.linguistic.onnx'
    
    onnx_model_path = os.path.join(onnx_model_file, onnx_model_name)
    onnx_model_new_path = os.path.join(onnx_model_file, onnx_model_new_name)

    model = onnx.load(onnx_model_path)
    input_dims = {
            "tokens": ['batch', 'n_tokens'],
            "word_div": ['batch', 'n_words'],
            "word_dur": ['batch', 'n_words']
        }
    output_dims = {
            "encoder_out": ['batch', 'n_tokens', 256],
            "x_masks": ['batch', 'n_tokens']
        }
    variable_length_model = update_model_dims.update_inputs_outputs_dims(model, input_dims, output_dims)
    # need to check model after the input/output sizes are updated
    onnx.checker.check_model(variable_length_model)
    onnx.save(variable_length_model, onnx_model_new_path)

def new_dur_main():
    onnx_model_file = '/home/john/MuerSinger2/DiffSinger/onnxrun/onnx_model/baili_dur'

    onnx_model_name = 'dur.onnx'
    onnx_model_new_name = 'batch_size_none.dur.onnx'
    
    onnx_model_path = os.path.join(onnx_model_file, onnx_model_name)
    onnx_model_new_path = os.path.join(onnx_model_file, onnx_model_new_name)

    model = onnx.load(onnx_model_path)
    input_dims = {
            "encoder_out": ['batch', 'n_tokens', 256],
            "x_masks": ['batch', 'n_tokens'],
            "ph_midi": ['batch', 'n_tokens']
        }
    output_dims = {
            "ph_dur_pred": ['batch', 'n_tokens']
        }
    variable_length_model = update_model_dims.update_inputs_outputs_dims(model, input_dims, output_dims)
    # need to check model after the input/output sizes are updated
    onnx.checker.check_model(variable_length_model)
    onnx.save(variable_length_model, onnx_model_new_path)
    
if __name__ == '__main__':
    # old_main()
    new_linguistic_pitch_main()
    new_pitch_main()
    
    new_linguistic_dur_main()
    new_dur_main()