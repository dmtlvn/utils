import warnings
import numpy as np
from copy import deepcopy
from tensorflow.keras.models import Model


def is_single(input_fovs, input_strides):
    if len(input_fovs) != 1 or len(input_strides) != 1:
        raise ValueError(f"Invalid number of input connections ({len(input_fovs)}) "
                         f"for layer `{type(layer).__name__}`")
    input_fov = input_fovs[0]
    input_stride = input_strides[0]
    return input_fov, input_stride


def conv2d_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)
    input_fov = np.array(input_fov, dtype = np.float32)
    input_stride = np.array(input_stride, dtype = np.float32)
    kernel_size = np.array(layer.kernel_size, dtype = np.float32)
    dilation_rate = np.array(layer.dilation_rate, dtype = np.float32)
    effective_size = kernel_size + (kernel_size - 1)*(dilation_rate - 1)
    expansion = effective_size + (effective_size - 1)*(input_stride - 1)
    fov = input_fov + expansion - 1
    strides = input_stride * np.array(layer.strides, dtype = np.float32) 
    return tuple(fov), tuple(strides)


def conv2dtranspose_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)
    input_fov = np.array(input_fov, dtype = np.float32)
    input_stride = np.array(input_stride, dtype = np.float32)
    kernel_size = np.array(layer.kernel_size, dtype = np.float32)
    dilation_rate = np.array(layer.dilation_rate, dtype = np.float32)
    effective_size = kernel_size + (kernel_size - 1)*(dilation_rate - 1)
    expansion = effective_size + (effective_size - 1)*(input_stride - 1)
    fov = input_fov + expansion - 1
    strides = input_stride / np.array(layer.strides, dtype = np.float32) 
    return tuple(fov), tuple(strides)

    
def pool2d_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)    
    input_fov = np.array(input_fov, dtype = np.float32)
    input_stride = np.array(input_stride, dtype = np.float32)
    pool_size = np.array(layer.pool_size, dtype = np.float32)    
    effective_size = pool_size + (pool_size - 1)*(input_stride - 1)
    fov = input_fov + effective_size - 1
    strides = input_stride * np.array(layer.strides, dtype = np.float32) 
    return tuple(fov), tuple(strides)


def globalpool2d_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)    
    input_fov = np.array(input_fov, dtype = np.float32)
    input_stride = np.array(input_stride, dtype = np.float32)
    if layer.data_format == "channels_last":
        pool_size = np.array(layer.input.shape[1:3], dtype = np.float32)
    else:
        pool_size = np.array(layer.input.shape[2:4], dtype = np.float32)
    effective_size = pool_size + (pool_size - 1)*(input_stride - 1)
    fov = input_fov + effective_size - 1
    return tuple(fov), tuple(input_stride)


def downsample_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides) 
    input_stride = np.array(input_stride, dtype = np.float32)
    strides = input_stride * np.array(layer.size, dtype = np.float32) 
    return tuple(input_fov), tuple(strides)


def upsample_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)    
    input_stride = np.array(input_stride, dtype = np.float32)
    strides = input_stride / np.array(layer.size, dtype = np.float32) 
    return tuple(input_fov), tuple(strides)


def resize_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)    
    input_stride = np.array(input_stride, dtype = np.float32)
    target_size = np.array(layer.size)
    if layer.channels_first:
        input_size = np.array(layer.input.shape[2:4], dtype = np.float32)
    else:
        input_size = np.array(layer.input.shape[1:3], dtype = np.float32)
    strides = input_stride * input_size / target_size 
    return tuple(input_fov), tuple(strides)


def merge_fov(layer, input_fovs, input_strides):
    input_fovs = np.array(input_fovs, dtype = np.float32)
    input_strides = np.array(input_strides, dtype = np.float32)
    fov = np.max(input_fovs, axis = 0)
    strides = np.max(input_strides, axis = 0)
    return tuple(fov), tuple(strides)   
    
    
def bypass_fov(layer, input_fovs, input_strides):
    input_fov, input_stride = is_single(input_fovs, input_strides)  
    return tuple(input_fov), tuple(input_stride)


_CONV_LAYERS = [
    # custom layers
    "GroupConv2D",    
    # keras layers
    "Conv2D",
    "Convolution2D",
    "DepthwiseConv2D",
    "SeparableConv2D",
    "SeparableConvolution2D",
    # "LocallyConnected2D",
]
_POOL_LAYERS = [
    "MaxPooling2D",
    "MaxPool2D",
    "AveragePooling2D",
    "AvgPool2D",
]
_GLOBAL_POOL_LAYERS = [
    "GlobalAveragePooling2D",
    "GlobalAvgPool2D",
    "GlobalMaxPooling2D",
    "GlobalMaxPool2D",
]
_MERGE_LAYERS = [
    # custom layers
    "DropAvg"    
    # keras layers
    "Add",
    "Average",
    "Maximum",
    "Minimum",
    "Multiply",
    "Subtract",
    "Concatenate",
]
_BYPASS_LAYERS = [
    # custom layers
    "Identity",
    "GroupShuffle",    
    # keras layers
    "ZeroPadding2D",
    "BatchNormalization",
    "LayerNormalization",
    "Activation",
    "ReLU",
    "ELU",
    "LeakyReLU",
    "PReLU",
    "Softmax",
    "ThresholdedReLU",
    "Dropout",
    "AlphaDropout",
    "SpatialDropout2D",
    "GaussianDropout",
    "GaussianNoise",
    "Cropping2D",
    "ActivityRegularization",
]
_UNDEFINED_LAYERS = [
    "ConvLSTM2D",
    
    "Lambda",
    "Dense",    
    "Flatten",    
    "Permute",
    "RepeatVector",
    "Reshape",    
    "Dot",
    
    "Conv1D",
    "Convolution1D",
    "Cropping1D",
    "SeparableConv1D",
    "SeparableConvolution1D",
    "UpSampling1D",
    "ZeroPadding1D",
    "SpatialDropout1D",
    "LocallyConnected1D",
    "AveragePooling1D",
    "AvgPool1D",
    "GlobalAveragePooling1D",
    "GlobalAvgPool1D",
    "GlobalMaxPooling1D",
    "GlobalMaxPool1D",
    "MaxPooling1D",
    "MaxPool1D",
    
    "Conv3D",
    "Convolution3D",
    "Conv3DTranspose",
    "Convolution3DTranspose",
    "Cropping3D",
    "UpSampling3D",    
    "ZeroPadding3D",
    "SpatialDropout3D",
    "AveragePooling3D",
    "AvgPool3D",
    "GlobalAveragePooling3D",
    "GlobalAvgPool3D",
    "GlobalMaxPooling3D",
    "GlobalMaxPool3D",
    "MaxPooling3D",
    "MaxPool3D",
    
    "Masking",    
    "AdditiveAttention",
    "Attention",
    "Embedding",
    "AbstractRNNCell",
    "RNN",
    "SimpleRNN",
    "SimpleRNNCell",
    "StackedRNNCells",
    "GRU",
    "GRUCell",
    "LSTM",
    "LSTMCell",
]


_DISPATCHER = {
    **{layer: conv2d_fov for layer in _CONV_LAYERS},
    **{layer: pool2d_fov for layer in _POOL_LAYERS},
    **{layer: merge_fov for layer in _MERGE_LAYERS},
    **{layer: bypass_fov for layer in _BYPASS_LAYERS},
    **{layer: globalpool2d_fov for layer in _GLOBAL_POOL_LAYERS},
    "Resize": resize_fov,
    "UpSampling2D": upsample_fov,
    "DownSampling2D": downsample_fov,
    "Conv2DTranspose": conv2dtranspose_fov,
    "Convolution2DTranspose": conv2dtranspose_fov,  
}


def model_to_graph(model):
    
    def expand_submodels(model):
        nodes = dict()
        for layer in model.layers:
            if isinstance(layer, Model):
                sub_nodes = expand_submodel(layer, nodes)
                nodes.update(sub_nodes)
            else:
                if isinstance(layer.input, list):
                    inputs = [t.name for t in layer.input]
                else:
                    inputs = [layer.input.name]
                if isinstance(layer.output, list):
                    outputs = [t.name for t in layer.output]
                else:
                    outputs = [layer.output.name]
                nodes[layer.name] = (inputs, outputs)
        return nodes
            
    def drop_inputs(model, nodes):
        for t in model.inputs:
            for name, (_, outputs) in list(nodes.items()):
                if t.name in outputs:
                    del nodes[name]
        return nodes
    
    def build_graph(nodes):
        adjacency_list = dict()
        for a_name, (a_inputs, a_outputs) in nodes.items():
            adjacency_list[a_name] = {"in": set(), "out": set()}
            for b_name, (b_inputs, b_outputs) in nodes.items():   
                adjacency_list[a_name]["in"].update(
                    {b_name for a_inp in a_inputs if a_inp in b_outputs}
                ) 
                adjacency_list[a_name]["out"].update(
                    {b_name for a_out in a_outputs if a_out in b_inputs}
                )
        return adjacency_list
    
    nodes = expand_submodels(model)
    nodes = drop_inputs(model, nodes)
    adjacency_list = build_graph(nodes)           
    return adjacency_list


def topological_sort(adjacency_map, copy = True):
    if copy:
        adjacency_map = deepcopy(adjacency_map)
    source_nodes = {n for n in adjacency_map if not adjacency_map[n]["in"]}
    sorted_nodes = list()
    while source_nodes:
        n = source_nodes.pop()
        sorted_nodes.append(n)
        for m in adjacency_map[n]['out']:
            adjacency_map[m]['in'].remove(n)
            if not adjacency_map[m]["in"]:
                source_nodes.add(m)
    return sorted_nodes


def estimate_fov(model, target_layer = None):
    """
    Computes an analytical estimation of the size of the receptive field of the 
    model a.k.a. field of view (FOV). FOV is calculated by consequitively 
    propagating FOVs from input layers through the model graph to output layers.
    For some layers (Dense, Reshape, Permute, LSTM etc.) the notion of FOV is 
    not well-defined, however they can be used along with properly defined. 
    layers. For this reason FOV propagation will stop at such layers, producing 
    a relevant warning. Layers for which FOV cannot be inferred will return 
    FOV = None. 
    NOTE: At the moment FOV is computed for 2D operations only.
        
    Parameters:
    
    model (Model) - a Keras model object
    
    target_layer (str, list) - a name of the layer, or a list of layer names, in
        the `model` to compute FOV for. If None, FOV will be computed for the 
        output layers (layers without any outbound connections). Default: None
    """
    adjacency_list = model_to_graph(model)
    output_nodes = [k for k, v in adjacency_list.items() if not v["out"]]
    sorted_nodes = topological_sort(adjacency_list, copy = True)
    node_fovs = {n: {"fovs": [], "strides": []} for n in sorted_nodes}
    undefined = list()
    for name in sorted_nodes:
        fovs = node_fovs[name]["fovs"]
        strides = node_fovs[name]["strides"]            
        layer = model.get_layer(name)
        layer_id = type(layer).__name__
        if layer_id in _UNDEFINED_LAYERS or (None, None) in fovs:
            fov, stride = (None, None), (None, None)
            undefined.append(layer_id)
        else:
            if not fovs:
                fovs.append((1, 1))
            if not strides:
                strides.append((1, 1))
            fov, stride = _DISPATCHER[layer_id](layer, fovs, strides)
        node_fovs[name]["result"] = fov
        for output in adjacency_list[name]["out"]:
            node_fovs[output]["fovs"].append(fov)
            node_fovs[output]["strides"].append(stride)
            
    if undefined:
        warnings.warn(f"Undefined FOV for layers {undefined}. Partial result is returned")
    
    if target_layer is not None:
        if isinstance(target_layer, str):
            return node_fovs[target_layer]["result"]
        else:
            return [node_fovs[name]["result"] for name in target_layer]
    else:
        if len(output_nodes) == 1:
            return node_fovs[output_nodes[0]]["result"]
        else:
            return [node_fovs[name]["result"] for name in output_nodes]
