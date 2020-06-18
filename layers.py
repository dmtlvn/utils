import tensorflow as tf
import lazylayers as L
import inspect
import copy
from tensorflow.keras import layers as K


def _join_layer(join):
    switch = {
        "add": L.Add,
        "mul": L.Multiply,
        "avg": L.Average,
        "drop": L.DropAvg,
        "concat": L.Concatenate,
    }
    return switch[join]


def _Activation(act):
    if isinstance(act, str):
        if act != "linear":
            return L.Activation(activation = act)
        else:
            return lambda x: x
    else:
        return act


def split_kwargs(kwargs, *classes):
    remaining_kwargs = copy.deepcopy(kwargs)
    class_kwargs = [dict() for C in classes]
    for k, v in list(remaining_kwargs.items()):
        for C_kwargs, C in zip(class_kwargs, classes):
            params = inspect.signature(C).parameters
            if k in params:
                C_kwargs[k] = v
        remaining_kwargs.pop(k)
    return class_kwargs, remaining_kwargs


class Identity(K.Layer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, inputs):
        return inputs


class Composition:
    """
    A series of sequentially connected modules, essentially a function 
    composition
    
    Parameters:
    
    layers (iterable) - sequence of modules, connected in the same order 
    """
        
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class Parallel:
    """
    A set of parallel layers applied to the same input
    
    Parameters:
    
    layers (iterable) - a set modules, to be applied in parallel
    """
        
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, inputs):
        """
        Returns a list if tensors
        """        
        return [layer(inputs) for layer in self.layers]
    
    
class SkipConnection:
    """
    A skip connection added to a given module
    
    Parameters:
    
    module - computing unit for the main branch
    
    join (str) - a type of merging operation between the main branch and the 
        skip connection. Default: "add"
        
    **kwargs - parameters of the join layer
    """
    
    def __init__(self, module, join = 'add', **kwargs):
        self.module = module
        self.join = _join_layer(join)(**kwargs) if isinstance(join, str) else join
        
    def __call__(self, inputs):
        y = self.module(inputs)
        y = self.join([y, inputs])
        return y
    
    
class ConvBlock:
    """
    A convolutional block consisting of convolutions, batchnorms and activations
    arranged in a specified order.
    
    Parameters:
    
    schema (str) - a string with single-character codes for layers:
            'A' - activation, 
            'B' - batchnorm,
            'C' - convolution
        Any whitespaces are ignored. Default: "CBA" (conv-bn-act)
        
    **kwargs - standard set of arguments for a `Conv2D` and `BatchNormalization`
        classes
    """
    
    def __init__(self, schema = "CBA", activation = "linear", **kwargs):    
        super().__init__()
        (conv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        
        conv = L.Conv2D(**conv_kwargs)
        bn = L.BatchNormalization(**bn_kwargs)
        act = _Activation(activation)
        
        alias = {"A": act, "B": bn, "C": conv}
        try:
            self.module = Composition([alias[c] for c in schema])
        except KeyError:
            raise ValueError("`schema` must be a string of 'A' (activation), 'B' (batchnorm), " \
                             "'C' (convolution) characters and whitespaces")   
    
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class DepthwiseConvBlock:
    """
    A depthwise convolution block consisting of depthwise convolutions, 
    batchnorms and activations arranged in a specified order.
    
    Parameters:
    
    schema (str) - a string with single-character codes for layers:
            'A' - activation, 
            'B' - batchnorm,
            'C' - depthwise convolution
        Any whitespaces are ignored. Default: "CBA" (dwconv-bn-act)
        
    **kwargs - standard set of arguments for a `DepthwiseConv2D` and 
        `BatchNormalization` classes
    """
    
    def __init__(self, schema = "CBA", activation = "linear", **kwargs):    
        super().__init__()
        (dwconv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        
        dwconv = L.DepthwiseConv2D(**dwconv_kwargs)
        bn = L.BatchNormalization(**bn_kwargs)
        act = _Activation(activation)
        
        alias = {"A": act, "B": bn, "C": dwconv}
        try:
            self.module = Composition([alias[c] for c in schema])
        except KeyError:
            raise ValueError("`schema` must be a string of 'A' (activation), 'B' (batchnorm), " \
                             "'C' (depthwise convolution) characters and whitespaces")   
    
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class FactorizedConv2D:
    """
    A 2D convolution factorized into a sequence of convolutions with a smaller 
    kernel. For example, a 5x5 convolution can be factorized as two consequitive
    3x3 convolutions or four 2x2 convolutions. For more details refer to an
    InceptionV2 paper (Fig. 5): https://arxiv.org/pdf/1512.00567.pdf
    
    Parameters:
    
    kernel_size (int, tuple) - base kernel size, essentially a field of view, of
        the block
        
    factor_size (int, tuple) - a factor-kernel shape
    
    **kwargs - a standard set of parameters of the `ConvBlock` class
    """
    
    def __init__(self, kernel_size, factor_size, **kwargs):
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        fh, fw = (factor_size, factor_size) if isinstance(factor_size, int) else factor_size
        
        if (kh - 1) % (fh - 1) != 0 or \
            (kw - 1) % (fw - 1) != 0 or \
            (kh - 1) // (fh - 1) != (kw - 1) // (fw - 1):
            raise ValueError(f"Kernel of size ({kh}, {kw}) cannot be factorized into " \
                             f"a sequence of kernels of size ({fh}, {fw})")
        if fh == 1 or fw == 1:
            raise ValueError(f"`factor_size` cannot be 1")
        
        n_layers = (kh - 1) // (fh - 1)
        conv = ConvBlock(kernel_size = (fh, fw), **kwargs)
        self.module = Composition([conv]*n_layers)
        
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class SpatialSeparableConv2D:
    """
    A 2D convolution factorized into a composition of two 1D convolutions in 
    the spatial domain, similar to how a depthwise-separable convolution is 
    factorized. For a kernel size HxWxC the resulting factorization will be a 
    composition of 1xWxC and Hx1xC convolutions. For more information refer 
    to the InceptionV2 paper (Fig. 6): https://arxiv.org/pdf/1512.00567.pdf
    
    Parameters:
    
    kernel_size (int, tuple) - base kernel size of the block
    
    row_first (bool) - if true, performs 1xW convolution first, otherwise Hx1
    
    filters_in (int) - number of filters in the first layer in the sequence, 
        depending on the `row_first` parameter
    
    **kwargs - a standard set of parameters of the `ConvBlock` class 
    """
    
    def __init__(self, filters, kernel_size, filters_in = None, row_first = True, **kwargs):        
        filters_in = filters if filters_in is None else filters_in
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size        
        if row_first:
            conv_1xN = ConvBlock(filters = filters_in, kernel_size = (1, kw), **kwargs)
            conv_Nx1 = ConvBlock(filters = filters, kernel_size = (kh, 1), **kwargs)
            self.module = Composition([conv_1xN, conv_Nx1])
        else:
            conv_Nx1 = ConvBlock(filters = filters_in, kernel_size = (kh, 1), **kwargs)
            conv_1xN = ConvBlock(filters = filters, kernel_size = (1, kw), **kwargs)
            self.module = Composition([conv_Nx1, conv_1xN])
        
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class FullSeparableConv2D:
    """
    A 2D convolution factorized into a composition of three 1D convolutions in 
    the both spatial and channel domains, similar to how a depthwise-separable 
    convolution is factorized. For a kernel size HxWxC the resulting 
    factorization will be a composition of 1xWx1, Hx1x1 and 1x1xC convolutions. 
    
    Parameters:
    
    kernel_size (int, tuple) - base kernel size of the block
    
    order (str) - a configuration string consisting of characters 'H' (Hx1), 
        'W' (1xW) and 'C' (1x1) characters, which determines the order of layers
        in the decomposition
    
    filters_in (int) - number
    
    **kwargs - a standard set of parameters of the `ConvBlock` and 
        `DepthwiseConvBlock` classes
    """
    
    def __init__(self, kernel_size, order = "HWC", schema = "CBA", **kwargs):        
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size  
        (conv_kwargs, dw_kwargs, bn_kwargs,), _ = split_kwargs(
            kwargs, L.Conv2D, L.DepthwiseConv2D, L.BatchNormalization
        )
        dw_kwargs = dict(schema = schema, **dw_kwargs, **bn_kwargs)
        pw_kwargs = dict(schema = schema, **conv_kwargs, **bn_kwargs)
        dwconv_1xN = DepthwiseConvBlock(kernel_size = (1, kw), **dw_kwargs)
        dwconv_Nx1 = DepthwiseConvBlock(kernel_size = (kh, 1), **dw_kwargs)
        pwconv_1x1 = ConvBlock(kernel_size = (1, 1), **pw_kwargs)
        alias = {"H": dwconv_Nx1, "W": dwconv_1xN, "C": pwconv_1x1}
        try:
            self.module = Composition([alias[c] for c in order])
        except KeyError:
            raise ValueError("`order` must be a string of 'HWC' characters")   
        
    def __call__(self, inputs):
        return self.module(inputs)
        
        
class SplitConv2D:
    """
    A 2D convolution splitted into two independent (parallel) 1D convolutions 
    and merged back together. For more information refer to the InceptionV2
    paper (Fig. 7): https://arxiv.org/pdf/1512.00567.pdf
    
    Parameters:
    
    kernel_size (int, tuple) - base kernel size of the block
    
    **kwargs - a standard set of parameters of the `ConvBlock` class 
    """
    
    def __init__(self, kernel_size, join = "concat", schema = "CBA", **kwargs):
        (conv_kwargs, bn_kwargs, join_kwargs), _ = split_kwargs(
            kwargs, L.Conv2D, L.BatchNormalization, _join_layer(join)
        )
        block_kwargs = dict(schema = schema, **conv_kwargs, **bn_kwargs)
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size        
        conv_1xN = ConvBlock(kernel_size = (1, kw), padding = "same", **block_kwargs)
        conv_Nx1 = ConvBlock(kernel_size = (kh, 1), padding = "same", **block_kwargs)
        kernel = Parallel([conv_1xN, conv_Nx1])
        join = _join_layer(join)(**join_kwargs)
        self.module = Composition([kernel, join])
        
    def __call__(self, inputs):
        return self.module(inputs)
        
    
class ResidualConv2D:
    """
    Resudial block. Consists of a sequence of convolutional blocks (ConvBlock 
    instances), followed by a merge with the skip-connection. For more details 
    refer to: https://arxiv.org/pdf/1603.05027.pdf
    
    Parameters:
        
    join (str) - type of a join between skip-connection and the main branch. 
        Default: 'add'.
        
    n_blocks (int) - number of ConvBlocks in the main branch. Default: 2.
    
    **kwargs - arguments for a ConvBlock. Default layer order in the ConvBlock 
        is 'CBA' (conv-bn-act). To create a full pre-activation module (Fig. 4e 
        from the paper) pass `schema = 'ABC'` (act-bn-conv) or `schema = 'BAC'` 
        (bn-act-conv). Every other parameter defaults to a standard Keras values
    """
    
    def __init__(self, join = 'add', n_blocks = 2, schema = "CBA", padding = "same", **kwargs):
        if isinstance(join, str):
            (conv_kwargs, join_kwargs), _ = split_kwargs(kwargs, L.Conv2D, _join_layer(join))
        else:
            (conv_kwargs,), _ = split_kwargs(kwargs, ConvBlock)
        conv = ConvBlock(schema = schema, padding = padding, **conv_kwargs)
        block = Composition([conv]*n_blocks)
        self.module = SkipConnection(block, join = join, **join_kwargs)
        
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class DenseNetBlock:
    """
    Densely connected sequence of layers. For more information refer to:
    https://arxiv.org/pdf/1608.06993.pdf
    
    Parameters:
    
    layers (iterable) - sequence of modules of the main branch, connected in the
        same order
        
    join (str, layer) - type of a join between skip-connection and the main 
        branch. Can be either a string or an actual layer object. To follow the 
        original paper, pass `join = 'concat'`. Default: 'concat'.
        
    **kwargs - a set of arguments for the join layer, in case it was specified 
        by the string
    """
    
    def __init__(self, layers, join = "concat", **kwargs):
        self.layers = layers
        self.join = _join_layer(join)(**kwargs) if isinstance(join, str) else join
        
    def __call__(self, inputs):      
        x = self.layers[0](inputs)
        skips = [x]
        for layer in self.layers[1:]:
            skips.append(layer(x))
            x = self.join(list(skips))
        return x
    
    
class DropAvg(K.Layer):
    """
    Implements an average of a random subset of the input tensors. Has two 
    strategies for sampling: 
        * global - selecting exactly one tensor; in this case no average is 
                   computed
        * local - selecting any non-empty subset of tensors. 
    For more details refer to: https://arxiv.org/pdf/1605.07648.pdf
    
    Parameters:
    
    rate (float/iterable) - probability of selecting tensors. If a single 
        number, it is used for all input tensors. If a sequence of numbers, it 
        is a separate probability for each tensors, in which case it must match 
        the number of input tensors.
    many (bool) - strategy selector. If true, samples a subset, otherwise 
        samples a single tensor.
    **kwargs - default keras.layers.Layer parameters
    """
    
    def __init__(self, rate = 0.5, is_global = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_global = is_global
        self.rate = tf.constant(rate, dtype = tf.float32)
        
    def local_strategy(self, inputs):
        n = len(inputs)
        stack = tf.stack(inputs, axis = -1)
        probs = tf.random.uniform((n,))
        probs = probs / tf.reduce_max(probs)
        idx = tf.cast(tf.where(probs > self.rate), tf.int32)
        keep = tf.gather(stack, idx, axis = -1)
        merge = tf.reduce_mean(keep, axis = (-2, -1))
        return merge
    
    def global_strategy(self, inputs):
        n = len(inputs)
        stack = tf.stack(inputs, axis = -1)
        idx = tf.random.int32(n)
        return stack[...,idx]
        
    def call(self, inputs):
        if isinstance(inputs, tf.Tensor):
            return inputs
        if self.is_global:
            return self.global_strategy(inputs)
        else:
            return self.local_strategy(inputs)
        
        
class FractalBlock:
    """
    A recursively defined module with a tree-like structure, where each level of
    the tree forms a parallel branch, and branches of the tree are connected via
    some merging operation. 
    For more details refer to: https://arxiv.org/pdf/1605.07648.pdf
    
    Because of an extensive number of layers (2**N) this module requires a lot 
    of memory, and makes practical sense with the DropPath procedure, proposed 
    in the paper, which selects only a subset of layers. This DropPath is used 
    by default. Refer to DropAvg class for more information
    
    Parameters:
    
    module - a base network module, used for construcion
    
    depth (int) - number of parallel branches / depth of the tree
    
    join (str, layer) - join method. Default: 'drop'
        
    **kwargs - arguments for the joining operation
    """
    
    def __init__(self, module, depth, join = "drop", **kwargs):
        self.module = module
        self.depth = depth
        self.join = _join_layer(join)(**kwargs) if isinstance(join, str) else join
        
    def __call__(self, inputs):  
        
        def build(branches, depth, subtree): 
            if depth == 0:
                branches[0] = self.module(branches[0])
                return
            build(branches, depth - 1, 1)
            build(branches, depth - 1, 0)
            branches[depth] = self.module(branches[depth])
            if subtree:
                x = self.join(branches[:depth+1])
                for i in range(depth+1):
                    branches[i] = x   
            return
        
        branches = [inputs]*self.depth
        build(branches, self.depth - 1, 1)
        return branches[0]
    
    
class SqueezeExcitation:
    """
    Channel attention mechanism. It is appended to a target module, and consists 
    of a global average pooling for squishing the spatial domain, a two-layer 
    subnetwork for generation of a modulating signal and a channelwise 
    multiplication with the original input for actual modulation/attention.
    For more information refer to: https://arxiv.org/pdf/1709.01507.pdf

    Parameters:
    
    reduction (int) - scaling factor of the inner layer of the module. Suppose, 
        input feature map has C channels, then the inner layer has C / reduction
        neurons, while the outer layer has C neurons. In most cases reduction is
        greater than 1.
        
    activation (str, layer) - activation of the last layer, which produces 
        attention coefficients
    """
    
    def __init__(self, reduction, activation = "sigmoid"):
        self.reduction = reduction
        self.activation = activation
        
    def __call__(self, inputs):
        _, _, _, C = inputs.shape
        squeeze = L.GlobalAveragePooling2D()
        F = L.Dense(units = int(C / self.reduction), activation = 'relu')
        G = L.Dense(units = C, activation = self.activation)
        R = L.Reshape(target_shape = (1, 1, C))
        excitation = Composition([squeeze, F, G, R])
        module = SkipConnection(excitation, join = "mul")
        return module(inputs)
    
                                                                                                    
class MobileNetV1Block: 
    """
    Depthwise separable convolution mixed with batchnorm.
    For more details refer to: https://arxiv.org/pdf/1704.04861.pdf
    
    Parameters:
    
    schema (str) - a string consisting of characters:
            'A' - activation
            'B' - batch normalization
            'C' - 1x1 convolution
            'D' - depthwise convolution
        Any whitespaces are ignored. Default: "DBACBA"
        
    filters (int) - number of the output channels of the 1x1 convcolution
    
    kernel_size (int/tuple) - kernel size of the depthwise convolution
    
    stride (int/tuple) - stride of the depthwize convolution
    
    **kwargs - standard set of arguments for a `Conv2D`, `DepthwiseConv2D` and 
        `BatchNormalization` classes
    """
    
    def __init__(
        self, kernel_size, strides = (1, 1), activation = 'linear', schema = "DBACBA", **kwargs
    ):    
        super().__init__()
        (conv_kwargs, dw_kwargs, bn_kwargs,), _ = split_kwargs(
            kwargs, L.Conv2D, L.DepthwiseConv2D, L.BatchNormalization
        )
        print(conv_kwargs, dw_kwargs, bn_kwargs)
        dwconv = L.DepthwiseConv2D(kernel_size = kernel_size, strides = strides, **dw_kwargs)
        conv = L.Conv2D(kernel_size = (1, 1), strides = (1, 1), **conv_kwargs)
        bn = L.BatchNormalization(**bn_kwargs)
        act = _Activation(activation)
        alias = {"A": act, "B": bn, "C": conv, "D": dwconv}
        try:
            self.module = Composition([alias[c] for c in schema.replace(" ", "")])
        except KeyError:
            raise ValueError("`schema` must be a string of 'A', 'B', 'C', 'D' characters " \
                             "and whitespaces")   
    
    def __call__(self, inputs):
        return self.module(inputs)

    
class MobileNetV2Block: 
    """
    Depthwise separable convolution mixed with batchnorm and linear bottlenecks.
    For more details refer to: https://arxiv.org/pdf/1704.04861.pdf
    
    Parameters:
    
    schema (str) - a string consisting of characters:
            'A' - activation
            'B' - batchnorm
            'C' - 1x1 convolution
            'D' - depthwise convolution
        Any whitespaces are ignored. For the inverted residual module from the 
        Fig. 4d use `schema = "CBADBALB". Default: "CBADBALB".
        
    filters (int) - number of the intermediate channels of the 1x1 non-linear 
        convcolution. Is usually several times bigger than the number of input 
        channels. Original value from the paper is `filters = 6*input_channels`
        
    out_filters (int) - number of output channels for the 1x1 linear convolution
    
    kernel_size (int/tuple) - kernel size of the depthwise convolution
    
    stride (int/tuple) - stride of the depthwize convolution
    
    **kwargs - standard set of arguments for a `Conv2D`, `DepthwiseConv2D` and 
        `BatchNormalization` classes
    """
    
    def __init__(
        self, 
        filters, 
        out_filters, 
        kernel_size, 
        strides = (1, 1), 
        activation = 'linear', 
        schema = "CBADBALB", 
        **kwargs
    ):    
        super().__init__()
        (conv_kwargs, dw_kwargs, bn_kwargs), _ = split_kwargs(
            kwargs, L.Conv2D, L.DepthwiseConv2D, L.BatchNormalization
        )
        conv_kwargs["kernel_size"] = (1, 1)
        conv_kwargs["strides"] = (1, 1)
        dwconv = L.DepthwiseConv2D(kernel_size = kernel_size, strides = strides, **dw_kwargs)
        conv = L.Conv2D(filters = filters, **conv_kwargs)
        linconv = L.Conv2D(filters = out_filters, **conv_kwargs)
        bn = L.BatchNormalization(**bn_kwargs)
        act = _Activation(activation)
        alias = {"A": act, "B": bn, "C": conv, "D": dwconv, "L": linconv}
        try:
            self.module = Composition([alias[c] for c in schema.replace(" ", "")])
        except KeyError:
            raise ValueError("`schema` must be a string of 'A', 'B, 'C', 'D', 'L' characters " \
                             "and white-spaces")   
    
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class MobileNetV3Block: 
    """
    A simple configurator of a MobileNet v3 module consisting of epthwise 
    separable convolutions mixed with batchnorm, linear bottlenecks and a 
    squeeze-excitation module. 
    For more details refer to: https://arxiv.org/pdf/1905.02244.pdf.
    
    Configuration is done via aliasing layers with string codes and sharing most
    of rarely used common hyper-parameters between layers of the same type. For 
    a more specific design better consider using a `Composition` module.
    
    Parameters:
    
    schema (str) - a string consisting of characters:
            'A' - activation
            'B' - batch normalization
            'C' - 1x1 non-linear convolution, 
            'D' - depthwise convolution
            'L' - 1x1 linear convolution
            'S' - squeeze-excitation module
        Whitespaces are ignored. Default: "CBA DBA S LB"
    
    filters (int) - number of the intermediate channels of the 1x1 non-linear 
        convcolution. Is usually several times bigger than the number of input 
        channels. Original value from the paper is `filters = 6*input_channels`
        
    out_filters (int) - number of output channels for the 1x1 linear convolution
    
    kernel_size (int/tuple) - kernel size of the depthwise convolution
    
    strides (int/tuple) - stride of the depthwize convolution. Default: (1, 1)
    
    se_factor (float) - dimensionality reduction parameter in the 
        Squeeze-Excitation block. Default: 4.0.
        
    **kwargs - standard set of arguments for a `Conv2D`, `DepthwiseConv2D` and 
        `BatchNormalization` classes
    """
    
    def __init__(
        self,
        filters, 
        out_filters, 
        kernel_size, 
        strides = (1, 1), 
        activation = 'linear', 
        se_factor = 4.0,
        schema = "CBADBASLB", 
        **kwargs
    ):    
        super().__init__()
        (conv_kwargs, dw_kwargs, bn_kwargs), _ = split_kwargs(
            kwargs, L.Conv2D, L.DepthwiseConv2D, L.BatchNormalization
        )
        conv_kwargs["kernel_size"] = (1, 1)
        conv_kwargs["strides"] = (1, 1)        
        dwconv = L.DepthwiseConv2D(kernel_size = kernel_size, strides = strides, **dw_kwargs)
        conv = L.Conv2D(filters = filters, **conv_kwargs)
        linconv = L.Conv2D(filters = out_filters, **conv_kwargs)
        bn = L.BatchNormalization(**bn_kwargs)
        act = _Activation(activation)
        se = SqueezeExcitation(reduction = se_factor)
        alias = {"A": act, "B": bn, "C": conv, "D": dwconv, "L": linconv, "S": se}
        try:
            self.module = Composition([alias[c] for c in schema.replace(" ", "")])
        except KeyError:
            raise ValueError("`schema` must be a string of 'ABCDLS' characters and white-spaces")   
    
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class InceptionV1A:
    """
    A parallel set of 1x1, 3x3, 5x5 convolutions and 3x3 max-pooling preceded by
    a 1x1 bottlenecks for dimensionality reduction. 
    For more information refer to: https://arxiv.org/pdf/1409.4842.pdf
    
    Parameters:
    
    filters_1x1 (int) - number of filters in a 1x1 branch
    
    filters_3x3 (int) - number of filters in a 3x3 branch
    
    filters_5x5 (int) - number of filters in a 5x5 branch 
    
    filters_pool (int) - number of in a pooling branch
    
    inputs_3x3 (int) - number of filters in a 3x3 bottleneck
    
    inputs_3x3 (int) - number of filters in a 5x5 bottleneck
    
    **kwargs - standard set of arguments for a `ConvBlock` class
    """
    
    def __init__(
        self, 
        filters_1x1, 
        filters_3x3, 
        filters_5x5, 
        filters_pool, 
        inputs_3x3,
        inputs_5x5,
        strides = (1, 1),
        padding = "same",
        schema = "CBA",
        **kwargs
    ):    
        super().__init__()
        (conv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        pw_kwargs = dict(
            strides = (1, 1), kernel_size = (1, 1), schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_kwargs = dict(
            strides = strides, padding = padding, schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_1x1 = ConvBlock(filters = filters_1x1, kernel_size = (1, 1), **conv_kwargs)
        conv_3x3 = ConvBlock(filters = filters_3x3, kernel_size = (3, 3), **conv_kwargs)
        conv_5x5 = ConvBlock(filters = filters_5x5, kernel_size = (5, 5), **conv_kwargs)
        reduction_3x3 = ConvBlock(filters = inputs_3x3, **pw_kwargs)
        reduction_5x5 = ConvBlock(filters = inputs_5x5, **pw_kwargs)
        reduction_pool = ConvBlock(filters = filters_pool, **pw_kwargs)
        pool = L.MaxPooling2D(pool_size = (3, 3), strides = strides, padding = padding)
        self.module = Composition([
            Parallel([
                conv_1x1,
                Composition([reduction_3x3, conv_3x3]),
                Composition([reduction_5x5, conv_5x5]),
                Composition([pool, reduction_pool]),
            ]),
            L.Concatenate(axis = -1)
        ])
        
    def __call__(self, inputs):
        return self.module(inputs)