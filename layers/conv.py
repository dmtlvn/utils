import tensorflow as tf
import inspect
import warnings
import copy

from tensorflow import keras as K
from tensorflow.python.keras.utils import conv_utils
from functools import partial

from . import lazylayers as L
from .helpers import split_kwargs, _Activation, _join_layer
from .primitives import Composition, Parallel, SkipConnection

   
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
    
    def __init__(self, kernel_size, factor_size, strides = (1, 1), **kwargs):
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
        stride_conv = ConvBlock(kernel_size = (fh, fw), strides = strides, **kwargs)
        self.module = Composition([conv]*(n_layers - 1) + [stride_conv])
        
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
    
    def __init__(
        self,
        filters, 
        kernel_size, 
        filters_in = None, 
        row_first = True, 
        strides = (1, 1), 
        **kwargs
    ):        
        filters_in = filters if filters_in is None else filters_in
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size        
        if row_first:
            conv_1xN = ConvBlock(filters = filters_in, kernel_size = (1, kw), **kwargs)
            conv_Nx1 = ConvBlock(
                filters = filters, kernel_size = (kh, 1), strides = strides, **kwargs
            )
            self.module = Composition([conv_1xN, conv_Nx1])
        else:
            conv_Nx1 = ConvBlock(filters = filters_in, kernel_size = (kh, 1), **kwargs)
            conv_1xN = ConvBlock(
                filters = filters, kernel_size = (1, kw), strides = strides, **kwargs
            )
            self.module = Composition([conv_Nx1, conv_1xN])
        
    def __call__(self, inputs):
        return self.module(inputs)
    
    
class FlattenedConv2D:
    """
    A 2D convolution factorized into a composition of three 1D convolutions in 
    both spatial and channel domains, similar to how a depthwise-separable 
    convolution is factorized. For a kernel size HxWxC the resulting 
    factorization will be a composition of 1xWx1, Hx1x1 and 1x1xC convolutions.
    For more information refer to: https://arxiv.org/pdf/1412.5474.pdf
    
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
        conv = ConvBlock(schema = schema, padding = padding, strides = (1, 1), **conv_kwargs)
        block = Composition([conv]*n_blocks)
        self.module = SkipConnection(block, join = join, **join_kwargs)
        
    def __call__(self, inputs):
        return self.module(inputs)
    
    
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
    
    def __init__(self, reduction, activation = "sigmoid", channels_first = False, **kwargs):
        self.channel_axis = -1 if not channels_first else 1
        self.reduction = reduction
        self.activation = activation
        self.kwargs = dict(kernel_size = 1, **kwargs)
        
    def __call__(self, inputs):
        data_format = "channels_first" if self.channel_axis == 1 else "channels_last"
        out_filters = inputs.shape[self.channel_axis]
        in_filters = int(out_filters / self.reduction)
        target_shape = [1, 1, out_filters] if self.channel_axis == -1 else [out_filters, 1, 1]
        squeeze = L.GlobalAvgPool2D(data_format)
        F = L.Conv2D(filters = in_filters, activation = 'relu', **self.kwargs)
        G = L.Conv2D(filters = out_filters, activation = self.activation, **self.kwargs)
        excitation = Composition([squeeze, F, G])
        module = SkipConnection(excitation, join = "mul")
        return module(inputs)
    
    
class ResNeXtBlock:
    """
    ResNeXt module, which is essentialy a grouped convolution sandwiched bitween
    two pointwise convolutions for dimensionality reduction and expansion and a
    skip connection. 
    For more information refer to: https://arxiv.org/pdf/1611.05431.pdf
    
    Parameters:
    
    filters (int) - number of filters in the first 1x1 convolution as well as in
        the grouped convolution layer
        
    groups (int) - number of groups in the grouped convolution 
    
    kernel_size (int) - kernel size of the grouped convolution
    
    schema (str) - module layout config string with letters representing layers 
        in a sequence:
            A - activation
            B - batch normalization
            C - contraction pointwise conv
            E - expansion pointwise conv
            G - grouped convolution
        All whitespaces are ignored. Default: "CBAGBAEBA"
        
    join (str, layer) - a join operation for a skip connection. Default: "add"
    
    **kwargs - a standard set of parameters for the Conv2D, GroupConv2D, 
        BatchNormalization and join layers
    """
    
    def __init__(
        self, 
        filters, 
        groups, 
        kernel_size, 
        activation = "linear", 
        schema = "CBAGBAEBA", 
        data_format = "channels_last",
        join = "add",
        **kwargs
    ):
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"Invalid data format `{data_format}`")
        self.channel_axis = -1 if data_format == "channels_last" else 1
        (conv_kwargs, gconv_kwargs, bn_kwargs, join_kwargs), _ = split_kwargs(
            kwargs, L.Conv2D, GroupConv2D, L.BatchNormalization, _join_layer(join)
        )
        conv_kwargs["activation"] = activation
        conv_kwargs["data_format"] = data_format
        conv_kwargs["strides"] = (1, 1)
        self.bn = L.BatchNormalization(**bn_kwargs)
        self.conv_compress = L.Conv2D(
            filters = filters,
            kernel_size = (1, 1),
            **conv_kwargs
        )
        self.conv_expand = partial(L.Conv2D, kernel_size = (1, 1), **conv_kwargs)
        self.gconv = GroupConv2D(
            filters = filters, 
            groups = groups, 
            strides = (1, 1),
            kernel_size = kernel_size, 
            data_format = data_format, 
            **gconv_kwargs
        )
        self.act = _Activation(activation)
        self.join = _join_layer(join)(**join_kwargs)
        self.schema = schema.replace(" ", "")
        
    def __call__(self, inputs):
        channels = inputs.shape[self.channel_axis]
        alias = {
            "A": self.act, 
            "B": self.bn, 
            "C": self.conv_compress, 
            "E": self.conv_expand(filters = channels), 
            "G": self.gconv
        }
        module = SkipConnection(
            Composition([alias[c] for c in self.schema]), 
            join = self.join,
        )
        outputs = module(inputs)
        return outputs
     
        
class ShuffleNetBlock:
    """
    A main block from the ShuffleNet, similar to MobileNetV3 block, but point-
    wise convolutions are replaced with group convolutions and a novel channel
    shuffle operation which induces inter-group information flow. For more 
    information refer to: https://arxiv.org/pdf/1707.01083.pdf
    
    Parameters:
    
    filters_in (int) - number of filters in the input bottleneck 1x1 group
        convolution
    
    filters_out (int) - number of filters in the output expansion 1x1 
        group convolution
        
    groups (int) number of groups in both bottleneck and expansion convolutions
    
    schema (str) - module layout config string with letters representing layers 
        in a sequence:
            A - [A]ctivation
            B - [B]atch normalization
            C - [C]ontraction pointwise conv
            D - [D]epthwise convolution
            E - [E]xpansion pointwise conv
            S - [S]huffle
        All whitespaces are ignored. Default: "CBASDBEB"
        
    join (str, layer) - a join operation for a residual connection. If stride is
        greater than 1, adds a 3x3 average pooling layer on the skip connection 
        with the specified stride
        
    **kwargs - a standard set of paramters for GroupConv2D, DepthwiseConv2D,
        BatchNormalization and join layers
    """
    
    def __init__(
        self, 
        filters_in,
        filters_out,
        groups, 
        kernel_size, 
        strides = (1, 1),
        padding = "same",
        activation = "linear", 
        schema = "CBASDBEB", 
        data_format = "channels_last",
        join = "concat",
        **kwargs
    ):
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"Invalid data format `{data_format}`")
        (gconv_kwargs, dwconv_kwargs, bn_kwargs, join_kwargs), _ = split_kwargs(
            kwargs, L.GroupConv2D, L.DepthwiseConv2D, L.BatchNormalization, _join_layer(join)
        )
        gconv_kwargs["data_format"] = data_format
        gconv_kwargs["groups"] = groups
        gconv_kwargs["kernel_size"] = (1, 1)
        bn = L.BatchNormalization(**bn_kwargs)
        conv_compress = L.GroupConv2D(filters = filters_in, **gconv_kwargs)
        conv_expand = L.GroupConv2D(filters = filters_out, **gconv_kwargs)
        dwconv = L.DepthwiseConv2D(
            strides = strides, 
            kernel_size = kernel_size, 
            padding = padding,
            data_format = data_format, 
            **dwconv_kwargs
        )
        act = _Activation(activation)
        shuffle = L.GroupShuffle(groups = groups, data_format = data_format)
        join = _join_layer(join)(**join_kwargs)
        pool = L.AveragePooling2D(
            pool_size = (3, 3), strides = strides, padding = padding, data_format = data_format
        )
        alias = {
            "A": act,
            "B": bn,
            "C": conv_compress,
            "D": dwconv,
            "E": conv_expand,
            "S": shuffle,
        }
        main = Composition([alias[c] for c in schema.replace(" ", "")])
        if strides == (1, 1):
            self.module = Composition([
                SkipConnection(main, join = join),
                act,
            ])
        else:
            self.module = Composition([
                Parallel([main, pool]),
                join,
                act,
            ])
            
    def __call__(self, inputs):
        return self.module(inputs)