from . import lazylayers as L
from .helpers import split_kwargs, _Activation, _join_layer
from .primitives import Composition, Parallel, SkipConnection


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
        (conv_kwargs, dw_kwargs, bn_kwargs,), _ = split_kwargs(
            kwargs, L.Conv2D, L.DepthwiseConv2D, L.BatchNormalization
        )
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
            'C' - non-linear 1x1 convolution
            'D' - depthwise convolution
            'L' - linear 1x1 convolution
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
    A simple configurator of a MobileNet v3 module consisting of depthwise 
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
            'C' - non-linear 1x1 convolution, 
            'D' - depthwise convolution
            'L' - linear 1x1 convolution
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
