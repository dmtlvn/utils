import tensorflow as tf
import lazylayers as L
import inspect
import warnings
import copy

from tensorflow import keras as K
from tensorflow.python.keras.utils import conv_utils
from functools import partial

from .helpers import split_kwargs
from .common import _Activation, _join_layer
from .primitives import Composition, Parallel, SkipConnection
from .conv import ConvBlock, FactorizedConv2D, SpatialSeparableConv2D


class InceptionV1A:
    """
    A parallel set of 1x1, 3x3, 5x5 convolutions and 3x3 max-pooling preceded by
    a 1x1 bottlenecks for dimensionality reduction. 
    For more information refer to: https://arxiv.org/pdf/1409.4842.pdf
    
    Parameters:
    
    filters_1x1 (int) - number of filters in a 1x1 branch
    
    filters_3x3 (int) - number of filters in a 3x3 branch
    
    filters_5x5 (int) - number of filters in a 5x5 branch 
    
    filters_pool (int) - number of filters in a pooling branch
    
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
        schema = "CBA",
        **kwargs
    ):    
        (conv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        pw_kwargs = dict(
            strides = (1, 1), kernel_size = (1, 1), schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_kwargs = dict(
            strides = strides, padding = "same", schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_1x1 = ConvBlock(filters = filters_1x1, kernel_size = (1, 1), **conv_kwargs)
        conv_3x3 = ConvBlock(filters = filters_3x3, kernel_size = (3, 3), **conv_kwargs)
        conv_5x5 = ConvBlock(filters = filters_5x5, kernel_size = (5, 5), **conv_kwargs)
        reduction_3x3 = ConvBlock(filters = inputs_3x3, **pw_kwargs)
        reduction_5x5 = ConvBlock(filters = inputs_5x5, **pw_kwargs)
        reduction_pool = ConvBlock(filters = filters_pool, **pw_kwargs)
        pool = L.MaxPooling2D(pool_size = (3, 3), strides = strides, padding = "same")
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
    
    
class InceptionV2A:
    """
    A parallel set of 1x1, 3x3, 5x5 convolutions and 3x3 max-pooling preceded by
    a 1x1 bottlenecks for dimensionality reduction. A 5x5 convolution is 
    factorized into a sequence of two 3x3 convolutions. For more information 
    refer to Fig. 5 in: https://arxiv.org/pdf/1409.4842.pdf
    
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
        schema = "CBA",
        **kwargs
    ):    
        (conv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        pw_kwargs = dict(
            strides = (1, 1), kernel_size = (1, 1), schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_kwargs = dict(
            strides = strides, padding = "same", schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_1x1 = ConvBlock(filters = filters_1x1, kernel_size = (1, 1), **conv_kwargs)
        conv_3x3 = ConvBlock(filters = filters_3x3, kernel_size = (3, 3), **conv_kwargs)
        conv_5x5 = FactorizedConv2D(
            filters = filters_5x5, kernel_size = (5, 5), factor_size = (3, 3), **conv_kwargs
        )
        reduction_3x3 = ConvBlock(filters = inputs_3x3, **pw_kwargs)
        reduction_5x5 = ConvBlock(filters = inputs_5x5, **pw_kwargs)
        reduction_pool = ConvBlock(filters = filters_pool, **pw_kwargs)
        pool = L.MaxPooling2D(pool_size = (3, 3), strides = strides, padding = "same")
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
    
    
class InceptionV2B:
    """
    A parallel set of 1x1, NxN, 2Nx2N convolutions and 3x3 max-pooling preceded 
    by a 1x1 bottlenecks for dimensionality reduction. Both NxN and 2Nx2N 
    convolutions are factorized into asymmetric 1xN and Nx1 convolutions. 
    A 2Nx2N kernel is further factorized into a sequence of NxN convolutions. 
    For more information refer to Fig. 6 in: https://arxiv.org/pdf/1409.4842.pdf
    As paper suggests, this module should not be used early in the network.
    
    Parameters:
    
    filters_1x1 (int) - number of filters in a 1x1 branch
    
    filters_3x3 (int) - number of filters in a NxN branch
    
    filters_5x5 (int) - number of filters in a 2Nx2N branch 
    
    filters_pool (int) - number of in a pooling branch
    
    inputs_3x3 (int) - number of filters in a NxN bottleneck
    
    inputs_3x3 (int) - number of filters in a 2Nx2N bottleneck
    
    kernel_size (int) - base kernel size N for factorization
    
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
        kernel_size = (3, 3),
        strides = (1, 1),
        schema = "CBA",
        **kwargs
    ):    
        (conv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        pw_kwargs = dict(kernel_size = (1, 1), schema = schema, **conv_kwargs, **bn_kwargs)
        conv_kwargs = dict(padding = "same", schema = schema, **conv_kwargs, **bn_kwargs)
        conv_1x1 = ConvBlock(
            filters = filters_1x1, kernel_size = (1, 1), strides = strides, **conv_kwargs
        )
        conv_3x3 = SpatialSeparableConv2D(
            filters = filters_3x3, kernel_size = kernel_size, strides = strides, **conv_kwargs
        )
        conv_5x5_1 = SpatialSeparableConv2D(
            filters = filters_5x5, kernel_size = kernel_size, **conv_kwargs
        )
        conv_5x5_2 = SpatialSeparableConv2D(
            filters = filters_5x5, kernel_size = kernel_size, strides = strides, **conv_kwargs
        )
        conv_5x5 = Composition([conv_5x5_1, conv_5x5_2])
        reduction_3x3 = ConvBlock(filters = inputs_3x3, **pw_kwargs)
        reduction_5x5 = ConvBlock(filters = inputs_5x5, **pw_kwargs)
        reduction_pool = ConvBlock(filters = filters_pool, **pw_kwargs)
        pool = L.MaxPooling2D(pool_size = (3, 3), strides = strides, padding = "same")
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
    
    
class InceptionV2C:
    """
    A parallel set of 1x1, 3x3, 5x5 convolutions and 3x3 max-pooling preceded by
    a 1x1 bottlenecks for dimensionality reduction. Both 5x5 and 3x3 
    convolutions are factorized into 1xN and Nx1 split-convolutions. A 5x5
    kernel is further factorized into a sequence of 3x3 convolutions. For more 
    information refer to Fig. 6 in: https://arxiv.org/pdf/1409.4842.pdf
    
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
        kernel_size = (3, 3),
        strides = (1, 1),
        schema = "CBA",
        **kwargs
    ):    
        (conv_kwargs, bn_kwargs), _ = split_kwargs(kwargs, L.Conv2D, L.BatchNormalization)
        pw_kwargs = dict(
            strides = (1, 1), kernel_size = (1, 1), schema = schema, **conv_kwargs, **bn_kwargs
        )
        conv_kwargs = dict(padding = "same", schema = schema, **conv_kwargs, **bn_kwargs)
        conv_1x1 = ConvBlock(
            filters = filters_1x1, kernel_size = (1, 1), strides = strides, **conv_kwargs
        )
        conv_1x3_3 = ConvBlock(
            filters = filters_3x3, kernel_size = (1, 3), strides = strides, **conv_kwargs
        )
        conv_3x1_3 = ConvBlock(
            filters = filters_3x3, kernel_size = (3, 1), strides = strides, **conv_kwargs
        )
        conv_3x3_5 = ConvBlock(filters = filters_5x5, kernel_size = (3, 3), **conv_kwargs)
        conv_1x3_5 = ConvBlock(
            filters = filters_5x5, kernel_size = (1, 3), strides = strides, **conv_kwargs
        )
        conv_3x1_5 = ConvBlock(
            filters = filters_5x5, kernel_size = (3, 1), strides = strides, **conv_kwargs
        )
        conv_3x3 = Parallel([conv_1x3_3, conv_3x1_3])
        conv_5x5 = Composition([
            conv_3x3_5, 
            Parallel([conv_1x3_5, conv_3x1_5])
        ])
        reduction_3x3 = ConvBlock(filters = inputs_3x3, **pw_kwargs)
        reduction_5x5 = ConvBlock(filters = inputs_5x5, **pw_kwargs)
        reduction_pool = ConvBlock(filters = filters_pool, **pw_kwargs)
        pool = L.MaxPooling2D(pool_size = (3, 3), strides = strides, padding = "same")
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