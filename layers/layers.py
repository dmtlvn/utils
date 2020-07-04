from functools import partial

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.keras.utils import conv_utils



class Identity(K.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def call(self, inputs):
        return inputs
    
    
class DownSampling2D(K.layers.Layer):
    """
    Downsamples an input tensor by an integer factor along spatial axes. Spatial
    axes are assumed to be (1, 2).
    
    Parameters:
    
    size (int, tuple) - downsampling factor
    data_format (str) - "NHWC" or "NCHW" string for channel-first and 
        channels-last input tensor layout respectively
    """
    
    def __init__(self, size, data_format = "NHWC", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data_format not in {"NCHW", "NHWC"}:
            raise ValueError(f"Invalid data format {data_format}")
        self.data_format = data_format
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        if self.size[0] < 1 or self.size[1] < 1:
            raise ValueError("Downsampling factor is < 1")

    def call(self, inputs):
        if self.size[0] == self.size[1] == 1:
            return inputs
        if self.data_format == "NHWC":
            _, H, W, _ = inputs.shape
        else:
            _, _, H, W = inputs.shape
        if (H % self.size[0] != 0) or (W % self.size[1] != 0):
            warnings.warn(f"Tensor {inputs.name} dimensions are not multiples of the "
                          "downscale factors")
        if self.data_format == "NHWC":
            return inputs[:, ::self.size[0], ::self.size[1], :]
        else:
            return inputs[:, :, ::self.size[0], ::self.size[1]]
        
        
class Resize(K.layers.Layer):
    """
    Resizes an input tensor to a given shape
    
    Parameters:
    
    target_shape (tuple) - target shape in the form (height, width)
    
    interpolation (str) - interpolation algorithm ('nearest', 'bilinear', 
        'bicubic', 'lanczos3', 'lanczos5', 'gaussian', 'area', 'mitchellcubic')
        Refer to tensorflow.image.resize documentation for more info. 
        Default: 'nearest'
        
    antialias (bool) - if true, applies antialiasing pre-filtering
    
    pad (bool) - if true, keeps the aspect ratio if the input images by padding 
        it with zeros
        
    channels_first (bool) - if true, uses NCHW tensor layout instead of NHWC
    """
    
    def __init__(
        self, 
        size, 
        interpolation = 'nearest',
        antialias = False,
        pad = False,
        channels_first = False, 
        *args, 
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.size = size
        self.channels_first = channels_first
        if not pad:
            self._resize_op = partial(tf.image.resize,
                size = size,
                method = interpolation,
                antialias = antialias,
                preserve_aspect_ratio = False,
            )
        else:
            target_height, target_width = size
            self._resize_op = partial(tf.image.resize_with_pad,
                target_height = target_height, 
                target_width = target_width, 
                method = interpolation,
                antialias = antialias,               
            )

        
    def call(self, inputs):
        images = tf.transpose(inputs, [0,2,3,1]) if self.channels_first else inputs
        outputs = self._resize_op(images)
        outputs = tf.transpose(outputs, [0,3,1,2]) if self.channels_first else outputs
        return outputs
    
    
class DropAvg(K.layers.Layer):
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
    
    def __init__(self, rate = 0.5, is_global = False, channels_first = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channel_axis = -1 if not channels_first else 1
        self.is_global = is_global
        self.rate = tf.constant(rate, dtype = tf.float32)
        
    def local_strategy(self, inputs):
        n = len(inputs)
        stack = tf.stack(inputs, axis = self.channel_axis)
        probs = tf.random.uniform((n,))
        probs = probs / tf.reduce_max(probs)
        idx = tf.cast(tf.where(probs > self.rate), tf.int32)
        keep = tf.gather(stack, idx, axis = self.channel_axis)
        if self.channel_axis == -1:
            merge = tf.reduce_mean(keep, axis = (-2, -1))
        else:
            merge = tf.reduce_mean(keep, axis = (1, 2))
        return merge
    
    def global_strategy(self, inputs):
        n = len(inputs)
        stack = tf.stack(inputs, axis = self.channel_axis)
        idx = tf.random.uniform(shape = (1,), minval = 0, maxval = n, dtype = tf.dtypes.int32)
        if self.channel_axis == -1:
            keep = stack[:,:,:,idx[0]]
        else:
            keep = stack[:,idx[0],:,:]
        return keep
        
    def call(self, inputs):
        if isinstance(inputs, tf.Tensor):
            return inputs
        if self.is_global:
            return self.global_strategy(inputs)
        else:
            return self.local_strategy(inputs)
        
        
class GroupShuffle(K.layers.Layer):
    """
    Channel shuffling operation for a grouped convolution. Given the number of
    groups, permutes the channels such that channels from each group are places
    into each group. Example:
    
    [ 1 2 3 | 4 5 6 | 7 8 9 ] --> [ 1 4 7 | 2 5 8 | 3 6 9 ]
    
    For more information refer to a ShuffleNet paper: 
    https://arxiv.org/pdf/1707.01083.pdf
    
    Parameters:
    
    groups (int) - number of groups
    
    data_format (str) - a standard Keras data format string. 
        Default: "channels_last"
    """
    
    def __init__(self, groups, data_format = "channels_last", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Invalid data format `{data_format}`")
        self.groups = groups
        self.data_format = data_format
        
    def call(self, inputs):
        if self.data_format == "channels_last":
            _, H, W, C = inputs.shape
            if C % self.groups:
                raise ValueError("Number of input channels must be a multiple of `groups`")
            g = tf.reshape(inputs, (-1, H, W, C // self.groups, self.groups))
            g = tf.transpose(g, [0, 1, 2, 4, 3])
            outputs = tf.reshape(g, (-1, H, W, C))
        else:
            _, C, H, W = inputs.shape
            if C % self.groups:
                raise ValueError("Number of input channels must be a multiple of `groups`")
            g = tf.reshape(inputs, (-1, self.groups, C // self.groups, H, W))
            g = tf.transpose(g, [0, 2, 1, 3, 4])
            outputs = tf.reshape(g, (-1, H, W, C))
        return outputs  
    
    
class GroupConv2D(K.layers.Layer):
    """
    Grouped convolutions. Splits an input channels into `groups` parts and 
    performs usual 2D convolutions with each part using independent kernels.
    Each convolution has `filters // groups` output channels.
    
    Parameters:
    
    groups (int) - number of groups
    **kwargs - standard parameters for the Conv2D layer
    """
    
    @staticmethod
    def convert_data_format(df_str):
        if df_str is None:
            return "NHWC", -1
        s = df_str.lower()
        if s == "channels_first":
            return "NCHW", 1
        elif s == "channels_last":
            return "NHWC", -1
        else:
            raise ValueError(f"Invalid data format `{df_str}`")       
    
    def __init__(
        self, 
        groups,
        filters,
        kernel_size,
        strides = 1,
        padding = "same",
        data_format = None,
        dilation_rate = 1,
        activation = "linear",
        use_bias = True,
        kernel_initializer = "glorot_uniform",
        bias_initializer = "zeros",
        kernel_regularizer = None,
        bias_regularizer = None,
        activity_regularizer = None,
        kernel_constraint = None,
        bias_constraint = None,
        trainable = True,
        name = None,
        **kwargs
    ):
        act_reg = K.regularizers.get(activity_regularizer)
        super().__init__(
            trainable = trainable, name = name, activity_regularizer = act_reg, **kwargs
        )
        self.groups = int(groups)
        self.filters = int(filters)
        if self.filters % self.groups:
            raise ValueError("Output filters must a multiple of `groups`")
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.padding = padding.upper()
        self.data_format, self.channel_axis = self.convert_data_format(data_format)
        self.activation = K.activations.get(activation)
        self.use_bias = use_bias
        self.weights_config = dict(
            name = "kernel",
            initializer = K.initializers.get(kernel_initializer),
            regularizer = K.regularizers.get(kernel_regularizer),
            constraint = K.constraints.get(kernel_constraint),
            trainable = True,
            dtype = self.dtype,
        )
        self.bias_config = dict(
            name = "bias",
            shape = (self.filters,),
            initializer = K.initializers.get(bias_initializer),
            regularizer = K.regularizers.get(bias_regularizer),
            constraint = K.constraints.get(bias_constraint),
            trainable = True,
            dtype = self.dtype,
        )
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError("Invalid input shape")
        channels = input_shape[self.channel_axis]
        group_shape = list(input_shape)
        if channels % self.groups:
            raise ValueError("Input channels must be a multiple of `groups`")

        kernel_shape = self.kernel_size + (channels // self.groups, self.filters // self.groups)
        self.kernels = [self.add_weight(shape = kernel_shape, **self.weights_config) 
                        for _ in range(self.groups)]
        self.bias = self.add_weight(**self.bias_config) if self.use_bias else None
        if not isinstance(self.padding, (list, tuple)):
            op_padding = self.padding.upper()
        self._conv_op = partial(
            tf.nn.conv2d, 
            strides = self.strides, 
            padding = self.padding, 
            data_format = self.data_format, 
            dilations = self.dilation_rate, 
        )
        self._bias_add_op = partial(
            tf.nn.bias_add,
            bias = self.bias,
            data_format = self.data_format,
        )
        self.built = True

    def call(self, inputs):
        if self.groups == 1:
            outputs = self._conv_op(inputs, self.kernels[0])
        else:
            input_groups = tf.split(inputs, self.groups, axis = self.channel_axis)
            output_groups = [self._conv_op(g, k) for g, k in zip(input_groups, self.kernels)]
            outputs = tf.concat(output_groups, axis = self.channel_axis)
        outputs = self._bias_add_op(outputs) if self.use_bias else outputs
        outputs = self.activation(outputs) if self.activation is not None else outputs
        return outputs
    
    
class GlobalPooling2D(K.layers.Layer):
    """Abstract class for different global pooling 2D layers.
    """

    def __init__(self, data_format = "channels_last", **kwargs):
        if data_format not in {"channels_first", "channels_last"}:
            raise ValueError(f"Invalid data format {data_format}")
        super().__init__(**kwargs)
        self.data_format = data_format
        self._supports_ragged_inputs = True

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        if self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, input_shape[3])
        else:
            return (input_shape[0], input_shape[1], 1, 1)

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
class GlobalAvgPool2D(GlobalPooling2D):
    """
    Global average pooling operation, but unlike Keras, returns a 4D tensor of
    shape [batch_size, 1, 1, channels]. Replicates the original layer in every
    other way
    """
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return tf.reduce_mean(inputs, axis = (1, 2), keepdims = True)
        else:
            return tf.reduce_mean(inputs, axis = (2, 3), keepdims = True)
        
        
class GlobalMaxPool2D(GlobalPooling2D):
    """
    Global max-ooling operation, but unlike Keras, returns a 4D tensor of
    shape [batch_size, 1, 1, channels]. Replicates the original layer in every
    other way
    """
    def call(self, inputs):
        if self.data_format == 'channels_last':
            return tf.reduce_max(inputs, axis = (1, 2), keepdims = True)
        else:
            return tf.reduce_max(inputs, axis = (2, 3), keepdims = True)