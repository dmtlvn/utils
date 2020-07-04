from tensorflow.keras import layers as K
from . import layers as L
import inspect
import copy


class LazyLayer:
    
    _layer_ = None
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    @property
    def layer(self):
        return self._layer_(*self.args, **self.kwargs)
    
    def build(self, *extra_args, **extra_kwargs):
        args = extra_args + self.args
        kwargs = {**self.kwargs, **extra_kwargs}
        return self._layer_(*args, **kwargs)
        
    def __call__(self, inputs, *extra_args, **extra_kwargs):
        return self.build(*extra_args, **extra_kwargs)(inputs)
    
    
class Meta(type):
    
    @classmethod
    def from_class(cls, other):
        
        def copy_signature(from_func, to_func):
            
            def to_func(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
            
            ret_func = copy.deepcopy(to_func)
            ret_func.__signature__ = inspect.signature(from_func)
            return ret_func
            
        name = other.__name__
        base = copy.deepcopy(LazyLayer)
        bases = (base,)
        dct = {
            "_layer_": other,
            "__init__": copy_signature(other.__init__, base.__init__),
        }
        return cls(name, bases, dct)


_layer_list = [
    L.Identity,
    L.DownSampling2D,
    L.Resize,
    L.DropAvg,
    L.GroupShuffle,
    L.GroupConv2D,
    L.GlobalAvgPool2D,
    L.GlobalMaxPool2D,
    
    K.Layer,
    K.Input,
    K.InputLayer,
    K.ELU,
    K.LeakyReLU,
    K.PReLU,
    K.ReLU,
    K.Softmax,
    K.ThresholdedReLU,
    K.Conv1D,
    K.Convolution1D,
    K.Conv2D,
    K.Convolution2D,
    K.Conv2DTranspose,
    K.Convolution2DTranspose,
    K.Conv3D,
    K.Convolution3D,
    K.Conv3DTranspose,
    K.Convolution3DTranspose,
    K.Cropping1D,
    K.Cropping2D,
    K.Cropping3D,
    K.DepthwiseConv2D,
    K.SeparableConv1D,
    K.SeparableConvolution1D,
    K.SeparableConv2D,
    K.SeparableConvolution2D,
    K.UpSampling1D,
    K.UpSampling2D,
    K.UpSampling3D,
    K.ZeroPadding1D,
    K.ZeroPadding2D,
    K.ZeroPadding3D,
    K.ConvLSTM2D,
    K.Activation,
    K.ActivityRegularization,
    K.Dense,
    K.Dropout,
    K.Flatten,
    K.Lambda,
    K.Masking,
    K.Permute,
    K.RepeatVector,
    K.Reshape,
    K.SpatialDropout1D,
    K.SpatialDropout2D,
    K.SpatialDropout3D,
    K.AdditiveAttention,
    K.Attention,
    K.Embedding,
    K.LocallyConnected1D,
    K.LocallyConnected2D,
    K.Add,
    K.Average,
    K.Concatenate,
    K.Dot,
    K.Maximum,
    K.Minimum,
    K.Multiply,
    K.Subtract,
    K.AlphaDropout,
    K.GaussianDropout,
    K.GaussianNoise,
    K.LayerNormalization,
    K.BatchNormalization,
    K.AveragePooling1D,
    K.AvgPool1D,
    K.AveragePooling2D,
    K.AvgPool2D,
    K.AveragePooling3D,
    K.AvgPool3D,
    K.GlobalAveragePooling1D,
    K.GlobalAvgPool1D,
#     K.GlobalAveragePooling2D,
#     K.GlobalAvgPool2D,
    K.GlobalAveragePooling3D,
    K.GlobalAvgPool3D,
    K.GlobalMaxPooling1D,
    K.GlobalMaxPool1D,
#     K.GlobalMaxPooling2D,
#     K.GlobalMaxPool2D,
    K.GlobalMaxPooling3D,
    K.GlobalMaxPool3D,
    K.MaxPooling1D,
    K.MaxPool1D,
    K.MaxPooling2D,
    K.MaxPool2D,
    K.MaxPooling3D,
    K.MaxPool3D,
    K.AbstractRNNCell,
    K.RNN,
    K.SimpleRNN,
    K.SimpleRNNCell,
    K.StackedRNNCells,
    K.GRU,
    K.GRUCell,
    K.LSTM,
    K.LSTMCell,
]

_globals = globals()
for layer in _layer_list:
    _globals[layer.__name__] = Meta.from_class(layer)
    

