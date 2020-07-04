from .helpers import _join_layer


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
        outputs = list()
        for layer in self.layers:
            x = layer(inputs)
            if isinstance(x, list):
                outputs.extend(x)
            else:
                outputs.append(x)
        return outputs
    
    
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
    
    
class DenseSkip:
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
