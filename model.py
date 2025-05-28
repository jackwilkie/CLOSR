""" MLP in pytorch
"""

import torch as T
from torch import Tensor
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from typing import Callable, List, Type, Optional

class Residual(nn.Module):
    '''
     Resiual layer for FF netoworks, resizes original input using linear transformation
    '''
    def __init__(self, 
                 layer: Type[nn.Module],
                 in_dim: int, 
                 out_dim: int = None,
                 layer_scale = None,
                 ) -> None:
        '''
        Parameters
        ----------
        layer : nn.Module
            Layer to implement residual connection for.
        in_dim : int
            Input size to residual layer.
        out_dim : TYPE, optional
            Output size of residual layer. The default is None (uses input size).
        '''
        super().__init__()
        self.layer = layer
        out_dim = out_dim or in_dim  # use input size if output size not provided
        self.layer_scale = nn.Parameter(T.ones(out_dim,) * layer_scale) if layer_scale is not None else 1.0
        
        if in_dim == out_dim:
            self.resize = nn.Identity()  # dont resize residual if input and output are same size
        else:
            self.resize = nn.Linear(in_dim, out_dim)  # otherwise resize with linear layer
    
    def forward(self, x):
        '''
        print('data device:')
        print(x.device)
        print('layer scale device:')
        print(self.layer_scale)
        print(self.layer_scale.device)
        '''
        return (self.layer(x) * self.layer_scale) + self.resize(x)  # residual connection

class Norm(nn.Module):
    def forward(self, x):
        return F.normalise(x, p = 2., dim = -1)
    
class DenseBlock(nn.Module):
    '''
     Dense blocks for feedforward network
    '''
    def __init__(
        self,
        in_dim: int,
        out_dim: int = None,
        dropout: float = 0.,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        activation: Callable[[], nn.Module] = nn.ReLU,
        residual: bool = False,
        layer_scale: Optional[float] = None,
        layernorm = None,
        bias: bool = True,
    ) -> None:
        '''
        Parameters
        ----------
        in_dim : int
            Input dimensions to dense layer.
        out_dim : int, optional
            Number of neurons in dense layer. The default is None (uses input size).
        dropout: float, optional
            Ammount of dropout after activation. The default is 0. (None).
        dropout_layer: Callable[nn.Module], optional
            Layer to be used for dropout.
        activation : Callable[nn.Module], optional
            Constructor function for activation layer. The default is nn.ReLU.
        residual : bool, optional
            Use residual connection if True. The default is False.
        '''
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim  # use input dimenstions if output size not provided
        
        norm_layer = layernorm or nn.Identity
        
        # -- check for gated activation function
        if hasattr(activation, 'gated') and activation.gated:
            gated = True
        else:
            gated = False
        
        # linear layer with activation 
        #print(nn.Sequential(nn.Linear(in_dim, out_dim * 2 if gated else out_dim, bias = bias), activation(), norm_layer(out_dim), dropout_layer(dropout)))
        #raise(ValueError('TESTING'))
        self.block = nn.Sequential(nn.Linear(in_dim, out_dim * 2 if gated else out_dim, bias = bias), activation(), norm_layer(out_dim), dropout_layer(dropout))
        
        if residual:  # add residual connection
            self.block = Residual(self.block, in_dim, out_dim, layer_scale=layer_scale)
        
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.block(x)
    
    
class GatedActivation(nn.Module):
    '''
    Gated activation, nonlinearity is applied to half of input which is used as weights for other half which is then returned
    '''
    gated = True
    
    def __init__(self,
                activation: Type[nn.Module],
                in_size: int
                ) -> None:
        '''
        Parameters
        ----------
        activation : nn.Module
            Constructor for activation function.
        in_size : int
            Input size to activation function.
        '''
        
        super().__init__()
        self.activation = activation()  # non linearity

    def forward(self, x: T.Tensor) -> T.Tensor:
        x, gates = x.chunk(2, dim = -1)  # split input into weights and output
        return x * self.activation(gates)  # apply nonlinearity to weigths, weights output and return
        
    
def make_mlp(
        d_in: int,
        neurons: List[int],
        activation: Callable[[],nn.Module] = nn.ReLU,
        dropout: float = 0.,
        residual: List[bool] = [],
        gated: bool = False,
        norm_layer: Callable[[int],nn.Module] = nn.Identity,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        d_out = None,
        final_layer_activation = None,
        layer_scale = None,
        block_norm = False,
        bias: bool = True,
    ) -> nn.Sequential:
    '''
    Constructor function to make mlp

    Parameters
    ----------
    d_in : int
        Number of input features.
    neurons : list
        Number of neurons in hidden layer, len(neurons) is depth of network.
    activation : Callable[nn.Module], optional
        Activation function. The default is nn.ReLU.
    dropout : float, optional
        Probability of dropping out connections. The default is 0..
    residual : list[bool], optional
        Wheter or not to use residual connections in each layer. The default is [].
    gated : bool, optional
        Whether or not to use gated activation functions. The default is False.
    norm_layer : Callable[nn.Module], optional
        Layer to normalise output. The default is nn.Identitiy.
    dropout_layer : Callable[nn.Module], optional
        Dropout layer. The default is nn.Dropout.

    Returns
    -------
    nn.Sequential
        MLP model.
    '''

    if not isinstance(neurons, list):
        neurons = [neurons]

    neurons = [d_in] + neurons  # prepend input size to list of neurons
    if d_out is not None:
        neurons = neurons + [d_out]
    layers = []  # init layers
    
    if isinstance(residual, bool):
        residual = [residual for _ in neurons]
    
    if len(residual) == 0:
        residual = [False for _ in neurons]
        
    elif len(residual) != len(neurons):
        raise ValueError('Length of residual argument must match length of neurons argument got, residual len: {len(residula)}, neuron len: {len(neurons)}')
    
    
    # iterate over layers
    for i in range(len(neurons) - 1):
        # add layer with actiation function and dropout
        if i != len(neurons) - 2:
            # gate activation if required
            if gated:
                layer_activation = partial(GatedActivation, in_size = neurons[i + 1], activation = activation, gated = gated)
            else:
                layer_activation = activation
                
            # add dense block
            layers += [DenseBlock(neurons[i], neurons[i + 1], activation = layer_activation, residual = residual[i], dropout = dropout, dropout_layer = dropout_layer, layer_scale=layer_scale, layernorm = norm_layer if block_norm else None, bias = bias)]
        
        else:  # final layer has no activation or dropout
            
            layers += [DenseBlock(neurons[i], neurons[i + 1], activation = nn.Identity if final_layer_activation is None else final_layer_activation, residual = residual[i], dropout_layer=nn.Identity, layer_scale=layer_scale,layernorm = norm_layer if block_norm else None, bias = bias)]
    
    # use layernorm on output if required
    if not block_norm:
        layers += [norm_layer(neurons[-1])]

    return nn.Sequential(*layers)  # create sequential network from layers
    

class MLPConfig:
    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        residual: List[bool] = [],
        gated: bool = False,
        norm_layer: Callable[[int],nn.Module] = nn.Identity,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        d_out = None, 
        layer_scale = None,
    ):
        self.d_in = d_in
        self.neuron = neurons
        self.activation = activation
        self.dropout = dropout
        self.residual = residual
        self.gated = gated
        self.norm_layer = norm_layer
        self.dropout_layer = dropout_layer
        self.d_out = d_out
        self.layer_scale = layer_scale


class HyperSphere(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        norm = x.norm(p=2, dim=-1, keepdim=True)  # Compute the L2 norm
        mask = norm > 1  # Find vectors with a norm greater than 1
        x[mask] = x[mask] / norm[mask]  # Scale down only those vectors
        return x
        
class ContrastiveMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        residual: List[bool] = [],
        gated: bool = False,
        norm_layer: Callable[[int],nn.Module] = nn.Identity,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        n_classes = None, 
        d_out: Optional[int] = None,
        final_layer_activation = None,
        project_to_sphere = False,
        layer_scale = None,
        block_norm = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        norm_layer = Identity if norm_layer is None else norm_layer
        neurons = neurons if isinstance(neurons,list) else [neurons]
        
        self.mlp = make_mlp(
            d_in = d_in,
            neurons = neurons,
            activation = activation,
            dropout = dropout,
            residual=residual,
            gated = gated,
            norm_layer=norm_layer,
            dropout_layer=dropout_layer,
            final_layer_activation=final_layer_activation,
            layer_scale= layer_scale,
            block_norm = block_norm,
            bias = bias,
        )
        
        self.proj = nn.Identity() if d_out is None else nn.Linear(neurons[-1], d_out)
        
        if project_to_sphere:
            self.proj = nn.Sequential(
                self.proj,
                HyperSphere(),
            )
        
        self.probe = nn.Identity() if n_classes is None else nn.Linear(neurons[-1], n_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def forward_cls(self, x: Tensor) -> Tensor:
        return self.probe(x)
    
    def forward_finetune(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_cls(x)
        return x

    def forward_probe(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x =  self.forward_features(x)
        return self.proj(x)