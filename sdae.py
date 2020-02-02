from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn


def build_units(dimensions: Iterable[int], activation: Optional[torch.nn.Module]) -> List[torch.nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear
    layer followed by an activation layer.
    #Arguments:
    dimensions: iterable of dimensions for the chain
    activation: activation layer to use e.g. nn.ReLU, set to None to disable
    #Return: 
    list of instances of Sequential
    """
    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [('linear', nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(('activation', activation))
        return nn.Sequential(OrderedDict(unit))
    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension
        in sliding_window(2, dimensions)
    ]


def default_initialise_weight_bias_(weight: torch.Tensor, bias: torch.Tensor, gain: float) -> None:
    """
    Default function to initialise the weights in the Linear units of the StackedDenoisingAutoEncoder.
    #Arguments
    weight: weight Tensor of the Linear unit
    bias: bias Tensor of the Linear unit
    gain: gain for use in initialiser
    Return: 
    None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)

class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(self,
                 dimensions: List[int],
                 activation: torch.nn.Module = nn.ReLU(),
                 final_activation: Optional[torch.nn.Module] = nn.ReLU(),
                 weight_init: Callable[[torch.Tensor, torch.Tensor, float], None] = default_initialise_weight_bias_,
                 gain: float = nn.init.calculate_gain('relu')):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimension input is the list of dimension occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].
        #Arguments:
        dimensions: list of dimensions occurring in a single stack
        activation: activation layer to use for all but final activation, default torch.nn.ReLU
        final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        gain: gain parameter to pass to weight_init
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        
        # construct the encoder
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(build_units([self.dimensions[-2], self.dimensions[-1]], None))
        self.encoder = nn.Sequential(*encoder_units)
        
        # construct the decoder
        decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(build_units([self.dimensions[1], self.dimensions[0]], final_activation))
        self.decoder = nn.Sequential(*decoder_units)
        
        # initialise the weights and biases in the layers
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.
        #Arguments:
        index: subautoencoder index
        #Return
        Tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError('Requested subautoencoder cannot be constructed, index out of range.')
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)