import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Optional


class DenoisingAutoencoder(nn.Module):
    def __init__(self,
                 embedding_dimension: int,
                 hidden_dimension: int,
                 activation: Optional[torch.nn.Module] = nn.ReLU(),
                 gain: float = nn.init.calculate_gain('relu'),
                 corruption: Optional[torch.nn.Module] = None,
                 tied: bool = False) -> None:
        """
        Autoencoder composed of two Linear units with optional encoder activation and corruption.
        #Arguments:
        embedding_dimension: embedding dimension, input to the encoder
        hidden_dimension: hidden dimension, output of the encoder
        activation: optional activation unit, defaults to nn.ReLU()
        gain: gain for use in weight initialisation
        corruption: optional unit to apply to corrupt input during training, defaults to None
        tied: whether the autoencoder weights are tied, default to False
        """
        super(DenoisingAutoencoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.activation = activation
        self.gain = gain
        self.corruption = corruption
        
        # encoder parameters
        self.encoder_weight = Parameter(
            torch.Tensor(hidden_dimension, embedding_dimension)
        )
        self.encoder_bias = Parameter(
            torch.Tensor(hidden_dimension)
        )
        self._initialise_weight_bias(self.encoder_weight, self.encoder_bias, self.gain)
       
        # decoder parameters
        self.decoder_weight = Parameter(
            torch.Tensor(embedding_dimension, hidden_dimension)
        ) if not tied else self.encoder_weight.t()
        self.decoder_bias = Parameter(
            torch.Tensor(embedding_dimension)
        )
        self._initialise_weight_bias(self.decoder_weight, self.decoder_bias, self.gain)

    @staticmethod
    def _initialise_weight_bias(weight: torch.Tensor, bias: torch.Tensor, gain: float):
        """
        Initialise the weights in a the Linear layers of the DenoisingAutoencoder.
        Arguments:
        weight: weight Tensor of the Linear layer
        bias: bias Tensor of the Linear layer
        gain: gain for use in initialiser
        Return:
        None
        """
        nn.init.xavier_uniform_(weight, gain)
        nn.init.constant_(bias, 0)

    def copy_weights(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
        """
        Utility method to copy the weights of self into the given encoder and decoder, where
        encoder and decoder should be instances of torch.nn.Linear.
        #Arguments:
        encoder: encoder Linear unit
        decoder: decoder Linear unit
        #Return:
        None
        """
        encoder.weight.data.copy_(self.encoder_weight)
        encoder.bias.data.copy_(self.encoder_bias)
        decoder.weight.data.copy_(self.decoder_weight)
        decoder.bias.data.copy_(self.decoder_bias)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        transformed = F.linear(batch, self.encoder_weight, self.encoder_bias)
        if self.activation is not None:
            transformed = self.activation(transformed)
        if self.corruption is not None:
            transformed = self.corruption(transformed)
        return transformed

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        return F.linear(batch, self.decoder_weight, self.decoder_bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))