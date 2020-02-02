import torch 
import torch.nn as nn
from lib.cluster import ClusterAssignment


class DEC(nn.Module):
    def __init__(self, cluster_number: int, hidden_dimension: int, encoder: torch.nn.Module, alpha: float = 1.0):
        """
        Module which holds all the moving parts of the DEC algorithm, includes the AE stage and the ClusterAssignment stage.
        #Arguments:
            cluster_number: number of clusters
            hidden_dimension: output of the encoder
            embedding_dimension: input of the ecncode
            encoder: encoder to be used
            alpha: the degrees of freedom in the t-distribution, default = 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension, alpha)
    
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch through the encoding part of the associated AE module.
        #Arguments:
          batch: [batch size, embedding dimension] FloatTensor
        #Return
          [batch size, number of cluster] FloatTesnor
        """
        return self.assignment(self.encoder(batch))

          