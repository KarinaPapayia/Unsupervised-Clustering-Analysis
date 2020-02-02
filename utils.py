import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from typing import Optional


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to determine the reassignments. 
    Require scikit-learn installed. Standard unsupervised evaluation metric.
    # Arguments
        y_true: list of true cluster number, an integer array 0- indexed.       
        y_predicted: list of predicted cluster numbersm an integer array 0-indexed.
        cluster_number: number of clusters, if None then calculated from input.
    # Return
        reassignment dictionary, clustering accuracy in [0, 1].
    """
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1 #assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype = np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum()/y_predicted.size
    return reassignment, accuracy
    

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch distribution q_ij
    # Arguments
        batch: [batch_size, number of cluster], tensor of dtype float
    #Retutn
        [batch_size, number of clusters], tensor of dtype float
    """
    
    weight = (batch **2) / torch.sum(batch, 0)
    return (weight.t() / torch. sum(weight, 1)).t()