import numpy as np
import torch
import torch.nn as nn
from typing import Union
from model.CSTGAN.modules import ST_UNet_RITS

class Generator(nn.Module):
    def __init__(
        self, 
        seq_len: int, 
        img_size: int, 
        rnn_hidden_size: int, 
        device: Union[str, torch.device],
        consistency_loss_weight: float = 0.1,
        reconstruction_loss_weight: float = 1.0,
        ):
        super(Generator, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.img_size = img_size
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size
        #
        self.consistency_loss_weight = consistency_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        # create models
        self.rits_f = ST_UNet_RITS(seq_len, img_size, rnn_hidden_size, device)
        self.rits_b = ST_UNet_RITS(seq_len, img_size, rnn_hidden_size, device)

    def _get_consistency_loss(self, 
        pred_f: torch.Tensor, pred_b: torch.Tensor
        )-> torch.Tensor:
        """Calculate the consistency loss between the imputation from two RITS models.

        Parameters
        ----------
        pred_f :
            The imputation from the forward RITS.

        pred_b :
            The imputation from the backward RITS (already gets reverted).

        Returns
        -------
        float tensor,
            The consistency loss.

        """
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def _reverse(self, ret: dict)-> dict:
        """Reverse the array values on the time dimension in the given dictionary.

        Parameters
        ----------
        ret :

        Returns
        -------
        dict,
            A dictionary contains values reversed on the time dimension from the given dict.

        """
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(
                indices, dtype=torch.long, device=tensor_.device, requires_grad=False
            )
            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret
    
    def forward(self, inputs: dict, task: str = "train") -> dict:
        """Forward processing of BRITS.

        Parameters
        ----------
        inputs :
            The input data.

        Returns
        -------
        dict, A dictionary includes all results.
        """
        # Results from the forward RITS.
        ret_f = self.rits_f(inputs, "forward", task)
        # Results from the backward RITS.
        ret_b = self._reverse(self.rits_b(inputs, "backward", task))

        imputed_data = (ret_f["imputed_data"] + ret_b["imputed_data"]) / 2

        if task.lower() == "test":
            # if not in training mode, return the classification result only
            return {
                "imputed_data": imputed_data,
            }

        consistency_loss = self._get_consistency_loss(
            ret_f["imputed_data"], ret_b["imputed_data"]
        )
        
        reconstruction_loss = ret_f["loss"] + ret_b["loss"]
        reconstruction_MAE = (
            ret_f["reconstruction_MAE"] + ret_b["reconstruction_MAE"]
        ) / 2
        
        loss = consistency_loss*self.consistency_loss_weight + reconstruction_loss*self.reconstruction_loss_weight
        ret_f["imputed_data"] = imputed_data
        ret_f["consistency_loss"] = consistency_loss
        ret_f["reconstruction_loss"] = reconstruction_loss
        ret_f["reconstruction_MAE"] = reconstruction_MAE
        ret_f["loss"] = loss

        return ret_f