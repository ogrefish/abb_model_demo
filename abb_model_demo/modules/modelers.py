"""
In this land are found modules that define, train and validate
models.
"""

import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from abb_model_demo.modules.data_loaders import CsvFeatureDfBuilder


class AbbNnSeqModelBuilder():
    """
    """

    def __init__(self, device,
                 feature_col_list=CsvFeatureDfBuilder.feature_col_list):
        self.device = device
        self._num_features = len(feature_col_list)


    def get_new_model(self):
        return nn.Sequential(nn.Linear(self._num_features, 32),
                             nn.BatchNorm1d(32),
                             nn.ReLU(),
                             nn.Dropout(p=0.05),
                             nn.Linear(32, 16),
                             nn.BatchNorm1d(16),
                             nn.ReLU(),
                             nn.Linear(16, 1),
                             )\
                 .double()\
                 .to(self.device)


class AbbNnModelTrainer():
    """
    """

    def __init__(self, hyperparam_dict):
        self.hyperparam_dict = hyperparam_dict

        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            emsg = f"missing hyper parameters: {missing_param_list}"
            raise ValueError(f"AbbNnModelTrainer.__init__ {emsg}")


    @property
    def expected_hyperparam_list(self):
        return [
            "num_epochs",
            ]


    @property
    def loss_name(self):
        return "mean_squared_error"


    @property
    def loss_function(self):
        return F.mse_loss


    @property
    def step_col(self):
        return "epoch"


    @property
    def dset_col(self):
        return "data_set"


    @property
    def summary_pred_col(self):
        return "preds"


    @property
    def summary_targ_col(self):
        return "targs"


    def _train_batch(self, dataloader, model, optimizer, device):
        model.train()
        avg_loss = 0.0
        for bi, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.batch_start()
            output = model(data)
            loss = self.loss_function(output, target)
            avg_loss += loss.item()
            loss.backward()
            optimizer.batch_step()
        avg_loss /= len(dataloader)  # div by num batches
        return avg_loss
        

    def validate(self, dataloader, model, optimizer, device,
                 build_preds_df=False):
        model.eval()
        avg_loss = 0.0
        bdf_list= []
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                avg_loss += self.loss_function(output, target, reduction="sum")\
                                .item()
                if build_preds_df:
                    df = pd.DataFrame(data=np.hstack([output.cpu().numpy(),
                                                      target.cpu().numpy()
                                                      ]),
                                      columns=[self.summary_pred_col,
                                               self.summary_targ_col]
                                      )
                    bdf_list.append(df)
        num_pts = len(dataloader.dataset)  # num points (not num batches)
        avg_loss /= num_pts  

        if len(bdf_list)>0:
            df = pd.concat(bdf_list)
            df.loc[:, self.loss_col] = np.divide(df.loc[:, self.loss_col],
                                                 num_pts)
        else:
            df = None
        return avg_loss, df


    def train_and_validate(self, train_dataloader, val_dataloader,
                           model, optimizer, device):
        loss_list = []
        for epoch in range(self.hyperparam_dict["num_epochs"]):
            train_loss = self._train_batch(
                dataloader=train_dataloader,
                model=model,
                optimizer=optimizer,
                device=device
                )
            val_loss, val_df = self.validate(
                dataloader=val_dataloader,
                model=model,
                optimizer=optimizer,
                device=device,
                build_preds_df=False
                )
            optimizer.epoch_step()

            loss_list.append( (epoch, "train", train_loss) )
            loss_list.append( (epoch, "val", val_loss) )

        loss_df = pd.DataFrame.from_records(
            data=loss_list,
            columns=[self.step_col, self.dset_col, self.loss_name]
            )

        _, val_preds_df = self.validate(
            dataloader=val_dataloader,
            model=model,
            optimizer=optimizer,
            device=device,
            build_preds_df=True
            )

        return loss_df, val_preds_df
