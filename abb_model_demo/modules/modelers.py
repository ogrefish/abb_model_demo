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

import xgboost

import sklearn.metrics as sm


"""
PyTorch Neural Net related section
"""


class AbbNnSeqModelBuilder():
    """
    """

    def __init__(self):
        pass


    def get_new_model(self, device, num_features):
        return nn.Sequential(nn.Linear(num_features, 32),
                             nn.BatchNorm1d(32),
                             nn.ReLU(),
                             nn.Dropout(p=0.05),
                             nn.Linear(32, 16),
                             nn.BatchNorm1d(16),
                             nn.ReLU(),
                             nn.Linear(16, 1),
                             )\
                 .double()\
                 .to(device)


class AbbNnModelTrainer():
    """
    """

    def __init__(self, logger, hyperparam_dict):
        self.logger = logger
        self.hyperparam_dict = hyperparam_dict

        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            raise ValueError(f"missing hyper parameters: {missing_param_list}")


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

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Epoch {epoch}: train loss {train_loss:0.03f} "
                    f"val loss {val_loss:0.03f}"
                    )

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


"""
XGBoost related section
"""


class AbbXgbModelBuilder():
    """
    """

    def __init__(self, hyperparam_dict):
        self.hyperparam_dict = hyperparam_dict

        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            raise ValueError(f"missing hyper parameters: {missing_param_list}")


    @property
    def expected_hyperparam_list(self):
        return [
            "learn_rate",
            "num_estimators",
            "subsample",
            "colsample_bytree",
            "gamma",
            "tree_method",
            "enable_categorical",
            "rseed"
            ]


    def get_new_model(self, loss_function):
        return xgboost.XGBRegressor(
            learning_rate=self.hyperparam_dict["learn_rate"],
            n_estimators=self.hyperparam_dict["num_estimators"],
            subsample=self.hyperparam_dict["subsample"],
            colsample_bytree=self.hyperparam_dict["colsample_bytree"],
            gamma=self.hyperparam_dict["gamma"],
            tree_method=self.hyperparam_dict["tree_method"],
            eval_metric=loss_function,
            enable_categorical=self.hyperparam_dict["enable_categorical"],
            random_state=self.hyperparam_dict["rseed"],
            )


class AbbXgbModelTrainer():
    """
    """

    def __init__(self, logger, hyperparam_dict):
        self.logger = logger
        self.hyperparam_dict = hyperparam_dict

        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            raise ValueError(f"missing hyper parameters: {missing_param_list}")


    @property
    def expected_hyperparam_list(self):
        return [
            "fit_verbose",
            ]


    @property
    def loss_function(self):
        return sm.mean_squared_error


    @property
    def loss_name(self):
        return "mean_squared_error"


    @property
    def step_col(self):
        return "iteration"


    @property
    def dset_col(self):
        return "data_set"


    @property
    def summary_pred_col(self):
        return "preds"


    @property
    def summary_targ_col(self):
        return "targs"


    def _get_xy_dfs(self, df, feature_col_list, target_col):
        return df.loc[:, feature_col_list], df.loc[:, target_col]


    def get_pred_vs_targ_df(self, val_df,
                            feature_col_list, target_col,
                            model):
        df = val_df.loc[:, [target_col]]\
                   .copy()\
                   .rename(columns={target_col: self.summary_targ_col})
        df.loc[:, self.summary_pred_col] \
            = model.predict(val_df.loc[:, feature_col_list])

        return df


    def get_eval_results_df(self, eval_results,
                            res_name_list=["train", "val"]):
        """
        Convert XGB's eval_results dict, which is like
        { "validaion_0": {
             "rmse": [#, #, ...],
             "mean_squared_error": [#, #, ...],
           ...
        }

        to a dataframe for easy SNS plotting, e.g.
                 rmse  mean_squared_error data_set
        0    0.278199            0.077395    train
        1    0.275473            0.075885    train
        2    0.269733            0.072756    train
        3    0.266310            0.070921    train
        4    0.262982            0.069159    train
        ..        ...                 ...      ...
        195  0.214134            0.045853     val
        196  0.214140            0.045856     val
        197  0.214104            0.045840     val
        198  0.214100            0.045839     val
        199  0.214108            0.045842     val
        """
        df_list = []
        for rname, metric_dict in zip(res_name_list, eval_results.values()):
            df = pd.DataFrame()
            for metric_name, value_list in metric_dict.items():
                df.loc[:, metric_name] = value_list
            df.loc[:, self.dset_col] = rname
            df_list.append(df)
        df = pd.concat(df_list)
        df.loc[:, self.step_col] = df.index
        return df


    def train_and_validate(self, train_df, val_df,
                           feature_col_list, target_col,
                           model):

        X_train, y_train = self._get_xy_dfs(df=train_df,
                                            feature_col_list=feature_col_list,
                                            target_col=target_col
                                            )
        X_val, y_val = self._get_xy_dfs(df=val_df,
                                        feature_col_list=feature_col_list,
                                        target_col=target_col
                                        )

        model.fit(X_train,
                  y_train,
                  eval_set=[
                      (X_train, y_train),
                      (X_val, y_val)
                  ],
                  verbose=self.hyperparam_dict["fit_verbose"]
                  )
        eval_results = model.evals_result()

        # df named for consistency with AbbNnModelTrainer
        val_preds_df = self.get_pred_vs_targ_df(
            val_df=val_df,
            feature_col_list=feature_col_list,
            target_col=target_col,
            model=model
            )

        # df named for consistency with AbbNnModelTrainer
        loss_df = self.get_eval_results_df(eval_results=eval_results)

        return loss_df, val_preds_df
