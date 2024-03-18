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
    Module to build a sequential neural network using PyTorch
    """

    def __init__(self):
        pass


    def get_new_model(self, device, num_features):
        """
        Returns a new neural network model.

        Args:
            device (str): The target device for the model.
            num_features (int): The number of features of the data.

        Returns:
            torn.nn.Sequential: A neural network model.
        """
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
    A class for training a model from AbbNnSeqModelBuilder.get_new_model

    Properties:
       expected_hyperparam_list (List[str]): list of keys that must be
         present in the hyperparam_dict passed to __init__
       loss_name (str): name of the loss function used to train
       loss_function (function): actual loss function used to train
       step_col (str): name of the step/epoch column in the loss DataFrame
       dset_col (str): name of the data set (train/val) column in the
         loss DataFrame
       summary_pred_col (str): name of the prediction column in the
         prediction summary DataFrame
       summary_targ_col (str): name of the target column in the
         prediction summary DataFrame

    """

    def __init__(self, logger, hyperparam_dict):
        """Initializes the class with the logger and hyperparam_dict.

        Raises ValueError if required hyperparameters are not in the dict

        Args:
            logger (logging.Logger): A logger for recording training events.
            hyperparam_dict (dict): A dictionary of hyperparameters for the model.
        """
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
        """
        Train the model on batches of data. One call to this function is one epoch.

        Args:
            dataloader (torch.data.DataLoader): A data loader for the training data.
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): An optimizer for the model's parameters.
            device (str): The target device for the model.

        Returns:
            avg_loss (float): the average value of the loss function over the batches
        """
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
        

    def validate(self, dataloader, model, device, build_preds_df=False):
        """
        Validates the model on the given data

        Args:
            dataloader (torch.data.DataLoader): A data loader for the validation data.
            model (nn.Module): The model to validate.
            device (str): The target device for the model.

            build_preds_df (bool, Optional): A boolean flag indicating if the results
                should be written to a DataFrame.

        Returns:
            tuple[float, pd.DataFrame]:
               - avg_loss (float): average value of the loss function over the data
               - df (pd.DataFrame): pd.DataFrame storing the target and prediction
                   values for each record. None if build_preds_df==False
        """
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
        """
        Trains and validates the model on the given data

        Args:
            train_dataloader (torch.data.DataLoader): A data loader for the training data.
            val_dataloader (torch.data.DataLoader): A data loader for the validation data.
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): An optimizer for the model's parameters.
            device (str): The target device for the model.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
               - loss_df (pd.DataFrame): DataFrame containing the summary train/val
                   loss information
               - val_preds_df (pd.DataFrame): DataFrame containing the target/prediction
                   values for the validation dataset
        """
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
            device=device,
            build_preds_df=True
            )

        return loss_df, val_preds_df


"""
XGBoost related section
"""


class AbbXgbModelBuilder():
    """
    Module to build an XGBRegressor model

    Properties:
       expected_hyperparam_list (List[str]): list of keys that must be
         present in the hyperparam_dict passed to __init__
    """

    def __init__(self, hyperparam_dict):
        """Initializes the class with the hyperparam_dict.

        Raises ValueError if required hyperparameters are not in the dict

        Args:
            hyperparam_dict (dict): A dictionary of hyperparameters for the model.
        """
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
        """
        Returns a new XGBRegressor model

        Args:
            loss_function (function): actual loss function used to train

        Returns:
            xgboost.XGBRegressor: a new model with hyperparameters set
        """
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
    A class for training a model from AbbXgbModelBuilder.get_new_model

    Properties:
       expected_hyperparam_list (List[str]): list of keys that must be
         present in the hyperparam_dict passed to __init__
       loss_name (str): name of the loss function used to train
       loss_function (function): actual loss function used to train
       step_col (str): name of the step/epoch column in the loss DataFrame
       dset_col (str): name of the data set (train/val) column in the
         loss DataFrame
       summary_pred_col (str): name of the prediction column in the
         prediction summary DataFrame
       summary_targ_col (str): name of the target column in the
         prediction summary DataFrame
    """

    def __init__(self, logger, hyperparam_dict):
        """Initializes the class with the logger and hyperparam_dict.

        Raises ValueError if required hyperparameters are not in the dict

        Args:
            logger (logging.Logger): A logger for recording training events.
            hyperparam_dict (dict): A dictionary of hyperparameters for the model.
        """
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
        """
        Prepares data from a Pandas DataFrame so it can be passed to
        an XGBoost model using the sklearn interface.

        Args:
            df (pandas.DataFrame): A DataFrame containing the data to prepare.
            feature_col_list (list): A list of feature column names.
            target_col (str): The name of the target column.

        Returns:
            tuple[pd.DataFrame, pd.Series]: the X,y values to be used for training
              or validation
        """
        return df.loc[:, feature_col_list], df.loc[:, target_col]


    def get_pred_vs_targ_df(self, val_df,
                            feature_col_list, target_col,
                            model):
        """
        Stores values predicted by the model in a a DataFrame. Data from val_df
        is used for predictions. The target column from val_df is copied to the
        new summary DataFrame

        Args:
            val_df (pandas.DataFrame): A DataFrame containing the data to generate
              the predictions for. Intended to be the validation dataset
            feature_col_list (list): A list of feature column names.
            target_col (str): The name of the target column.
            model (XGBoost.XGBRegressor): A trained XGBoost regressor model object.

        Returns:
            pandas.DataFrame: A DataFrame containing the summary prediction and
              target values
        """
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

        Args:
            eval_results (dict): An evaluation results dictionary created by XGB
            res_name_list (list): A list of meaningful strings representing to
              be used instead of XGB's eval_results default keys. The list order
              must correspond to the order of keys in eval_results

        Returns:
            pd.DataFrame: A DataFrame containing the evaluation results.
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
        """
        Trains a model on the training data and validates its performance
        on the validation data.

        Args:
            train_df (pd.DataFrame): The DataFrame containing the training data.
            val_df (pd.DataFrame): The DataFrame containing the validation data.
            feature_col_list (List[str]): A list of strings representing the
              names of the feature columns
            target_col (str): The name of the target variable column
            model (object): The XGB model to be trained

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
              - loss_df (pd.DataFrame): DataFrame containing the train/val loss
                  values at each training step
              - val_preds_df (pd.DataFrame): DataFrame containing the
                  prediction and target values for the validation dataset
        """

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
