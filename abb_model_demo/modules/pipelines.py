"""
In this land are found modules that provide the pipelines.

Pipelines have modules as components, and understand how to
use those components in a step-by-step way to carry out
the full training process.
"""

from os import device_encoding
import torch
from torch.utils.data import DataLoader

from abb_model_demo.modules.data_loaders import DfDataSet


"""
PyTorch Neural Net related section
"""


class AbbNnSeqPipeline():
    """
    A pipeline for a PyTorch neural network model.

    Steps:
      - find the best device (prefer a gpu)
      - load & clean the data
      - random train/val split
      - setup model and optimizer
      - train the model
      - get training and validation summary info
      - plot loss curve
      - plot preds vs target for validation data

    Some hyperparameters are specific to the NN Pipeline (this object),
    while others are already set in the components passed into this
    pipeline. See esp. the model_builder, optimizer and trainer objects.
    The values will be set via the steering script.

    Properties:
       expected_hyperparam_list (List[str]): list of keys that must be
         present in the hyperparam_dict passed to __init__
    """

    def __init__(self, df_builder, model_builder, optimizer,
                 trainer, visualizer, hyperparam_dict):
        """
        Initialize the model with the provided components.

        Args:
            df_builder (CsvFeatureDfBuilder): A class that loads and cleans data
              and builds a pd.DataFrame
            model_builder (AbbNnSeqModelBuilder): A class that builds neural network models.
            optimizer (AbbNnOptimizerBase subclass): An optimizer for the model.
            trainer (AbbNnModelTrainer): A class that trains the model.
            visualizer (AbbPredsVisualizer): A class for visualizing the training process.
            hyperparam_dict (dict): A dictionary containing hyperparameters for the model.

        Raises:
            ValueError: If required hyper parameters are not included in the `hyperparam_dict`.
        """
        self.df_builder = df_builder
        self.model_builder = model_builder
        self.optimizer = optimizer
        self.trainer = trainer
        self.visualizer = visualizer
        self.hyperparam_dict = hyperparam_dict

        missing_param_list = [ k for k in self.expected_hyperparam_list
                               if k not in self.hyperparam_dict.keys()
                               ]
        if len(missing_param_list)>0:
            raise ValueError(f"missing hyper parameters: {missing_param_list}")


    @property
    def expected_hyperparam_list(self):
        return [
            "batch_size",
            "shuffle",
            "rseed",
            ]


    def _get_data_loader(self, data_df):
        """
        Get the data loader for the specified dataset.

        Args:
            data_df (pd.DataFrame): The DataFrame containing the data.

        Returns:
            torch.data.DataLoader: The data loader for the dataset.
        """
        dataset = DfDataSet(data_df=data_df,
                            feature_col_list=self.df_builder.feature_col_list,
                            target_col=self.df_builder.target_col
                            )
        dataloader = DataLoader(dataset,
                                batch_size=self.hyperparam_dict["batch_size"],
                                shuffle=self.hyperparam_dict["shuffle"]
                                )
        return dataloader


    def _find_device(self):
        """
        Find the available device to use for training.

        Order of preference:
           1) "cuda" (linux GPU)
           2) "mps" (macos GPU)
           3) "cpu"

        Returns:
            str: The available device.
        """
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return device


    def run_pipeline(self):
        """
        Runs the pipeline. Steps:
          - find the best device (prefer a gpu)
          - load & clean the data
          - random train/val split
          - setup model and optimizer
          - train the model
          - get training and validation summary info
          - plot loss curve
          - plot preds vs target for validation data

        Returns:
            dict: A dictionary containing the trained model parameters
              containing keys:
              - train_df (pd.DataFrame): training dataset
              - val_df (pd.DataFrame): validation dataset
              - model (torch.nn.Module): trained model object
              - loss_df (pd.DataFrame): loss at each epoc for train & val data
              - val_preds_df (pd.DataFrame): prediction vs target for val data
              - loss_curve_plot (tuple[Figure, Axes]): loss curve plot
              - pred_vs_targ_plot (tuple[Figure, Axes]): prediction vs target
                plot of validation data
        """

        # fix random seed
        torch.manual_seed(self.hyperparam_dict["rseed"])

        # find the device
        device = self._find_device()

        # setup train & validation data
        train_df, val_df = self.df_builder.get_train_val_dfs()
        train_dataloader = self._get_data_loader(data_df=train_df)
        val_dataloader = self._get_data_loader(data_df=val_df)

        # get the model object
        num_features = len(self.df_builder.feature_col_list)
        model = self.model_builder.get_new_model(device=device,
                                                 num_features=num_features)

        # build the optimizer
        self.optimizer.prepare_optimizer(model_parameters=model.parameters())

        # train & validate
        loss_df, val_preds_df = self.trainer.train_and_validate(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            model=model,
            optimizer=self.optimizer,
            device=device
            )

        # plots !
        loss_curve_fig, loss_curve_ax \
            = self.visualizer.draw_loss_curve(
                loss_df=loss_df,
                step_col=self.trainer.step_col,
                loss_col=self.trainer.loss_name,
                dset_col=self.trainer.dset_col
                )
        pred_vs_targ_fig, pred_vs_targ_ax \
            = self.visualizer.draw_pred_vs_targ(
                val_preds_df=val_preds_df,
                pred_col=self.trainer.summary_pred_col,
                targ_col=self.trainer.summary_targ_col
                )

        result_dict = {
            "train_df": train_df,
            "val_df": val_df,
            "model": model,
            "loss_df": loss_df,
            "val_preds_df": val_preds_df,
            "loss_curve_plot": (loss_curve_fig, loss_curve_ax),
            "pred_vs_targ_plot": (pred_vs_targ_fig, pred_vs_targ_ax),
            }
        return result_dict


"""
XGBoost related section
"""


class AbbXgbPipeline():
    """
    A pipeline for a PyTorch neural network model.

    Steps:
      - load & clean the data
      - random train/val split
      - setup model
      - train the model
      - get training and validation summary info
      - plot loss curve
      - plot preds vs target for validation data

    Hyperparameters are already setup in the components passed into the
    pipeline (i.e. the model_builder and trainer objects).
    The values will be set via the steering script.
    """

    def __init__(self, df_builder, model_builder, trainer, visualizer):
        """
        Initialize the model with the provided components.

        Args:
            df_builder (CsvFeatureDfBuilder): A class that loads and cleans data
              and builds a pd.DataFrame
            model_builder (AbbNnSeqModelBuilder): A class that builds neural network models.
            trainer (AbbNnModelTrainer): A class that trains the model.
            visualizer (AbbPredsVisualizer): A class for visualizing the training process.
        """
        self.df_builder = df_builder
        self.model_builder = model_builder
        self.trainer = trainer
        self.visualizer = visualizer


    def run_pipeline(self):
        """
        Runs the pipeline. Steps:
          - load & clean the data
          - random train/val split
          - setup model
          - train the model
          - get training and validation summary info
          - plot loss curve
          - plot preds vs target for validation data

        Returns:
            dict: A dictionary containing the trained model parameters
              containing keys:
              - train_df (pd.DataFrame): training dataset
              - val_df (pd.DataFrame): validation dataset
              - model (torch.nn.Module): trained model object
              - loss_df (pd.DataFrame): loss at each epoc for train & val data
              - val_preds_df (pd.DataFrame): prediction vs target for val data
              - loss_curve_plot (tuple[Figure, Axes]): loss curve plot
              - pred_vs_targ_plot (tuple[Figure, Axes]): prediction vs target
                plot of validation data
        """

        # setup train & validation data
        train_df, val_df = self.df_builder.get_train_val_dfs()

        # get the model object
        model = self.model_builder.get_new_model(
            loss_function=self.trainer.loss_function
            )

        # train & validate
        loss_df, val_preds_df = self.trainer.train_and_validate(
            train_df=train_df,
            val_df=val_df,
            feature_col_list=self.df_builder.feature_col_list,
            target_col=self.df_builder.target_col,
            model=model
            )

        # plots !
        loss_curve_fig, loss_curve_ax \
            = self.visualizer.draw_loss_curve(
                loss_df=loss_df,
                step_col=self.trainer.step_col,
                loss_col=self.trainer.loss_name,
                dset_col=self.trainer.dset_col
                )
        pred_vs_targ_fig, pred_vs_targ_ax \
            = self.visualizer.draw_pred_vs_targ(
                val_preds_df=val_preds_df,
                pred_col=self.trainer.summary_pred_col,
                targ_col=self.trainer.summary_targ_col
                )

        result_dict = {
            "train_df": train_df,
            "val_df": val_df,
            "model": model,
            "loss_df": loss_df,
            "val_preds_df": val_preds_df,
            "loss_curve_plot": (loss_curve_fig, loss_curve_ax),
            "pred_vs_targ_plot": (pred_vs_targ_fig, pred_vs_targ_ax),
            }
        return result_dict
