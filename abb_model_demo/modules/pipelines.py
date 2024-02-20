"""
"""

import torch
from torch.utils.data import DataLoader

from abb_model_demo.modules.data_loaders import DfDataSet


class AbbNnSeqPipeline():
    """
    """

    def __init__(self, df_builder, model_builder, optimizer,
                 trainer, visualizer, hyperparam_dict):
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
            emsg = f"missing hyper parameters: {missing_param_list}"
            raise ValueError(f"AbbNnSeqPipeline.__init__ {emsg}")


    @property
    def expected_hyperparam_list(self):
        return [
            "batch_size",
            "shuffle",
            "rseed",
            ]


    def _get_data_loader(self, data_df):
        dataset = DfDataSet(data_df=data_df)
        dataloader = DataLoader(dataset,
                                batch_size=self.hyperparam_dict["batch_size"],
                                shuffle=self.hyperparam_dict["shuffle"]
                                )
        return dataloader


    def run_pipeline(self):

        # fix random seed
        torch.manual_seed(self.hyperparam_dict["rseed"])

        # setup train & validation data
        train_df, val_df = self.df_builder.get_train_val_dfs()
        train_dataloader = self._get_data_loader(data_df=train_df)
        val_dataloader = self._get_data_loader(data_df=val_df)

        # get the model object
        model = self.model_builder.get_new_model()

        # train & validate
        loss_df, val_preds_df = self.trainer.train_and_validate()

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

