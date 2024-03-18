"""
In this land are found modules that create plots.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class AbbPredsVisualizer():
    """
    Class for creating plots using summary DataFrames
    """

    def __init__(self):
        pass


    def draw_loss_curve(self, loss_df, step_col, loss_col, dset_col):
        """
        Draw the loss curve for a given dataset.

        Args:
            loss_df (pd.DataFrame): A DataFrame containing the loss values.
            step_col (str): The column containing step (or epoch) values.
            loss_col (str): The column containing loss values.
            dset_col (str): The column indicating data set type (train or val)

        Returns:
            matplotlib.figure, matplotlib.axes: A tuple containing the figure and the axes.
        """
        fig, ax = plt.subplots(1)
        g = sns.lineplot(data=loss_df,
                         x=step_col,
                         y=loss_col,
                         hue=dset_col,
                         ax=ax)
        loss_max = loss_df.loc[:, loss_col].max()
        loss_min = loss_df.loc[:, loss_col].min()
        if (loss_max>0) and (loss_min>0):
            loss_factor = np.divide(loss_max, loss_min)
            if loss_factor>10.0:
                # yaxis covers more than one order of magnitude.. make it log
                g.set(yscale="log")
        fig.tight_layout()
        return fig, ax


    def draw_pred_vs_targ(self, val_preds_df, pred_col, targ_col,
                          val_range=(1e1, 5e3)):
        """
        Draw the prediction vs target plot for a given dataset.

        Args:
            val_preds_df (pd.DataFrame): A DataFrame containing the prediction values.
            pred_col (str): The column containing predicted prices.
            targ_col (str): The column containing target prices.
            val_range (tuple): A tuple of (lower, upper) values for price range.

        Returns:
            matplotlib.figure, matplotlib.axes: A tuple containing the figure and the axes.
        """
        fig, ax = plt.subplots(1)
        g = sns.histplot(x=np.power(10.0, val_preds_df.loc[:, targ_col]),
                         y=np.power(10.0, val_preds_df.loc[:, pred_col]),
                         log_scale=[True, True],
                         cbar=True,
                         cbar_kws={"label": "num listings"},
                         ax=ax
                        )
        ax.set_xlim(*val_range)
        ax.set_ylim(*val_range)
        ax.set_xlabel("target listing price ($)")
        ax.set_ylabel("predicted listing price ($)")
        ax.plot(ax.get_xlim(), ax.get_ylim(),
                linestyle="-", color="white",
                lw=1, scalex=False, scaley=False,
                alpha=0.65)
        fig.tight_layout()
        return fig, ax


