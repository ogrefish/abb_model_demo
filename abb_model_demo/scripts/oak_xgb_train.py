"""
Steering script for the XGBoost model training.
"""

import os
import logging
import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from abb_model_demo.modules.data_loaders import CsvFeatureDfBuilder
from abb_model_demo.modules.modelers \
    import AbbXgbModelBuilder, AbbXgbModelTrainer
from abb_model_demo.modules.visualizers import AbbPredsVisualizer
from abb_model_demo.modules.pipelines import AbbXgbPipeline


def get_default_hyperparam_dict():
    """Default set of hyper parameters. Easiest to just change this and re-run.

    Could be moved to a config file -- but it seems convenient here given the
    small size of this demo codebase.
    """
    return {
        "learn_rate": 5e-2,
        "num_estimators": 160,
        "subsample": 0.80,
        "colsample_bytree": 0.35,
        "gamma": 0.50,
        "tree_method": "hist",
        "enable_categorical": True,
        "rseed": 1,
        "fit_verbose": False
        }


def build_xgb_pipeline(input_fn, train_frac, logger, hyperparam_dict):
    """This function is essentially a factory for building pipelines. This
    routine takes care to ensure only components that work well together are
    implemented together in the pipeline.

    Could be separated out into a factory class with multiple build
    (class)methods if the code-base grew enough to warrant it -- i.e. if there
    are multiple options for each component and some combinations don't make
    sense.

    Args:
        input_fn (str): Path to the input data file.
        train_frac (float): Fraction of the training data to use for building the pipeline.
        logger (logging.Logger): Logger object to record pipeline building progress.
        hyperparam_dict (dict): Dictionary of hyperparameters for the optimizer.

    Returns:
        AbbXgbPipeline: A sequence neural network pipeline.
    """

    pipeline = AbbXgbPipeline(
        df_builder=CsvFeatureDfBuilder(input_fn=input_fn,
                                       train_frac=train_frac,
                                       logger=logger),
        model_builder=AbbXgbModelBuilder(hyperparam_dict=hyperparam_dict),
        trainer=AbbXgbModelTrainer(logger=logger,
                                   hyperparam_dict=hyperparam_dict),
        visualizer=AbbPredsVisualizer(),
        )

    return pipeline


def main(input_fn, train_frac, log_level, hyperparam_dict,
         plot_save_dir):
    """
    Build and run the model training pipeline. Then save summary plots.

    Args:
        input_fn (str): Path to the input data file.
        train_frac (float): Fraction of the training data to use for building the pipeline.
        log_level (str): the level at which to log using `logging`
        hyperparam_dict (dict): Dictionary of hyperparameters for the optimizer.
        plot_save_dir (str): the directory in which to save plots.
           if None, no plots will be saved
    """

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(log_level.upper())

    pipeline = build_xgb_pipeline(input_fn=input_fn,
                                  train_frac=train_frac,
                                  logger=logger,
                                  hyperparam_dict=hyperparam_dict)

    # all the action happens here!
    # see `build_xgb_pipeline` to tell which class this pipeline object is
    # (it's AbbXgbPipeline) and look at the function def inside pipeline.py
    # to see what this does...
    # but no surprise, it sets up the data, model, optimizer, does the
    # training and validation and then makes the plots
    results_dict = pipeline.run_pipeline()

    # save the plots?
    if plot_save_dir is not None:
        odir = os.path.expandvars(plot_save_dir)
        os.makedirs(odir, exist_ok=True)

        loss_curve_fig, _ = results_dict["loss_curve_plot"]
        loss_curve_ofn = os.path.join(odir, "oak_xgb_loss_curve.png")
        loss_curve_fig.savefig(loss_curve_ofn)
        logger.info(f"Saved plot {loss_curve_ofn}")

        pred_vs_targ_fig, _ = results_dict["pred_vs_targ_plot"]
        pred_vs_targ_ofn = os.path.join(odir, "oak_xgb_pred_vs_targ.png")
        pred_vs_targ_fig.savefig(pred_vs_targ_ofn)
        logger.info(f"Saved plot {pred_vs_targ_ofn}")

    # show the plots
    plt.show()


def get_args():
    """
    Parses command line arguments and returns the arguments object

    Returns:
       argparse.Namespace: the parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-i", "--input_fn",
                        default="abb_model_demo/data/listings.csv.gz",
                        help="full path to input data file. may include env vars"
                        )
    parser.add_argument("-p", "--plot_save_dir",
                        default=None,
                        help="output directory in which to store train/test files"
                        )
    parser.add_argument("-f", "--train_frac",
                        default=0.80,
                        help="fraction of data to keep in training set. "
                        "will keep 1-train_frac in the validation set."
                        )
    parser.add_argument("-l", "--log_level",
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="the log level to use for log messages"
                        )
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    main(input_fn=args.input_fn,
         train_frac=args.train_frac,
         log_level=args.log_level,
         hyperparam_dict=get_default_hyperparam_dict(),
         plot_save_dir=args.plot_save_dir
         )

