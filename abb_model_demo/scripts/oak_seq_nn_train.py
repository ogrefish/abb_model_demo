"""
"""

import os
import logging
import argparse
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from abb_model_demo.modules.data_loaders import CsvFeatureDfBuilder
from abb_model_demo.modules.modelers \
    import AbbNnSeqModelBuilder, AbbNnModelTrainer
from abb_model_demo.modules.optimizers \
    import AbbNnAdadeltaOptimizer, AbbNnAdamOptimizer
from abb_model_demo.modules.visualizers import AbbPredsVisualizer
from abb_model_demo.modules.pipelines import AbbNnSeqPipeline


def get_default_hyperparam_dict():
    """Default set of hyper parameters. Easiest to just change this and re-run.

    Could be moved to a config file -- but it seems convenient here given the
    small size of this demo codebase.
    """
    return {
        "num_epochs": 350,
        "batch_size": 300,
        "shuffle": True,
        "rseed": 1,
        "learn_rate": 3e-4,
        "weight_decay": 1e-3,      # only for adam opt
        "scheduler_step_size": 1,  # only for adadelta opt
        "scheduler_gamma": 0.7,    # only for adadelta opt
        }




def build_nn_seq_pipeline(input_fn, train_frac, logger,
                          optimizer_type, hyperparam_dict):
    """This function is essentially a factory for building pipelines. This
    routine takes care to ensure only components that work well together are
    implemented together in the pipeline.

    Could be separated out into a factory class with multiple build
    (class)methods if the code-base grew enough to warrant it -- i.e. if there
    are multiple options for each component and some combinations don't make
    sense.
    """

    if optimizer_type=="adam":
        optimizer = AbbNnAdamOptimizer(hyperparam_dict=hyperparam_dict)
    elif optimizer_type=="adadelta":
        optimizer = AbbNnAdadeltaOptimizer(hyperparam_dict=hyperparam_dict)
    else:
        raise ValueError(f"Unkown optimizer_type: {optimizer_type}")

    pipeline = AbbNnSeqPipeline(
        df_builder=CsvFeatureDfBuilder(input_fn=input_fn,
                                       train_frac=train_frac,
                                       logger=logger),
        model_builder=AbbNnSeqModelBuilder(),
        optimizer=optimizer,
        trainer=AbbNnModelTrainer(logger=logger,
                                  hyperparam_dict=hyperparam_dict),
        visualizer=AbbPredsVisualizer(),
        hyperparam_dict=hyperparam_dict
        )

    return pipeline


def main(input_fn, train_frac, optimizer_type, log_level, hyperparam_dict,
         plot_save_dir):
    """
    """

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(log_level.upper())

    pipeline = build_nn_seq_pipeline(input_fn=input_fn,
                                     train_frac=train_frac,
                                     logger=logger,
                                     optimizer_type=optimizer_type,
                                     hyperparam_dict=hyperparam_dict)

    # all the action happens here!
    # see `build_nn_seq_pipeline` to tell which class this pipeline object is
    # (it's AbbNnSeqPipeline) and look at the function def inside pipeline.py
    # to see what this does...
    # but no surprise, it sets up the data, model, optimizer, does the
    # training and validation and then makes the plots
    results_dict = pipeline.run_pipeline()

    # save the plots?
    if plot_save_dir is not None:
        odir = os.path.expandvars(plot_save_dir)
        os.makedirs(odir, exist_ok=True)

        loss_curve_fig, _ = results_dict["loss_curve_plot"]
        loss_curve_ofn = os.path.join(odir, "oak_seq_nn_loss_curve.png")
        loss_curve_fig.savefig(loss_curve_ofn)
        logger.info(f"Saved plot {loss_curve_ofn}")

        pred_vs_targ_fig, _ = results_dict["pred_vs_targ_plot"]
        pred_vs_targ_ofn = os.path.join(odir, "oak_seq_nn_pred_vs_targ.png")
        pred_vs_targ_fig.savefig(pred_vs_targ_ofn)
        logger.info(f"Saved plot {pred_vs_targ_ofn}")

    # show the plots
    plt.show()


def get_args():
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
    parser.add_argument("-o", "--optimizer_type",
                        default="adam",
                        choices=["adam", "adadelta"],
                        help="type of optimizer to use during training"
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
         optimizer_type=args.optimizer_type,
         log_level=args.log_level,
         hyperparam_dict=get_default_hyperparam_dict(),
         plot_save_dir=args.plot_save_dir
         )

