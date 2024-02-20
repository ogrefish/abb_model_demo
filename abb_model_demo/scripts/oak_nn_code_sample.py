"""This script is intended to serve as a quickly digestible
code sample. The main goals are:
  * Define a NN model and train it
  * Stay close to the replaceable-modules-in-a-pipeline design
  * Keep the code small, simple, clear

This is a simplified version of the full project's code for training the NN.
The full project shows additional code design features as "composition over
inheritance" style object oriented methods.

*NOTE on the data!* Please see the
[README.md](https://github.com/ogrefish/abb_model_demo/tree/csamp?tab=readme-ov-file#data-set)
section "Data Set" which describes what this data is (and AirBnB scrap), what
the model is predicting (listing price) and how features were found that have
some predictive power for that target. Feature distribution plots are in
`abb_model_demo/data/plots/`.

Sphinx style docstrings have been used in lieu of type hinting. Type hinting is
great, but the lack of enforcement is a significant a drawback (and potentially
confusing for folks new to Python). My hope is the docstrings go a long way to
indicate what types are expected while avoiding confusion that types might be
enforced.

USAGE:
    python abb_model_demo/scripts/oak_nn_code_sample.py

See `get_args` function below for CLI options.

See `pyproject.toml` for dependency version expectations.

"""

# general Python imports
import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abb_model_demo.modules import visualizers
plt.style.use('ggplot')

# ML imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# some components from the full project that are still used here
from abb_model_demo.modules.data_loaders \
    import CsvFeatureDfBuilder, DfDataSet
from abb_model_demo.modules.visualizers import AbbPredsVisualizer


"""Some constants. In the project, these are properties of the training
component; see `AbbNnModelTrainer` in `modules/modelers.py`. These are column
names used in the creation of results dataframes and passed to the
visualization component for plotting. """
STEP_COL="epoch"
DSET_COL="data_set"
LOSS_NAME="mean_squared_error"
SUMMARY_PRED_COL="preds"
SUMMARY_TARG_COL="targs"


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
        "weight_decay": 1e-3,
        }


def run_pipeline(input_fn, train_frac, logger, hyperparam_dict):
    """The main action is here!

    In the full project, pipelines are objects that have components. The
    components are objects that perform some operation, which is why they are
    named with verbs.

    To keep the code sample simple, only the data builder and result visualizer
    components are directly used.

    The other components' functionality has been implemented directly in this
    script to keep things quickly digestible.

    Compare to `AbbNnSeqPipeline.run_pipeline` found in `modules/pipelines.py`

    :param input_fn: the input data filename (with path)
    :type input_fn: str
    :param train_frac: the fraction of data to keep in the training set, with
    1-train_frac reserved for the validation set
    :type train_frac: float
    :param logger: where to log messages to
    :type logger: logging.Logger
    :param hyperparam_dict: dictionary of hyperparameters, needing "rseed",
    "learn_rate" and "weight_decay" keys
    :type hyperparam_dict: dict


    :return: a Python dictionary containing the train and validation data frames
    ("train_df" and "val_df" keys), the model itself ("model" key), the loss
    and predtions-vs-targets results dataframes ("loss_df" and "val_preds_df"
    keys), and two fig/ax tuples for the summary plots ("loss_curve_plot"
    and "pred_vs_targ_plot" keys)
    :rtype: dict

    """

    # fix random seed
    torch.manual_seed(hyperparam_dict["rseed"])

    # find the device
    device = _find_device()

    # setup train & validation data - here an actual component is used
    # see `CsvFeatureDfBuilder` in `modules/data_loaders.py`
    df_builder = CsvFeatureDfBuilder(input_fn=input_fn,
                                     train_frac=train_frac,
                                     logger=logger)
    train_df, val_df = df_builder.get_train_val_dfs()
    train_dataloader = _get_data_loader(
        data_df=train_df,
        feature_col_list=df_builder.feature_col_list,
        target_col=df_builder.target_col,
        hyperparam_dict=hyperparam_dict
        )
    val_dataloader = _get_data_loader(
        data_df=val_df,
        feature_col_list=df_builder.feature_col_list,
        target_col=df_builder.target_col,
        hyperparam_dict=hyperparam_dict
        )

    # make the model object.
    # 5 features: property type category, number of beds,
    # accommodation capacity, neighborhood category and availability yes/no
    num_features = len(df_builder.feature_col_list)
    model = get_new_model(device=device, num_features=num_features)
                          
    # build the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparam_dict["learn_rate"],
        weight_decay=hyperparam_dict["weight_decay"]
        )

    # train & validate
    loss_df, val_preds_df = train_and_validate(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        optimizer=optimizer,
        device=device,
        hyperparam_dict=hyperparam_dict,
        logger=logger
        )

    # plots ! here another actual component is used
    # see `AbbPredsVisualizer` in `modules/visualizers.py`
    visualizer = AbbPredsVisualizer()
    loss_curve_fig, loss_curve_ax \
        = visualizer.draw_loss_curve(
            loss_df=loss_df,
            step_col=STEP_COL,
            loss_col=LOSS_NAME,
            dset_col=DSET_COL
        )
    pred_vs_targ_fig, pred_vs_targ_ax \
        = visualizer.draw_pred_vs_targ(
            val_preds_df=val_preds_df,
            pred_col=SUMMARY_PRED_COL,
            targ_col=SUMMARY_TARG_COL
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
    

def _find_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def _get_data_loader(data_df, feature_col_list, target_col,
                     hyperparam_dict):
    """

    :param data_df: a Pandas DataFrame containing the data to be loaded
    :type data_df: pd.DataFrame
    :param feature_col_list: a list of feature column names found in data_df
    :type feature_col_list: List[str]
    :param target_col: name of the target (label) column in data_df
    :type target_col: str
    :param hyperparam_dict: dictionary of hyperparameters, needing "shuffle"
    and "batch_size" keys
    :type hyperparam_dict: dict

    :return: DataLoader for the data in the input dataframe
    :rtype: torch.utils.data.DataLoader
    """
    dataset = DfDataSet(data_df=data_df,
                        feature_col_list=feature_col_list,
                        target_col=target_col
                        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hyperparam_dict["batch_size"],
        shuffle=hyperparam_dict["shuffle"]
        )
    return dataloader


def get_new_model(device, num_features):
    """
    Make a simple 3-layer NN model with more parameters than is necessary
    for the features and data set size, to see if it can nevertheless
    converge to a model that makes reasonable predictions on the validation
    data.

    :param device: which device to load the model to. see `_find_device`
    :type device: str
    :param num_features: the number of features to use as input in 1st layer
    :type num_features: int
    :return: the neural network model object
    :rtype: nn.Sequential
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


def _train_batch(dataloader, model, optimizer, device):
    """Trains over batches of data in the dataloader. Expected to be called
    from a loop over epochs. Loss is MSE.

    :param dataloader: dataloader for feature & target (training) data
    :type dataloader: torch.utils.data.DataLoader
    :param model: the (NN) model to fit to the data
    :type model: torch.nn.Module (specifically torch.nn.Sequential)
    :param optimizer: the optimizer to use to minimize loss
    :type optimizer: torch.optim.Optimizer (specifically torch.optim.Adam)
    :device: the device to send the data to for model training
    :type device: str

    :return: average loss over the batch steps
    :rtype: float
    """
    model.train()
    avg_loss = 0.0
    for bi, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss /= len(dataloader)  # div by num batches
    return avg_loss
        

def validate(dataloader, model, device, build_preds_df=False):
    """Calculates predictions of the model over the specified data. Optionally
    builds a summary DataFrame of prediction and target values.

    :param dataloader: dataloader for feature & target (validation) data
    :type dataloader: torch.utils.data.DataLoader
    :param model: the (NN) model to fit to the data
    :type model: torch.nn.Module (specifically torch.nn.Sequential)
    :device: the device to send the data to for model training
    :type device: str
    :param build_preds_df: (Optional; default=False) whether to build a summary
    dataframe of the predicted and target values
    :type build_preds_df: boolean

    :return: tupl of the avg loss (MSE) over the data, and an optional summary
    Pandas DataFrame containing predicted and target values
    :rtype: Tuple[float, Optional[pd.DataFrame]]

    """
    model.eval()
    avg_loss = 0.0
    bdf_list= []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss += F.mse_loss(output, target, reduction="sum")\
                         .item()
            if build_preds_df:
                df = pd.DataFrame(data=np.hstack([output.cpu().numpy(),
                                                  target.cpu().numpy()
                                                  ]),
                                  columns=[SUMMARY_PRED_COL,
                                           SUMMARY_TARG_COL]
                                  )
                bdf_list.append(df)
    num_pts = len(dataloader.dataset)  # num points (not num batches)
    avg_loss /= num_pts  

    if len(bdf_list)>0:
        df = pd.concat(bdf_list)
    else:
        df = None
    return avg_loss, df


def train_and_validate(train_dataloader, val_dataloader,
                       model, optimizer, device,
                       hyperparam_dict, logger):
    """Trains the model by looping over batches of data in the train dataloader,
    then repeating that process for multiple epochs. Loss being minimized is
    mean square error (MSE). Also runs the model over the validation data set
    in order to generate the loss curves during training as well as a final
    summary of predicted versus target (listing price) values.

    :param train_dataloader: dataloader for feature & target (training) data
    :type train_dataloader: torch.utils.data.DataLoader
    :param val_dataloader: dataloader for feature & target (validation) data
    :type val_dataloader: torch.utils.data.DataLoader
    :param model: the (NN) model to fit to the data
    :type model: torch.nn.Module (specifically torch.nn.Sequential)
    :param optimizer: the optimizer to use to minimize loss
    :type optimizer: torch.optim.Optimizer (specifically torch.optim.Adam)
    :device: the device to send the data to for model training
    :type device: str
    :param hyperparam_dict: dictionary of hyperparameters, needing
    "num_epochs" key
    :type hyperparam_dict: dict

    :return: average loss over the batch steps
    :rtype: float
    """
    loss_list = []
    for epoch in range(hyperparam_dict["num_epochs"]):
        train_loss = _train_batch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            device=device
            )
        val_loss, val_df = validate(
            dataloader=val_dataloader,
            model=model,
            device=device,
            build_preds_df=False
            )

        loss_list.append( (epoch, "train", train_loss) )
        loss_list.append( (epoch, "val", val_loss) )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Epoch {epoch}: train loss {train_loss:0.03f} "
                f"val loss {val_loss:0.03f}"
                )

    loss_df = pd.DataFrame.from_records(
        data=loss_list,
        columns=[STEP_COL, DSET_COL, LOSS_NAME]
        )

    _, val_preds_df = validate(
        dataloader=val_dataloader,
        model=model,
        device=device,
        build_preds_df=True
        )

    return loss_df, val_preds_df


def main(input_fn, train_frac, log_level, hyperparam_dict):
    """
    Streamlined but analogous to `main` in `oak_seq_nn_train.py`.

    Creates the logging object, calls `run_pipeline` (where all
    the fun stuff happens) and then displays the plots.

    :param input_fn: input filename of data (with path)
    :type input_fn: str
    :param train_frac: the fraction of data to keep in the training set, with
    1-train_frac reserved for the validation set
    :type train_frac: float
    :param log_level: level to use for logging messages
    :type log_level: str
    :param hyperparam_dict: dictionary of hyperparameters, needing "rseed",
    "learn_rate" and "weight_decay" keys
    :type hyperparam_dict: dict

    :return: a Python dictionary containing the train and validation data frames
    ("train_df" and "val_df" keys), the model itself ("model" key), the loss
    and predtions-vs-targets results dataframes ("loss_df" and "val_preds_df"
    keys), and two fig/ax tuples for the summary plots ("loss_curve_plot"
    and "pred_vs_targ_plot" keys)
    :rtype: dict
    """

    logger = logging.getLogger(__name__)
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(log_level.upper())

    # all the action happens here!
    # analogous to `AbbNnSeqPipeline.run_pipeline` in the full project
    results_dict = run_pipeline(input_fn=input_fn,
                                train_frac=train_frac,
                                logger=logger,
                                hyperparam_dict=hyperparam_dict
                                )
    logger.info(f"Done! Got results dict with keys: {results_dict.keys()}")

    # show the plots
    plt.show()

    return results_dict


def get_args():
    """
    Parses command line arguments and returns the arguments object

    :return: the parsed command line arguments
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-i", "--input_fn",
                        default="abb_model_demo/data/listings.csv.gz",
                        help="full path to input data file. may include env vars"
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
         hyperparam_dict=get_default_hyperparam_dict()
         )

