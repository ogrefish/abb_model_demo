"""
Simple script used to make exploratory plots, focusing
on features that have some discriminating power on
the listing price.

Note that no descriptions or amenities are available in
the data set being used.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.ticker as mtick
import seaborn as sns



def get_train_df(input_dir, train_fn):
    return pd.read_csv(os.path.expandvars(
        os.path.join(input_dir, train_fn))
        )


def save_plot(fig, plot_name, plot_dir, plot_tag, verbosity=10):
    ppath = os.path.expandvars(plot_dir)
    os.makedirs(ppath, exist_ok=True)
    fpn = os.path.join(ppath, f"{plot_name}_{plot_tag}.png")
    fig.savefig(fpn)
    if verbosity>9:
        print(f"saved {fpn}")


def draw_host_tf_props(df, plot_dir, plot_tag):
    g = sns.displot(df,
                    x="price_amt",
                    col="host_has_profile_pic",
                    row="host_identity_verified",
                    hue="host_is_superhost",
                    log_scale=[True, False],
                    )
    if plot_tag is not None:
        save_plot(fig=g.figure, plot_name="host_tf_props",
                  plot_dir=plot_dir, plot_tag=plot_tag)
    return g


def draw_property_type(df, plot_dir, plot_tag):
    fig, ax = plt.subplots(1)
    g = sns.histplot(df,
                     x="price_amt",
                     hue="property_type_fw",
                     log_scale=[True, False],
                     ax=ax
                     )
    if plot_tag is not None:
        save_plot(fig=g.figure, plot_name="prop_type_fw",
                  plot_dir=plot_dir, plot_tag=plot_tag)
    return g


def draw_accom_beds(df, plot_dir, plot_tag):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    g = sns.boxplot(df,
                    y="price_amt",
                    x="accommodates",
                    log_scale=[False, True],
                    ax=ax[0]
                    )
    g = sns.boxplot(df,
                    y="price_amt",
                    x="beds",
                    log_scale=[False, True],
                    ax=ax[1]
                    )
    fig.tight_layout()
    if plot_tag is not None:
        save_plot(fig=g.figure, plot_name="accom_beds",
                  plot_dir=plot_dir, plot_tag=plot_tag)
    return g


def draw_neighborhoods(df, plot_dir, plot_tag):
    fig, ax = plt.subplots(1, figsize=(15,5))
    g = sns.boxplot(df,
                    y="price_amt",
                    x="neighbourhood_cleansed_id",
                    log_scale=[False, True],
                    ax=ax
                    )
    g.tick_params(axis="x", labelrotation=-45, labelsize="xx-small")
    fig.tight_layout()
    if plot_tag is not None:
        save_plot(fig=g.figure, plot_name="neighborhoods",
                  plot_dir=plot_dir, plot_tag=plot_tag)
    return g


def draw_avail(df, plot_dir, plot_tag):
    fig, ax = plt.subplots(1)
    g = sns.boxplot(df,
                    x="has_availability",
                    y="price_amt",
                    log_scale=[False, True],
                    ax=ax
                    )
    if plot_tag is not None:
        save_plot(fig=g.figure, plot_name="avail",
                  plot_dir=plot_dir, plot_tag=plot_tag)
    return g


def main(input_dir, plot_dir, train_fn, test_fn, plot_tag):

    df = get_train_df(input_dir=input_dir, train_fn=train_fn)

    ##
    ## basic distributions
    ##

    # host t/f properties
    # majority of hosts have identity verified and profile pic both True
    # there is not much price separation
    draw_host_tf_props(df=df, plot_dir=plot_dir, plot_tag=plot_tag)

    # property type separates price. many types -- keep only first word
    draw_property_type(df=df, plot_dir=plot_dir, plot_tag=plot_tag)

    # accomodates & beds separate price
    draw_accom_beds(df=df, plot_dir=plot_dir, plot_tag=plot_tag)

    # neighborhood separates price, but can be sparse
    draw_neighborhoods(df=df, plot_dir=plot_dir, plot_tag=plot_tag)

    # availability has some variance in price.. maybe random?
    draw_avail(df=df, plot_dir=plot_dir, plot_tag=plot_tag)

    plt.show()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("-i", "--input_dir",
                        help="full path to input data file. may include env vars"
                        )
    parser.add_argument("-o", "--plot_dir",
                        help="output directory in which to store train/test files"
                        )
    parser.add_argument("-r", "--train_fn",
                        default="train_listing.csv.gz",
                        help="filename in which to store the train data"
                        )
    parser.add_argument("-s", "--test_fn",
                        default="test_listing.csv.gz",
                        help="filename in which to store the test data"
                        )
    parser.add_argument("-p", "--plot_tag",
                        default=None,
                        help="tag to put in filename of plots. "
                        "if None, no plots will be saved"
                        )
    return parser.parse_args()


if __name__=="__main__":

    args = get_args()
    main(input_dir=args.input_dir,
         plot_dir=args.plot_dir,
         train_fn=args.train_fn,
         test_fn=args.test_fn,
         plot_tag=args.plot_tag
         )

