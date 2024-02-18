# abb_model_demo

This code is intended to be indicative of my general coding style,
to show some basic data exploration and to demonstrate building
some very simple models using XGBoost and PyTorch.

## Data Set

The data set was obtained at

[http://data.insideairbnb.com/united-states/ca/oakland/2023-12-20/data/listings.csv.gz](http://data.insideairbnb.com/united-states/ca/oakland/2023-12-20/data/listings.csv.gz)

It contains a scrape of 2,749 AirBnB listings in Oakland, CA. The listings have been scrubbed and do not contain all data. The scrap was performed on December 20, 2023. This data set (or an updated version) can be explored at [http://insideairbnb.com/oakland/](http://insideairbnb.com/oakland/).

## Data Exploration

See the following script which was used to visualize the listing price distributions segmented by various potential feature variables.

    scripts/oak_feature_explore.py 

### Host Properties

Properties of the host, such as whether the host is verified or has a profile picture, does not have much discriminating power for listing price, as shown by the plots below. These variables are NOT used as features for predicting listing price.

[host properties plots](abb_model_demo/data/plots/host_tf_props_oaklist.png)




