# abb_model_demo

This code is intended to be indicative of my general coding style,
to show some basic data exploration and to demonstrate building
some very simple models using XGBoost and PyTorch.

## Data Set

The data set was obtained at

[http://data.insideairbnb.com/united-states/ca/oakland/2023-12-20/data/listings.csv.gz](http://data.insideairbnb.com/united-states/ca/oakland/2023-12-20/data/listings.csv.gz)

It contains a scrape of 2,749 AirBnB listings in Oakland, CA from December 20, 2023. The listings have been scrubbed (by insideairbnb.com) and do not contain text descriptions nor amenities. This data set (or an updated version) can be explored at [http://insideairbnb.com/oakland/](http://insideairbnb.com/oakland/).

## Data Exploration

See the following script which was used to visualize the listing price distributions segmented by various potential feature variables.

    scripts/oak_feature_explore.py 

### Capacity

The number of beds and number of people a listing can accommodate are often correlated with the listing price.

![accommodates plots](abb_model_demo/data/plots/accom_beds_oaklist.png)

### Property Type

The data set has both a room type and a property type label. These two features are highly correlated with each other. For example, a listing with a room type "private room" will also have a property type "private apartment".

The property type feature was created by keeping only the first word of the property type. This buckets similar listings with similar price distributions together. The room type is not used as a feature, as it is redundant.

The first word of the property type does separate listing price.

![property type plots](abb_model_demo/data/plots/prop_type_fw_oaklist.png)

### Availability

Listings that are not available generally have a higher price. On the buy side, this could indicate that the most desirable listings are the most likely to rent out, even if they cost more. On the lease side, this could reflect urgency to rent the listing out and avoid empty lodging that generates no income.

(Box plot used here because Seaborn did not draw layered 1D histograms nicely on log-counts vs log-price scale axes.)

![availability plots](abb_model_demo/data/plots/avail_oaklist.png)


### Neighborhood

The listing price does vary with neighborhood, although some neighborhoods have very few example listings.

A future improvement worth exploring could be to bucket neighborhoods with similar price distributions together to improve signal to noise for the model.

![neighborhood plots](abb_model_demo/data/plots/neighborhoods_oaklist.png)

### Host Properties

Properties of the host, such as whether the host is verified or has a profile picture, does not have much discriminating power for listing price, as shown by the plots below. These variables are NOT used as features for predicting listing price.

![host properties plots](abb_model_demo/data/plots/host_tf_props_oaklist.png)




