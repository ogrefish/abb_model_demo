"""
In this land are found modules that prepare data for training
and evaluation.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud


class CsvFeatureDfBuilder(object):
    """Prepares specific features for the (Oakland) "instide AirBnB"
    data set. These features were determined via data exploration
    (see scripts).

    Takes an input file (csv or gzip csv) to a Pandas DF of
    cleaned feature & target data.

    Note on OO design: if we had several data sets and needed different feature
    builders, we could define an A.B.C. for this to inherit a standardized
    interface. As this is not the case, we the design is kept simple for now.
    Future extension to refactor would be straight forward.

    Note on method design: attempts have been made to avoid non-obvious side
    effects. This is why internal methods accept inputs (like the df) rather
    than store them as class members. Function names should indicate whether
    they directly alter their inputs (add, clean, etc).

    Note on column names: decision to hard-code column names is intended to
    drive simplicity & readability. In the future, if there is a variety of
    data sources that use different names for the same variable, then it's
    straight-forward to refactor and build a "variable context" class or
    dictionary.

    """


    def __init__(self, input_fn, train_frac, logger):
        """
        Initializes the DataLoader class.

        Args:
            input_fn (str): Path to the input csv file.
            train_frac (float): Fraction of the training data to load.
            logger (logging.Logger): A logger object to record progress.
        """
        self.input_fn = input_fn
        self.train_frac = train_frac
        self.logger = logger


    @property
    def feature_col_list(self):
        """
        The list of feature column names
        """
        return [
            "property_type_fw_id",
            "accommodates",
            "beds",
            "neighbourhood_cleansed_id",
            "has_availability_num",
            ]


    @property
    def target_col(self):
        """
        The target (to be predicted) column name
        """
        return "log_price_amt"


    def _get_input_df(self, input_fn):
        """
        Reads the input file and returns a pandas dataframe.

        Args:
            input_fn (str): Path to the input csv file.

        Returns:
            pandas.DataFrame: A dataframe containing the loaded data.
        """
        infn = os.path.expandvars(input_fn)
        df = pd.read_csv(infn)
        self.logger.info(f"Read input file: {infn}")
        return df


    def _clean_prices(self, df):
        """
        Removes records with missing target valeus and formats the price column.

        Args:
            df (pandas.DataFrame): The input dataframe.

        Returns:
            pandas.DataFrame: The cleaned dataframe.
        """

        # clear records that are missing the target variable
        cdf = df.loc[~(df["price"].isnull())]\
                .reset_index(drop=True)
        # add numerical price column
        cdf.loc[:, "price_amt"] = pd.to_numeric(
            cdf.loc[:, "price"].str.replace("$", "", regex=False)\
                               .str.replace(",", "", regex=False)\
                               .str.strip()
            )
        # use log10 (not ln) to keep values interpret-able
        cdf.loc[:, "log_price_amt"] = np.log10(cdf.loc[:, "price_amt"])
        return cdf


    def _add_feat_cols(self, df):
        """
        Adds feature columns to the dataframe.
        * True/false strings translated to ordinal ("t" to 1, "f" to 0)
        * Special care taken with "beds" column: missing values
          are filled with "accommodates"/2.0
        * The first word of the "property_type" string is selected
        * Categorical features are created through simple ordinal labeling of
          distinct values for: property type first word and neighborhood
        * Categorical features are given the dtype "category"

        See class notes about why hard-coded column names are in use.
        Basically, prioritizing:
          * Readability & simplicity
          * Avoiding user error (e.g. a typo makes a model not trainable)
          * Ensuring the design will make extension / refactor simple
            in case of a future use case for varying column names

        Args:
            df (pandas.DataFrame): The input dataframe.

        Returns:
            pandas.DataFrame: The resulting dataframe including new
               feature columns.
        """
        # NOTE: column names must match target_col and feature_col_list
        # properties
        df.loc[:, "property_type_fw"] = df.loc[:, "property_type"]\
                                          .str.split().str.get(0)
        df.loc[:, "property_type_fw_id"] \
            = df.groupby("property_type_fw").transform("ngroup")
        df.loc[:, "neighbourhood_cleansed_id"] \
            = df.groupby("neighbourhood_cleansed").transform("ngroup")

        def translate_tf(val):
            # use dict so an exception will be raised for unexpected
            # values and we'll know something bad happened
            val_map = { "t": 1, "f": 0, None: None }
            return val_map[val]
        df.loc[:, "has_availability_num"] = df.loc[:, "has_availability"]\
                                              .transform(translate_tf)\
                                              .astype("int")
        # beds is sometimes null. approximate it from accommodates.
        # this is potentially "tricksy" and its impact on a model may
        # need to be investigated
        null_bed_mask = df["beds"].isnull()
        df.loc[null_bed_mask, "beds"] = np.floor(
            np.maximum(
                np.divide(df.loc[null_bed_mask, "accommodates"], 2.0),
                1
                )\
            )

        # set explicit category dtypes
        df = df.astype( { c:"category" for c
                          in ["property_type_fw_id",
                              "neighbourhood_cleansed_id",
                              "has_availability_num"]
                          }
                        )
        return df


    def _get_train_val_split(self, idf, train_frac):
        """
        Gets the training and validation split of the data.

        Args:
            idf (pandas.DataFrame): A pandas DataFrame containing
              the full dataset.
            train_frac (float): Fraction of the training data to load

        Returns:
            tuple: A tuple containing the training and validation data
             dataframes.
        """
        # shuffle all the data then slice
        shuffle_df = idf.sample(frac=1).reset_index(drop=True)
        num_samples = len(shuffle_df)
        assert(num_samples>9), \
            f"Not enough samples to train/val split. Have {num_samples}; need 10"
        first_val_samp = int(num_samples*train_frac)
        train_df, val_df = shuffle_df.iloc[:first_val_samp], \
                           shuffle_df.iloc[first_val_samp:]
        return train_df, val_df


    def _get_model_variables_df(self, idf):
        """
        Choose only the columns relevant for training.

        Args:
            idf (pandas.DataFrame): A pandas DataFrame

        Returns:
            pandas.DataFrame: A pandas DataFrame with only the target and
              feature columns
        """
        return idf.loc[:, [self.target_col] + self.feature_col_list]


    def save_df(self, odf, output_dir, output_fn):
        """
        Saves the raw dataframe to a csv file.

        Args:
            odf (pandas.DataFrame): A pandas DataFrame containing data to
              be saved.
            output_dir (str): Directory for the save file.
            output_fn (str): Name of the save file.
        """
        ofn = os.path.join(os.path.expandvars(output_dir),
                           output_fn)
        res = odf.to_csv(ofn, index=False, mode="w")
        self.logger.info(f"Saved {odf.shape} DF to: {ofn}")
        return ofn, res


    def get_train_val_dfs(self):
        """
        Main "pipeline" function that performs each step:
          * read data from disk
          * clean price data
          * create feature columns
          * select only cols relevant for training
          * randomly split data to train/val sets

        Returns:
            tuple: A tuple containing the train DataFrame and
              the validation DataFrame.
        """
        idf = self._get_input_df(input_fn=self.input_fn)
        idf = self._clean_prices(idf)
        idf = self._add_feat_cols(idf)
        idf = self._get_model_variables_df(idf)
        train_df, val_df = self._get_train_val_split(idf=idf,
                                                     train_frac=self.train_frac)
        self.logger.info(f"train data size {train_df.shape}")
        self.logger.info(f"validation data size {val_df.shape}")
        return train_df, val_df


class DfDataSet(tud.Dataset):
    """A PyTorch dataset class to load and prepare train and validation data
      from a Pandas DataFrame.
    """

    def __init__(self, data_df, feature_col_list, target_col):
        """
        Init "dataframe dataset" class. The feature & target sensor must be
        None in order for the automatic data loading to happen on the first
        iteration through the data.

        Args:
            data_df (pandas.DataFrame): A DataFrame containing the training data.
            feature_col_list (list): A list of feature column names.
            target_col (str): The name of the target column.
        """
        super().__init__()
        self.feature_tensor = None
        self.target_tensor = None
        self.data_df = data_df
        self.feature_col_list = feature_col_list
        self.target_col = target_col

        if isinstance(self.data_df, pd.DataFrame)==False:
            raise TypeError(f"unexpected data_df type: {type(self.data_df)}")


    def load_data(self):
        """
        Create the feature and target PyTorch Tensor objects from the
        underlying Pandas DataFrame. Intended to be called automatically
        from `self.__getitem__` if and only if the feature/target tensor
        objects are both None (i.e. on the first iteration through the data)
        """
        if (self.feature_tensor is None) and (self.target_tensor is None):
            self.feature_tensor = torch.tensor(
                self.data_df.loc[:, self.feature_col_list].values
                )
            self.target_tensor = torch.tensor(
                self.data_df.loc[:, self.target_col].values.reshape(-1, 1)
                )
        else:
            raise AttributeError(f"data already loaded. feats={self.feature_tensor}, "
                                 f"tgt={self.target_tensor}"
                                 )


    def __len__(self):
        """Returns the length of the underyling Pandas DataFrame
        """
        return len(self.data_df)


    def __getitem__(self, idx):
        """
        Method to retrieve a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the features and target PyTorch Tensor
              objects for the sample at the given index.
        """
        if (self.feature_tensor is None) and (self.target_tensor is None):
            self.load_data()
        return self.feature_tensor[idx], self.target_tensor[idx]
