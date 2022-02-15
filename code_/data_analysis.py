import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class EDA(object):
    def __init__(self, config_path: str, eda_config_path: str):
        with open(config_path, "r") as jsonfile:
            self.config = json.load(jsonfile)
        print("Read successful")

        with open(eda_config_path, "r") as jsonfile:
            self.eda_config = json.load(jsonfile)
        print("Read successful")

        self.df_path = self.config.get("feature_engineered_data_path")
        self.df = pd.read_csv(self.df_path)
        self.missing_value_pct_threshold = self.eda_config.get("missing_value_pct_threshold")
        self.correlation_threshold = self.eda_config.get("correlation_threshold")
        self.categorical_pct_threshold = self.eda_config.get("categorical_pct_threshold")
        self.zero_pct_threshold = self.eda_config.get("zero_pct_threshold")
        self.num_categories_threshold = self.eda_config.get("num_categories_threshold")

        self.categorical_cols = [col for col in self.df.columns if col.startswith(("number_", "most_frequent_", "customer_")) or col.endswith("_frequency")]
        self.numerical_cols = [col for col in self.df.columns if col not in self.categorical_cols]

        self.df[self.categorical_cols] = self.df[self.categorical_cols].astype('str')
        self.df[self.numerical_cols] = self.df[self.numerical_cols].astype('float')

    # Get the categorical nad numerical features of a dataframe.
    def get_cat_num_features(self):
        self.categorical_cols = [col for col in self.df.columns if col.startswith(("number_", "most_frequent_", "customer_")) or col.endswith("_frequency")]
        self.numerical_cols = [col for col in self.df.columns if col not in self.categorical_cols]
        return

    def remove_missing_value(self, is_drop=True) -> pd.DataFrame:
        """
        get information regarding missing values of each column of a dataframe.
        the method remove columns enjoying more than missing_value_pct_threshold missing values.
        :param: is_drop: if True, the method will drop the columns enjoying more than missing_value_pct_threshold missing values.
        :return: dataframe with missing values information.
        """
        self.get_cat_num_features()

        # Create dataframe with missing values information.
        missing_value_df = pd.DataFrame(self.df.isnull().sum()).reset_index()
        missing_value_df.columns = ['column_name', 'missing_count']
        missing_value_df['missing_pct'] = (missing_value_df['missing_count'] / len(self.df)) * 100

        # Sort the dataframe by missing_pct in descending order.
        missing_value_df = missing_value_df.sort_values(by='missing_pct', ascending=False)

        # Write the missing value information to a csv file.
        missing_value_df.to_csv(self.config.get("missing_value_df_path"), index=False)

        # Get the columns including more than missing_value_pct_threshold missing values.
        to_drop = missing_value_df[missing_value_df['missing_pct'] >= self.missing_value_pct_threshold]['column_name'].tolist()

        # Drop the columns with more than missing_value_pct_threshold missing values if is_drop=True.
        if is_drop and len(to_drop) > 0:
            self.df.drop(to_drop, axis=1, inplace=True)
            self.get_cat_num_features()

        return missing_value_df


