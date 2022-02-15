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


