import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


class EDA(object):
    def __init__(self, config_path: str, eda_config_path: str, figures_config_path: str):
        with open(config_path, "r") as jsonfile:
            self.config = json.load(jsonfile)
        print("Read successful")

        with open(eda_config_path, "r") as jsonfile:
            self.eda_config = json.load(jsonfile)
        print("Read successful")

        with open(figures_config_path, "r") as jsonfile:
            self.figures_config = json.load(jsonfile)
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
        missing_value_df.to_csv(self.config.get("missing_value_features_path"), index=False)

        # Get the columns including more than missing_value_pct_threshold missing values.
        to_drop = missing_value_df[missing_value_df['missing_pct'] >= self.missing_value_pct_threshold]['column_name'].tolist()

        # Drop the columns with more than missing_value_pct_threshold missing values if is_drop=True.
        if is_drop and len(to_drop) > 0:
            self.df.drop(to_drop, axis=1, inplace=True)
            self.get_cat_num_features()

        return missing_value_df

    def corr_plot(self, path):
        """
        plot correlation matrix of a dataframe.
        :return:
        """

        # Evaluate correlation matrix
        corr_matrix_plot = self.df.corr()

        # Write correlation matrix to a csv file.
        corr_matrix_plot.to_csv(self.config.get("correlation_matrix_path"), index=False)

        # Plot correlation matrix.
        f, ax = plt.subplots(figsize=(40, 32))

        # Diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with a color bar
        sns.heatmap(corr_matrix_plot, cmap=cmap, center=0, linewidths=.25, cbar_kws={"shrink": 0.6})

        # Set the ylabels
        ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
        ax.set_yticklabels(list(corr_matrix_plot.index), size=int(320 / corr_matrix_plot.shape[0]))

        # Set the xlabels
        ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
        ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(320 / corr_matrix_plot.shape[1]))

        # Set the title
        plt.title("Correlation Plot", size=14)

        # Save the figure
        plt.savefig(path)

    def remove_correlated_features(self, is_drop=True) -> pd.DataFrame:
        """
        get information regarding correlated features of a dataframe.
        :param: is_drop: if True, the method will drop the correlated features.
        :return: dataframe with correlated features information.
        """
        corr_matrix = self.df.corr()

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > self.correlation_threshold)]

        # Dataframe to hold correlated pairs
        collinear_df = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > self.correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > self.correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            collinear_df = pd.concat([collinear_df, temp_df], ignore_index=True)

        collinear_df.to_csv(self.config.get("correlated_features_path"), index=False)
        to_drop = collinear_df['drop_feature'].tolist()

        if is_drop and len(to_drop) > 0:
            self.df.drop(to_drop, axis=1, inplace=True)
            self.get_cat_num_features()

        return collinear_df

    def remove_almost_zero_numerical_features(self, is_drop=True) -> pd.DataFrame:
        """
        get information regarding almost zero numerical features of a dataframe.
        :param: is_drop: if True, the method will drop the columns enjoying more than zero_pct_threshold zero values.
        :return: dataframe with almost zero numerical features information.
        """
        self.get_cat_num_features()

        # Get the dataframe including each column and the percentage of zero values in it.
        zero_pct_df = self.df[self.numerical_cols].apply(lambda x: (x == 0).sum() / len(x)).reset_index().rename({"index": "column_name", 0: 'zero_pct'}, axis=1)

        # Write the dataframe to a csv file.
        zero_pct_df.to_csv(self.config.get("zero_pct_features_path"), index=False)

        # Get the column names of the columns with more than zero_pct_threshold zero values.
        to_drop = zero_pct_df[zero_pct_df['zero_pct'] >= self.zero_pct_threshold]['column_name'].tolist()

        # Drop the columns with more than zero_pct_threshold zero values if is_drop is True.
        if is_drop:
            self.df.drop(to_drop, axis=1, inplace=True)
            self.get_cat_num_features()

        return zero_pct_df

    def remove_highly_variable_categorical_features(self, is_drop=True):
        """
        get information regarding categorical features in a dataframe which are enjoying high sub-categories.
        :param is_drop: if True, the method will drop the categorical features with high number of sub-categories.
        :return:
        """
        self.get_cat_num_features()
        print("-----------------------------------------------------")
        print(self.categorical_cols)

        initial_num_categories_df = self.df[self.categorical_cols].apply(lambda col: len(col.unique())).reset_index().rename({"index": "column_name", 0: 'num_categories'}, axis=1)
        initial_num_categories_df.to_csv(self.config.get("initial_num_categories_features_path"), index=False)

        for cat_col in self.categorical_cols:
            if cat_col != "customer_id":

                print(cat_col)
                print("----------------------------------------------------")

                if len(self.df[cat_col].unique()) > self.num_categories_threshold:
                    # Replace the subcategories freq_pct of which is less than categorical_pct_threshold.
                    # Get the dataframe including the information regarding the frequency of each subcategory.
                    freq_df = self.df[cat_col].value_counts(normalize=True).reset_index().rename({"index": "category", cat_col: 'freq'}, axis=1)

                    # Keep the subcategories which are more than categorical_pct_threshold and convert them to  alist.
                    categories = freq_df[freq_df['freq'] >= self.categorical_pct_threshold]["category"].tolist()

                    print(freq_df)
                    print("----------------------------------------------------")

                    print(categories)
                    print("----------------------------------------------------")
                    print("----------------------------------------------------")

                    # If Categories is not empty, replace the subcategories with a specific sub-category whose freq_pct is the least value grater than categorical_pct_threshold.
                    if len(categories) > 0:
                        self.df.loc[~self.df[cat_col].isin(categories), cat_col] = freq_df["category"].tolist()[len(categories)]

                    # Otherwise, drop that column.
                    else:
                        if is_drop:
                            self.df.drop(cat_col, axis=1, inplace=True)

        self.get_cat_num_features()

        final_num_categories_df = self.df[self.categorical_cols].apply(lambda col: len(col.unique())).reset_index().rename({"index": "column_name", 0: 'num_categories'}, axis=1)
        final_num_categories_df.to_csv(self.config.get("final_num_categories_features_path"), index=False)

        return

    def get_eda_df(self):
        """
        apply all the eda methods to the dataframe
        :return: dataframe with all the eda methods applied on.
        """
        self.remove_missing_value()
        # Plot the initial dataframe correlation matrix.
        self.corr_plot(self.figures_config.get("initial_correlation_plot_path"))

        # Do EDA operations on the dataframe.
        self.remove_correlated_features()
        self.remove_almost_zero_numerical_features()
        self.remove_highly_variable_categorical_features()

        # Reorder the columns of fina dataframe..
        self.get_cat_num_features()
        self.df = self.df[self.categorical_cols + self.numerical_cols]

        # Plot the final dataframe correlation matrix.
        self.corr_plot(self.figures_config.get("final_correlation_plot_path"))

        # Write the final dataframe to a csv file.
        self.df.to_csv(self.config.get("eda_data_path"), index=False)


if __name__ == "__main__":
    eda = EDA(config_path="config_files/config.json",
              eda_config_path="config_files/eda_config.json",
              figures_config_path="config_files/figures_config.json")
    eda.get_eda_df()
