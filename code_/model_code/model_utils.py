import json
import os
import pandas as pd
from code_.data_code.data_utils import get_csv_gz


def create_dir(file_name: str):
    """

    :param: file_name: The file name the directory of which should be created.
    :return:
    """
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)


def save_to_csv(report: pd.DataFrame, report_save_path: str):
    """

    :param: report: Pandas dataframe of the report.
    :param: report_save_path: The path where the report should be saved.
    :return:
    """
    create_dir(report_save_path)
    report.to_csv(report_save_path, index=False)


def get_labeled_data(config_path: str, model_type: str) -> pd.DataFrame:
    """
    Reads in the dataframes  and joined them together to get the labeled data.
    :param: config_path: path to the config file
    :return: dataframe with the data
    """
    # Read the config file
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)
    print("Read successful")

    # Read dataframes.
    eda_df = pd.read_csv(config.get("eda_data_path"))
    labeled_df = get_csv_gz(config.get("labeled_data_path"))

    # Join the dataframes to get the final training dataframes.
    df_final = pd.merge(eda_df, labeled_df, on="customer_id", how="inner")
    print("Final dataframe created successfully")

    if model_type == "lightgbm":
        for col in eda_df.columns:
            col_type = df_final[col].dtype.name
            if col_type in ["category", "object", "int64", "int32"]:
                df_final[col] = df_final[col].astype('category')

    elif model_type in ["catboost", "tabular"]:
        for col in eda_df.columns:
            col_type = df_final[col].dtype.name
            if col_type in ["category", "object", "int64", "int32"]:
                df_final[col] = df_final[col].astype('object')

    else:
        raise ValueError(f"Model type: {model_type} is not supported")

    return df_final
