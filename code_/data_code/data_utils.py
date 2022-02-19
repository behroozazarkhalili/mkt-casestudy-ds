import pandas as pd


def get_csv_gz(df_path: str) -> pd.DataFrame:
    """read a csv.gz file
    :param: df_path: path to csv.gz file
    :return: dataframe
    """
    df = pd.read_csv(df_path, compression='gzip')
    return df


def get_grouped_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """group dataframe rows into list in Pandas Groupby
    :param: df: dataframe
    :param: col: column to group by
    :return: grouped dataframe
    >>> get_grouped_df(pd.DataFrame({'a': [1, 1, 2, 2, 3, 3], 'b': [1, 2, 3, 4, 5, 6]}))
    pd.DataFrame({'a':[1,2,3],'b':[[1,2],[3,4],[5,6]]})
    """
    assert col in df.columns, f"{col} not in dataframe"
    grouped_df = df.groupby(col).agg(lambda x: list(x)).reset_index()
    return grouped_df


def convert_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """convert a string column to a date column with yyyy-mm-dd format"""
    assert col in df.columns, f"{col} not in dataframe"
    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")
    return df


def get_diff_between_dates_in_list(x: list) -> list:
    """ get the difference between consecutive dates
    :param: x: list of dates
    :return: list of differences
    >>> get_diff_between_dates_in_list([pd.to_datetime('2019-01-01'), pd.to_datetime('2019-01-02'), pd.to_datetime('2019-01-03')])
    [1, 1]
    """
    diff = [(x[i + 1] - x[i]).days for i in range(len(x) - 1)]
    return diff


def get_number_return(x: list, interval: int) -> int:
    """get the number of return in an interval for a list of dates
    :param: x: list of dates
    :param: interval: interval in days
    :return: number of return in interval
    >>> get_number_return([pd.to_datetime('2019-01-01'), pd.to_datetime('2019-01-02'), pd.to_datetime('2019-01-04')], 2)
    1
    """
    diff = get_diff_between_dates_in_list(x)
    return len([i for i in diff if i <= interval])


def get_list_statistics(x: list) -> tuple:
    """get the statistics of a list
    :param: x: list
    :return: tuple of statistics
    >>> get_list_statistics([1, 2, 3, 4, 5])
    (1, 5, 15, 3.0)
    """
    return min(x), max(x), sum(x), round(sum(x) / len(x), 4)


def get_frequency_info(x: list) -> tuple:
    """get the most frequent item and its frequency in a list
    :param: x: list
    :return: tuple of most frequent item and its frequency, and length of unique items.
    >>> get_frequency_info([1.1, 2.2, 3.5, 4, 5, 1.1, 0.2, 3.24, 0.4, 1.1, 5.5])
    (1.1, 3, 9)
    """
    most_freq_item = max(set(x), key=x.count)
    return most_freq_item, x.count(most_freq_item), len(set(x))


def convert_dtypes(df: pd.DataFrame, columns_dtypes: dict) -> pd.DataFrame:
    """convert dtypes of columns in a dataframe
    :param: df: dataframe
    :param: columns_dtypes: dictionary of columns and dtypes
    :return: dataframe with converted dtypes
    >>> convert_dtypes(pd.DataFrame({'a': [1, 2, 3], 'b': [1.1, 2.2, 3.3]}), {'a': 'int64', 'b': 'float64'})
    """
    for col, dtype in columns_dtypes.items():
        assert col in df.columns, f"{col} not in dataframe"
        df[col] = df[col].astype(dtype)
    return df


def get_amount_by_special_id(available_special_ids, special_ids_list, amount_paid_list):
    """get the amount paid by each payment id
    :param: available_special_ids: list of available special ids
    :param: special_ids_list: list of special ids
    :param: amount_paid_list: list of amounts paid
    :return: dictionary of special ids and amounts paid by them
    >>> get_amount_by_special_id([1, 2, 3, 4], [1, 2, 3, 1, 2, 3], [1.1, 2.2, 3.3, 2.1, 3.2, 4.3])
    (3.2, 5.4, 7.6, 0, 0)
    """
    amount_by_special_id = {}
    for special_id in available_special_ids:
        amount_by_special_id[special_id] = sum([amount for special_id_, amount in zip(special_ids_list, amount_paid_list) if special_id_ == special_id])
    return tuple(amount_by_special_id.values())
