import json
import math

from code_.data.data_utils import get_csv_gz, get_grouped_df, get_number_return, convert_to_datetime, get_list_statistics, get_frequency_info, get_amount_by_special_id


def get_feature_engineered_data(config_path: str):
    # Get config file.
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)
    print("Read successful")

    # Read the data
    df = get_csv_gz(config.get("unlabeled_data_path"))
    # Convert order_date to datetime format.
    df = convert_to_datetime(df, "order_date")

    # Grouped the data by "customer_id", and get list-column of the grouped dataframes.
    g_df = get_grouped_df(df, col="customer_id")

    # Get the number of return of each customer within six months.
    g_df["number_return"] = g_df.apply(lambda row: get_number_return(row["order_date"], 6 * 30), axis=1)

    # Get the average order hour and trigonometric functions of it for each customer.
    g_df["avg_order_hour"] = g_df.apply(lambda row: sum(row["order_hour"]) / len(row["order_hour"]), axis=1)
    g_df["sin_order_hour"] = g_df.apply(lambda row: round(math.sin(2 * math.pi * row["avg_order_hour"] / 24), 3), axis=1)
    g_df["cos_order_hour"] = g_df.apply(lambda row: round(math.cos(2 * math.pi * row["avg_order_hour"] / 24), 3), axis=1)

    # Get the number of order for each customer.
    g_df["number_order"] = g_df.apply(lambda row: len(row["order_date"]), axis=1)

    # Get the percentage of failure for each order of each customer.
    g_df["failure_percentage"] = g_df.apply(lambda row: sum(row["is_failed"]) / len(row["is_failed"]), axis=1)

    # Get the statistics of delivery_fee for each customer.
    g_df["min_delivery_fee"], g_df["max_delivery_fee"], g_df["total_delivery_fee"], g_df["avg_delivery_fee"] = zip(*g_df["delivery_fee"].map(get_list_statistics))

    # Get the statistics of amount_paid for each customer.
    g_df["min_amount_paid"], g_df["max_amount_paid"], g_df["total_amount_paid"], g_df["avg_amount_paid"] = zip(*g_df["amount_paid"].map(get_list_statistics))

    # Get the statistics of voucher_amount for each customer.
    g_df["min_voucher_amount"], g_df["max_voucher_amount"], g_df["total_voucher_amount"], g_df["avg_voucher_amount"] = zip(*g_df["voucher_amount"].map(get_list_statistics))

    # Get the most frequent restaurant, the number of ordering from this restaurant, and the number of unique restaurants each customer ordered.
    g_df["most_frequent_restaurant"], g_df["restaurant_frequency"], g_df["number_unique_restaurant"] = zip(*g_df["restaurant_id"].map(get_frequency_info))

    # Get the most frequent city, the number of ordering from this city, and the number of unique cities each customer ordered.
    g_df["most_frequent_city"], g_df["city_frequency"], g_df["number_unique_city"] = zip(*g_df["city_id"].map(get_frequency_info))

    # Get the most frequent payment method, the number of ordering from this payment method, and the number of unique payment methods each customer ordered.
    g_df["most_frequent_payment"], g_df["payment_frequency"], g_df["number_unique_payment"] = zip(*g_df["payment_id"].map(get_frequency_info))

    # Get the most frequent transmission method, the number of ordering from this transmission method, and the number of unique transmission methods each customer ordered.
    g_df["most_frequent_transmission"], g_df["transmission_frequency"], g_df["number_unique_transmission"] = zip(*g_df["transmission_id"].map(get_frequency_info))

    # Get the most frequent platform, the number of ordering from this platform, and the number of unique platforms each customer ordered.
    g_df["most_frequent_platform"], g_df["platform_frequency"], g_df["number_unique_platform"] = zip(*g_df["platform_id"].map(get_frequency_info))

    # Get the amount_paid by each payment_id and each customer.
    payment_ids = sorted(df["payment_id"].unique())
    payment_text = "amount_payment_"
    pids_columns = list(zip(*g_df[["amount_paid", "payment_id"]].apply(lambda row: get_amount_by_special_id(payment_ids, row["payment_id"], row["amount_paid"]), axis=1)))

    for i in range(len(pids_columns)):
        g_df[payment_text + str(payment_ids[i])] = pids_columns[i]

    # Get the amount_paid by each transmission_id and each customer.
    transmission_ids = sorted(df["transmission_id"].unique())
    transmission_text = "amount_transmission_"
    tids_columns = list(
        zip(*g_df[["amount_paid", "transmission_id"]].apply(lambda row: get_amount_by_special_id(transmission_ids, row["transmission_id"], row["amount_paid"]), axis=1)))

    for i in range(len(tids_columns)):
        g_df[transmission_text + str(transmission_ids[i])] = tids_columns[i]

    # Get the amount_paid by each platform_id and each customer.
    platform_ids = sorted(df["platform_id"].unique())
    platform_text = "amount_platform_"
    pids_columns = list(zip(*g_df[["amount_paid", "platform_id"]].apply(lambda row: get_amount_by_special_id(platform_ids, row["platform_id"], row["amount_paid"]), axis=1)))

    for i in range(len(pids_columns)):
        g_df[platform_text + str(platform_ids[i])] = pids_columns[i]

    # Save the feature-engineered dataframe to a csv file.
    df_columns = df.columns[1:]
    g_df_columns = g_df.columns
    fe_df = g_df[[column for column in g_df_columns if column not in df_columns]]
    fe_df.to_csv(config.get("feature_engineered_data_path"), index=False)

    return fe_df


# Run the feature engineering process.
if __name__ == "__main__":
    fe_df = get_feature_engineered_data("config_files/config.json")
    print(fe_df.head())
