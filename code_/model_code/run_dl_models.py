from time import time

import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import NodeConfig, TabNetModelConfig, CategoryEmbeddingModelConfig
from pytorch_tabular.utils import get_class_weighted_cross_entropy
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from code_.model_code.model_utils import get_labeled_data, save_to_csv, get_feature_importance_avg


def dl_training(config_path: str, model_type: str, number_of_folds: int = 5, max_epochs: int = 100, batch_size: int = 1024):
    """
    This function trains a deep learning model.
    :param: config_path: path to the config file
    :param model_type: name of the model to train
    :param number_of_folds: number of folds to use for cross-validation
    :param max_epochs: maximum number of epochs to train the model
    :param batch_size: batch size to use for training
    :return:
    """

    if model_type in ["tabnet", "node", "category_embedding"]:
        g_model_type = "tabular"
    else:
        raise ValueError(f"Model type: {model_type} is not supported")

    # Get the required dataframe.
    df_final = get_labeled_data(config_path, g_model_type)

    # Drop the columns that are not required.
    df_final.drop(columns=['customer_id'], inplace=True)
    print(df_final.columns)

    # Set the number of folds for the cross-validation.
    n_splits = number_of_folds

    # Set the random state for the reproducibility.
    random_state = 1234

    # Get the StratifiedKFold object.
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Get the categorical and numerical features.
    cat_features = [col for col in df_final.columns if df_final[col].dtype != 'float64' and col not in ['customer_id', 'is_returning_customer']]
    num_features = [col for col in df_final.columns if df_final[col].dtype == 'float64']

    # Set the model config based on the model type.
    if model_type == 'node':
        model_config = NodeConfig(task="classification", num_layers=4, num_trees=256,
                                  depth=5, embed_categorical=True, learning_rate=1e-3,
                                  choice_function="entmax15", bin_function="entmoid15",
                                  metrics=["f1", "accuracy"], metrics_params=[{"num_classes": 2}, {}])
    elif model_type == "tabnet":
        model_config = TabNetModelConfig(task="classification", n_d=2, n_a=2,
                                         n_steps=3, n_independent=2, n_shared=2,
                                         metrics=["f1", "accuracy"], metrics_params=[{"num_classes": 2}, {}])

    elif model_type == "category_embedding":
        model_config = CategoryEmbeddingModelConfig(task="classification", layers="1024-256-64", activation="LeakyReLU", learning_rate=1e-3)

    else:
        raise ValueError(f"Model type: {model_type} is not supported")

    # Set the data config.
    data_config = DataConfig(target=['is_returning_customer'], continuous_cols=num_features, categorical_cols=cat_features, )

    # Set the trainer config.
    trainer_config = TrainerConfig(auto_lr_find=False, batch_size=batch_size, max_epochs=max_epochs, auto_select_gpus=False)

    # Set the optimizer config.
    optimizer_config = OptimizerConfig()

    # Create the list to save scores.
    metrics = list()

    # Create the list to save the models.
    models = list()

    # Iterate over the folds.
    for i, [train_index, test_index] in enumerate(cv.split(df_final[cat_features + num_features], df_final["is_returning_customer"])):
        # Get the train and test data.
        df_test = df_final.iloc[test_index, :]
        df_train = df_final.iloc[train_index, :]
        y_test = df_test["is_returning_customer"] * 1

        # Create the tabular dl model
        model = TabularModel(data_config=data_config, model_config=model_config, optimizer_config=optimizer_config, trainer_config=trainer_config)

        # Start running the training.
        start = time()

        # Define the weighted loss to handle imbalanced dataset.
        weighted_loss = get_class_weighted_cross_entropy(df_train["is_returning_customer"].values.ravel(), mu=0.1)

        # Fit the model.
        model.fit(train=df_train, loss=weighted_loss)

        elapsed_time = time() - start
        print(f"Training time: {elapsed_time}")

        # Get the predictions.
        y_pred = model.predict(df_test)

        # Assert prediction column is in the dataframe.
        assert "prediction" in y_pred.columns, "prediction column not found"

        # Summarize the fit of the model
        report = pd.DataFrame(classification_report(y_test, y_pred["prediction"], output_dict=True)).transpose().reset_index()
        print(report)

        # Save the report to a csv file.
        report_save_path = f"model_files/{model_type}/csv_files/fold_{i}_report.csv"
        save_to_csv(report, report_save_path)

        # Get the AUC score for the fold.
        auc_score = roc_auc_score(y_test, y_pred["prediction"])
        print(auc_score)

        # Add the model of the fold to the list.
        models.append(model)

        # Add the AUC score of the fold to the list.
        metrics.append(auc_score)

    # Save the AUC scores to a csv file.
    auc_df = pd.DataFrame([(i, x) for i, x in enumerate(metrics)], columns=["fold", "auc_score"])
    auc_df.to_csv(f"model_files/{model_type}/csv_files/auc_scores.csv")

    return models, metrics


if __name__ == "__main__":
    dl_training("config_files/config.json", "node", 5, 20, 1024)
    get_feature_importance_avg("model_files", "node")

    dl_training("config_files/config.json", "tabnet", 5, 20, 1024)
    get_feature_importance_avg("model_files", "tabnet")

    dl_training("config_files/config.json", "category_embedding", 5, 20, 1024)
    get_feature_importance_avg("model_files", "category_embedding")
