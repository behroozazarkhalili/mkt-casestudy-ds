from time import time

import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models import NodeConfig, TabNetModelConfig, CategoryEmbeddingModelConfig
from pytorch_tabular.utils import get_class_weighted_cross_entropy
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from code_.model_code.model_utils import get_labeled_data, save_to_csv


def dl_training(config_path: str, model_type: str, number_of_folds: int = 5, max_epochs: int = 100, batch_size: int = 1024):
    """
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

    df_final = get_labeled_data(config_path, g_model_type)

    df_final.drop(columns=['customer_id'], inplace=True)
    print(df_final.columns)

    n_splits = number_of_folds
    random_state = 1234
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    cat_features = [col for col in df_final.columns if df_final[col].dtype != 'float64' and col not in ['customer_id', 'is_returning_customer']]
    num_features = [col for col in df_final.columns if df_final[col].dtype == 'float64']

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

    data_config = DataConfig(target=['is_returning_customer'], continuous_cols=num_features, categorical_cols=cat_features, )
    trainer_config = TrainerConfig(auto_lr_find=False, batch_size=batch_size, max_epochs=max_epochs, auto_select_gpus=False)
    optimizer_config = OptimizerConfig()

    metrics = list()
    models = []

    for i, [train_index, test_index] in enumerate(cv.split(df_final[cat_features + num_features], df_final["is_returning_customer"])):
        df_test = df_final.iloc[test_index, :]
        df_train = df_final.iloc[train_index, :]
        y_test = df_test["is_returning_customer"] * 1

        model = TabularModel(data_config=data_config, model_config=model_config, optimizer_config=optimizer_config, trainer_config=trainer_config)

        start = time()
        weighted_loss = get_class_weighted_cross_entropy(df_train["is_returning_customer"].values.ravel(), mu=0.1)
        model.fit(train=df_train, loss=weighted_loss)
        elapsed_time = time() - start
        print(f"Training time: {elapsed_time}")

        y_pred = model.predict(df_test)

        # Assert prediction column is in the dataframe.
        assert "prediction" in y_pred.columns, "prediction column not found"

        # Summarize the fit of the model
        report = pd.DataFrame(classification_report(y_test, y_pred["prediction"], output_dict=True)).transpose()
        print(report)

        # Save the report to a csv file.
        report_save_path = f"model_files/{model_type}/csv_files/fold_{i}_report.csv"
        save_to_csv(report, report_save_path)

        auc_score = roc_auc_score(y_test, y_pred["prediction"])
        print(auc_score)

        models.append(model)
        metrics.append(auc_score)

    auc_df = pd.DataFrame([(i, x) for i, x in enumerate(metrics)], columns=["fold", "auc_score"])
    auc_df.to_csv(f"model_files/{model_type}/csv_files/auc_scores.csv")

    return models, metrics


if __name__ == "__main__":
    dl_training("config_files/config.json", "node", 5, 2, 1024)
    dl_training("config_files/config.json", "category_embedding", 5, 2, 1024)
    dl_training("config_files/config.json", "tabnet", 5, 2, 1024)
