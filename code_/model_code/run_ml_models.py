import json
from time import time
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from code_.model_code.model_utils import save_to_csv, get_labeled_data, get_feature_importance_avg


def training(config_path: str, hp_config_path: str, model_type: str, number_of_folds: int = 5):
    """
    This function trains the model and returns the models, metrics, and feature importances.
    :param: config_path: The path to the config file.
    :param: hp_config_path: The path to the hyperparameter configuration file.
    :param: model_type: The type of model to train.
    :param: number_of_folds: The number of folds to use for cross-validation.
    :return:
    """

    # Get the required dataframe.
    df_final = get_labeled_data(config_path, model_type)

    # Read the hp_config file
    with open(hp_config_path, "r") as jsonfile:
        hp_config = json.load(jsonfile)
    print("Read hp_config file successfully")

    best_params_path = hp_config.get(model_type).get("best_params_path")

    # Set the number of folds for the cross-validation.
    n_splits = number_of_folds

    # Set the random seed for reproducibility.
    random_state = 1234

    # Create the StratifiedKFold object.
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Get the categorical and numerical features.
    cat_features = [col for col in df_final.columns if df_final[col].dtype != 'float64' and col not in ['customer_id', 'is_returning_customer']]
    num_features = [col for col in df_final.columns if df_final[col].dtype == 'float64']

    # Create the list to save scores
    metrics = list()

    # Create the list to save the models.
    models = list()

    # Create the list to save the features  importance.
    feature_importances = []

    # Get the hyperparameters.
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    # Iterate over the folds.
    for i, [train_index, test_index] in enumerate(cv.split(df_final[cat_features + num_features], df_final["is_returning_customer"])):
        print(i)
        # Get the training and test data for the fold.
        df_test = df_final.iloc[test_index, :]
        x_test = df_test[cat_features + num_features]
        y_test = df_test["is_returning_customer"] * 1

        df_train = df_final.iloc[train_index, :]
        x_train = df_train[cat_features + num_features]
        y_train = df_train["is_returning_customer"] * 1

        # Get the training and evaluation data for the fold.
        x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.1, random_state=101, stratify=y_train)

        # Start running the training.
        start = time()

        if model_type == "lightgbm":
            # Define the lightgbm model.
            model = lgb.LGBMClassifier(objective='binary',
                                       is_unbalance=True,
                                       seed=1234,
                                       max_depth=2 * best_params.get("max_depth"),
                                       num_iterations=100 * best_params.get("num_iterations"),
                                       boosting_type=best_params.get("boosting_type"),
                                       tree_learner=best_params.get("tree_learner"),
                                       learning_rate=best_params.get("learning_rate"),
                                       num_leaves=4 * best_params.get("num_leaves"),
                                       bagging_fraction=best_params.get("bagging_fraction"),
                                       feature_fraction=best_params.get("feature_fraction")
                                       )
            # Fit the model.
            model.fit(x_train, y_train, eval_set=(x_eval, y_eval), feature_name='auto', categorical_feature="auto",
                      callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)])

        elif model_type == "catboost":
            # Define the catboost model.
            model = cb.CatBoostClassifier(auto_class_weights="Balanced",
                                          loss_function='Logloss',
                                          eval_metric='AUC:hints=skip_train~false',
                                          metric_period=100,
                                          use_best_model=True,
                                          random_seed=1234,
                                          depth=2 * best_params.get("depth"),
                                          iterations=100 * best_params.get("iterations"),
                                          l2_leaf_reg=best_params.get("l2_leaf_reg"),
                                          bootstrap_type=best_params.get("bootstrap_type"),
                                          learning_rate=best_params.get("learning_rate"),
                                          od_type=best_params.get("od_type"),
                                          bagging_temperature=best_params.get("bagging_temperature"),
                                          used_ram_limit=best_params.get("used_ram_limit")
                                          )

            # Define the index of categorical features.
            categorical_features_indices = list(range(0, len(cat_features)))
            # Fit the model.
            model.fit(x_train, y_train, cat_features=categorical_features_indices,
                      eval_set=cb.Pool(data=x_eval, label=y_eval, cat_features=categorical_features_indices), verbose=True, plot=True)

        else:
            raise ValueError(f"The model type: {model_type} is not supported.")

        elapsed_time = time() - start
        print('elapse time is about:, ', elapsed_time)

        # Get the predictions.
        y_pred = model.predict(x_test)

        # Summarize the fit of the model
        report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().reset_index()
        print(report)

        # Save the report to a csv file.
        report_save_path = f"model_files/{model_type}/csv_files/fold_{i}_report.csv"
        save_to_csv(report, report_save_path)

        # Print the confusion matrix.
        print(confusion_matrix(y_test, y_pred))

        # Get the AUC score for the fold.
        auc_score = round(roc_auc_score(y_test, y_pred), 4)

        # Get the feature importances for each fold.
        feature_importance = np.round(model.feature_importances_ / sum(model.feature_importances_), 4)
        if model_type == "lightgbm":

            fi = pd.DataFrame([model.feature_name_, feature_importance.tolist()]).transpose()

        elif model_type == "catboost":
            fi = pd.DataFrame([model.feature_names_, feature_importance.tolist()]).transpose()

        else:
            raise ValueError(f"The model type: {model_type} is not supported.")

        fi.columns = ["Features", "Scores"]
        fi.sort_values(by="Scores", ascending=False, inplace=True)

        # Save the feature importances to a csv file.
        feature_importance_save_path = f"model_files/{model_type}/csv_files/fold_{i}_feature_importance.csv"
        save_to_csv(fi, feature_importance_save_path)

        # Add the model of the fold to the list.
        models.append(model)

        # Add the AUC score of the fold to the list.
        metrics.append(auc_score)

        # Add the feature importances of the fold to the list.
        feature_importances.append(feature_importance)

    # Save the AUC score to a csv file.
    auc_df = pd.DataFrame([(i, x) for i, x in enumerate(metrics)], columns=["fold", "auc_score"])
    auc_df.to_csv(f"model_files/{model_type}/csv_files/auc_scores.csv")

    return models, metrics, feature_importances


if __name__ == "__main__":
    training("config_files/config.json", "config_files/hp_config.json", "lightgbm", 5)
    get_feature_importance_avg("model_files", "lightgbm")

    training("config_files/config.json", "config_files/hp_config.json", "catboost", 5)
    get_feature_importance_avg("model_files", "catboost")
