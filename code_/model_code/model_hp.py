import json
from functools import partial
from time import time
from typing import Callable

import catboost as cb
import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from code_.model_code.model_utils import create_dir, get_labeled_data


def lgb_objective(trial, df_final: pd.DataFrame) -> float:
    """
    :param: trail: object to define the hyperparameters.
    :param: df_final: dataframe with the features and the target.
    :return:
    """
    # Set the number of cross-validation split
    n_splits = 5

    # Set the random set for reproducibility.
    random_state = 1234

    # Create StratifiedKFold object.
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Get the categorical and numerical features.
    cat_features = [col for col in df_final.columns if df_final[col].dtype != 'float64' and col not in ['customer_id', 'is_returning_customer']]
    num_features = [col for col in df_final.columns if df_final[col].dtype == 'float64']

    # Initialize the metric.
    metrics = list()

    # Define the hyperparameters space.
    params = {
        "max_depth": 2 * trial.suggest_int("max_depth", 1, 8),
        "num_iterations": 100 * trial.suggest_int("num_iterations", 1, 10),
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "rf", "dart"]),
        "tree_learner": trial.suggest_categorical("tree_learner", ["serial", "feature", "data", "voting"]),
        "learning_rate": trial.suggest_loguniform('learning_rate', 5e-3, 5e-1),
        "num_leaves": 4 * trial.suggest_int('num_leaves', 1, 10),
        "bagging_fraction": trial.suggest_uniform('bagging_fraction', 5e-1, 1.0 - 1e-2),
        "feature_fraction": trial.suggest_uniform('feature_fraction', 5e-1, 1.0 - 1e-2),
    }

    if params.get("boosting_type") == "rf" and params.get("bagging_fraction") < 1:
        params["bagging_freq"] = 1

    # Iterate over the cross-validation splits.
    for i, [train_index, test_index] in enumerate(cv.split(df_final[cat_features + num_features], df_final["is_returning_customer"])):
        # Define the training and test sets.
        df_test = df_final.iloc[test_index, :]
        x_test = df_test[cat_features + num_features]
        y_test = df_test["is_returning_customer"] * 1

        df_train = df_final.iloc[train_index, :]
        x_train = df_train[cat_features + num_features]
        y_train = df_train["is_returning_customer"] * 1

        # Define the training and evaluation sets.
        x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.1, random_state=101, stratify=y_train)

        # Define the lightbgm model.
        model = lgb.LGBMClassifier(objective='binary', is_unbalance=True, seed=1234, **params)

        # Fit the model.
        model.fit(x_train, y_train, eval_set=(x_eval, y_eval), feature_name='auto', categorical_feature="auto",
                  callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)])

        # Get the predictions for the test set.
        y_pred = model.predict(x_test)

        # Get the AUC score for the current fold.
        auc_score = round(roc_auc_score(y_test, y_pred), 4)

        # Add the current AUC score to the list.
        metrics.append(auc_score)

        print("-------------------------------------------------------")
        print(params)
        print("-------------------------------------------------------")

        # Get the mean AUC score.
        score = sum(metrics) / len(metrics)
        print(score)

    return score


def cb_objective(trial, df_final: pd.DataFrame, gpu_enabled: bool = False):
    """
    :param: trail: object to define the hyperparameters.
    :param: df_final: dataframe with the features and the target.
    :param: gpu_enabled: boolean to enable the GPU.
    :return:
    """
    # Set the number of cross-validation split.
    n_splits = 5

    # Set the random set for reproducibility.
    random_state = 1234

    # Create StratifiedKFold object.
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    # Get the categorical and numerical features.
    cat_features = [col for col in df_final.columns if df_final[col].dtype != 'float64' and col not in ['customer_id', 'is_returning_customer']]
    categorical_features_indices = list(range(0, len(cat_features)))
    num_features = [col for col in df_final.columns if df_final[col].dtype == 'float64']

    # Initialize the metric.
    metrics = list()

    # Define the hyperparameters space.
    params = {
        "depth": 2 * trial.suggest_int("depth", 1, 8),
        "iterations": 100 * trial.suggest_int("iterations", 1, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "learning_rate": trial.suggest_loguniform('learning_rate', 5e-3, 1e-1),
        "od_type": trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
        "used_ram_limit": "20gb",
    }

    if gpu_enabled:
        params["task_type"] = "GPU"
        params["bootstrap_type"] = "Poisson"

    else:
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.01, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.8, 1)

    # Iterate over the cross-validation splits.
    for i, [train_index, test_index] in enumerate(cv.split(df_final[cat_features + num_features], df_final["is_returning_customer"])):
        # Define the training and test sets.
        df_test = df_final.iloc[test_index, :]
        x_test = df_test[cat_features + num_features]
        y_test = df_test["is_returning_customer"] * 1

        df_train = df_final.iloc[train_index, :]
        x_train = df_train[cat_features + num_features]
        y_train = df_train["is_returning_customer"] * 1

        # Define the training and evaluation sets.
        x_train, x_eval, y_train, y_eval = train_test_split(x_train, y_train, test_size=0.1, random_state=101, stratify=y_train)

        # Define the catboost model.
        model = cb.CatBoostClassifier(auto_class_weights="Balanced",
                                      loss_function='Logloss',
                                      eval_metric='AUC:hints=skip_train~false',
                                      metric_period=100,
                                      use_best_model=True,
                                      **params)
        # Fit the model.
        model.fit(x_train, y_train, cat_features=categorical_features_indices,
                  eval_set=cb.Pool(data=x_eval, label=y_eval, cat_features=categorical_features_indices), verbose=True, plot=True)

        # Get the predictions for the test set.
        y_pred = model.predict(x_test)

        # Get the AUC score for the current fold.
        auc_score = round(roc_auc_score(y_test, y_pred), 4)

        # Add the current AUC score to the list.
        metrics.append(auc_score)

    print("-------------------------------------------------------")
    print(params)
    print("-------------------------------------------------------")

    # Get the mean AUC score.
    score = sum(metrics) / len(metrics)
    print(score)

    return score

