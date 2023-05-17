import pandas as pd
from catboost import CatBoostRegressor


def train_catboost_diff(x: pd.DataFrame, target_name: str, end_train: str):
    x = x[(x["timestamp"] > "2020-11-02") & (x[target_name] != 0)]

    feat_names = list(x.columns)
    feat_names.remove("gufi")
    feat_names.remove("timestamp")
    feat_names.remove(target_name)

    cat_features = [i for i in range(len(feat_names)) if "_cat_" in feat_names[i]]

    x_train = x[x["timestamp"] < end_train][feat_names]
    x_val = x[x["timestamp"] >= end_train][feat_names]

    y_train = (
        x[x["timestamp"] < end_train][target_name] - x_train["etd_time_till_est_dep"]
    )
    y_val = x[x["timestamp"] >= end_train][target_name] - x_val["etd_time_till_est_dep"]

    print("#" * 20 + " " * 5 + "training with ", x_train.shape, " " * 5 + "#" * 20)
    print("#" * 20 + " " * 5 + "validating with ", x_val.shape, " " * 5 + "#" * 20)

    best_score = 1e9
    best_model = None

    for depth in [7]:
        for eta in [0.01]:

            model = CatBoostRegressor(
                eta=eta,
                depth=depth,
                rsm=1,
                subsample=0.8,
                max_leaves=21,
                l2_leaf_reg=3,
                min_data_in_leaf=5000,
                n_estimators=20000,
                task_type="CPU",
                thread_count=-1,
                grow_policy="Lossguide",
                has_time=True,
                random_seed=4,
                loss_function="MAE",
                boosting_type="Plain",
                max_ctr_complexity=12,
                bootstrap_type="Bernoulli",
            )

            model.fit(
                x_train,
                y_train,
                eval_set=(x_val, y_val),
                use_best_model=True,
                verbose=200,
                cat_features=cat_features,
                early_stopping_rounds=60,
            )

            score = abs(
                pd.Series(model.predict(x_val)).astype(int).clip(1, 299).values
                - y_val.values
            ).mean()
            print("Error in test:", score)

            if score < best_score:
                print("BEST SCORE", score)
                best_score = score
                best_model = model

    # Retrieve feature contributions
    feat_imps = pd.DataFrame(
        {"features": feat_names, "imp": best_model.feature_importances_}
    )
    feat_imps["imp"] = feat_imps["imp"] / feat_imps["imp"].sum()
    feat_imps.sort_values("imp", ascending=False, inplace=True)

    return best_model, best_model, feat_imps


def train_catboost(x: pd.DataFrame, target_name: str, end_train: str):
    x = x[(x["timestamp"] > "2020-11-02") & (x[target_name] != 0)]

    feat_names = list(x.columns)
    feat_names.remove("gufi")
    feat_names.remove("timestamp")
    feat_names.remove(target_name)

    cat_features = [i for i in range(len(feat_names)) if "_cat_" in feat_names[i]]

    x_train = x[x["timestamp"] < end_train][feat_names]
    x_val = x[x["timestamp"] >= end_train][feat_names]

    y_train = x[x["timestamp"] < end_train][target_name]
    y_val = x[x["timestamp"] >= end_train][target_name]

    print("#" * 20 + " " * 5 + "training with ", x_train.shape, " " * 5 + "#" * 20)
    print("#" * 20 + " " * 5 + "validating with ", x_val.shape, " " * 5 + "#" * 20)

    best_score = 1e9
    best_model = None

    for depth in [7]:
        for eta in [0.01]:

            model = CatBoostRegressor(
                eta=eta,
                depth=depth,
                rsm=1,
                subsample=0.8,
                max_leaves=21,
                l2_leaf_reg=3,
                min_data_in_leaf=5000,
                n_estimators=20000,
                task_type="CPU",
                thread_count=-1,
                grow_policy="Lossguide",
                has_time=True,
                random_seed=4,
                loss_function="MAE",
                boosting_type="Plain",
                max_ctr_complexity=12,
                bootstrap_type="Bernoulli",
            )

            model.fit(
                x_train,
                y_train,
                eval_set=(x_val, y_val),
                use_best_model=True,
                verbose=200,
                cat_features=cat_features,
                early_stopping_rounds=60,
            )

            score = abs(
                pd.Series(model.predict(x_val)).astype(int).clip(1, 299).values
                - y_val.values
            ).mean()
            print("Error in test:", score)

            if score < best_score:
                print("BEST SCORE", score)
                best_score = score
                best_model = model

    return best_model
