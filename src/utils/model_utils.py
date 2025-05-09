import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from mlflow.models.signature import infer_signature

models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {"fit_intercept": [True, False], "positive": [False, True]}
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {"n_estimators": [50], "max_depth": [None, 10]}
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(),
        "params": {"n_estimators": [50], "learning_rate": [0.1], "max_depth": [3]}
    }
}

def train_and_evaluate_model(name, model_def, X_train, y_train, X_val, y_val):
    with mlflow.start_run(run_name=name, nested=True):
        grid = GridSearchCV(model_def["model"], model_def["params"], cv=3, n_jobs=-1, scoring="r2")
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({"val_r2": r2, "val_mse": mse})
        mlflow.sklearn.log_model(best_model, "model")

        return {
            "model_name": name,
            "model": best_model,
            "r2": r2,
            "mse": mse,
            "run_id": mlflow.active_run().info.run_id
        }

def train_and_evaluate_all_models(X_train, y_train, X_val, y_val, X_train_red, X_val_red):
    results = []
    for name, info in models.items():
        results.append(train_and_evaluate_model(name, info, X_train, y_train, X_val, y_val))
        results.append(train_and_evaluate_model(name + "_reduced", info, X_train_red, y_train, X_val_red, y_val))
    return sorted(results, key=lambda x: x['r2'], reverse=True)

def register_best_model(results, X_test, X_test_red, y_test):
    best = results[0]
    best_model = best["model"]
    X_test_final = X_test_red if "reduced" in best["model_name"] else X_test
    y_pred = best_model.predict(X_test_final)

    signature = infer_signature(X_test_final, y_pred)
    input_example = X_test_final.iloc[:1]

    mlflow.set_experiment("Ca_Housing_Prediction")
    model_name = f"california_housing_{best['model_name']}"

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=model_name
    )


    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[-1].version
    client.transition_model_version_stage(model_name, latest_version, stage="Production", archive_existing_versions=True)

    return {
        "model_name": model_name,
        "mlflow_uri": f"models:/{model_name}/Production",
        "test_r2": r2_score(y_test, y_pred),
        "test_mse": mean_squared_error(y_test, y_pred)
    }
