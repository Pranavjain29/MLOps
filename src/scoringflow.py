# src/scoringflow.py

from metaflow import FlowSpec, step
import pandas as pd
import mlflow.pyfunc

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        self.X_new = data.frame.drop("MedHouseVal", axis=1).sample(20, random_state=42)
        self.next(self.load_model)

    @step
    def load_model(self):
        model_name = "california_housing_Gradient Boosting"  # or dynamically pass
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        self.next(self.predict)

    @step
    def predict(self):
        self.preds = self.model.predict(self.X_new)
        self.X_new["Predicted_MedHouseVal"] = self.preds
        self.X_new.to_csv("data/predictions.csv", index=False)
        print("Saved predictions to data/predictions.csv")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")

if __name__ == '__main__':
    ScoringFlow()
