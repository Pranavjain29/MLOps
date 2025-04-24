# src/trainingflowgcp.py

from metaflow import FlowSpec, step, kubernetes, resources, timeout, retry, catch, conda
from utils.data_utils import load_and_split_data, perform_feature_selection
from utils.model_utils import train_and_evaluate_all_models, register_best_model
import mlflow

class TrainingFlow(FlowSpec):

    @conda(libraries={"scikit-learn": "1.3.0", "pandas": "1.5.3", "mlflow": "2.11.1"})
    @step
    def start(self):
        self.df, self.X, self.y = load_and_split_data()
        self.next(self.split_data)

    @conda(libraries={"scikit-learn": "1.3.0"})
    @step
    def split_data(self):
        from sklearn.model_selection import train_test_split
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        self.next(self.select_features)

    @conda(libraries={"scikit-learn": "1.3.0"})
    @step
    def select_features(self):
        self.selected_features = perform_feature_selection(self.X_train, self.y_train)
        self.X_train_reduced = self.X_train[self.selected_features]
        self.X_val_reduced = self.X_val[self.selected_features]
        self.X_test_reduced = self.X_test[self.selected_features]
        self.next(self.train_models)

    @conda(libraries={"scikit-learn": "1.3.0", "pandas": "1.5.3", "mlflow": "2.11.1"})
    @kubernetes(cpu=2, memory=4000)
    @resources(cpu=2, memory=4000)
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="error")
    @step
    def train_models(self):
        self.all_results = train_and_evaluate_all_models(self.X_train, self.y_train, self.X_val, self.y_val,
                                                         self.X_train_reduced, self.X_val_reduced)
        self.next(self.register_model)

    @conda(libraries={"scikit-learn": "1.3.0", "pandas": "1.5.3", "mlflow": "2.11.1"})
    @kubernetes(cpu=2, memory=4000)
    @resources(cpu=2, memory=4000)
    @timeout(seconds=600)
    @retry(times=2)
    @catch(var="error")
    @step
    def register_model(self):
        self.best_model_info = register_best_model(self.all_results, self.X_test, self.X_test_reduced, self.y_test)
        self.next(self.end)

    @step
    def end(self):
        print(f"Best model: {self.best_model_info['model_name']}")
        print(f"MLflow URI: {self.best_model_info['mlflow_uri']}")

if __name__ == '__main__':
    TrainingFlow()
