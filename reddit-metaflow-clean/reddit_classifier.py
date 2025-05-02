from metaflow import FlowSpec, step, conda_base

@conda_base(python='3.10', packages={
    'pandas': '2.2.3',
    'scikit-learn': '1.6.1',
    'numpy': '2.2.5',
    'joblib': '1.4.2'
})
class RedditClassifier(FlowSpec):

    @step
    def start(self):
        print("✅ Starting RedditClassifier Flow")
        self.next(self.load_data)

    @step
    def load_data(self):
        import pandas as pd

        print("📦 Loading input data...")
        self.x_sample = pd.read_csv('sample_reddit.csv', header=None).to_numpy().reshape((-1,))
        print("✅ Data loaded.")
        self.next(self.load_model)

    @step
    def load_model(self):
        import joblib

        print("📦 Loading trained model pipeline...")
        self.loaded_pipeline = joblib.load("reddit_model_pipeline.joblib")
        print("✅ Model loaded.")
        self.next(self.predict)

    @step
    def predict(self):
        print("🔮 Making predictions...")
        self.predictions = self.loaded_pipeline.predict_proba(self.x_sample)
        print("✅ Predictions done.")
        self.next(self.save)

    @step
    def save(self):
        import pandas as pd

        print("💾 Saving predictions to 'sample_preds.csv'")
        pd.DataFrame(self.predictions).to_csv("sample_preds.csv", index=False, header=False)
        print("✅ Saved.")
        self.next(self.end)

    @step
    def end(self):
        print("🏁 Flow complete!")

if __name__ == '__main__':
    RedditClassifier()
