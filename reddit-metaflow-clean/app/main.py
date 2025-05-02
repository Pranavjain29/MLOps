from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

class request_body(BaseModel):
    reddit_comment: str

@app.get('/')
def main():
    return {'message': 'This is a model for classifying Reddit comments'}

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")

@app.post('/predict')
def predict(data: request_body):
    X = [data.reddit_comment]
    probs = model_pipeline.predict_proba(X)[0]
    label = int(probs[1] > 0.5)  # 1 = remove, 0 = keep
    return {
        'comment': data.reddit_comment,
        'remove_probability': round(probs[1], 4),
        'keep_probability': round(probs[0], 4),
        'predicted_label': label,
        'label_meaning': 'REMOVE' if label == 1 else 'KEEP'
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
