# MLOps Course Repository

Welcome to my MLOps course repository! This repository contains materials and resources for the Machine Learning Operations (MLOps) course I'm taking.

## Dependencies

The main dependencies for this course are:

```plaintext
mlflow==2.15.1
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
```

## Setup

To set up this environment locally:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/mlops-course-repo.git
   cd mlops-course-repo
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Verify the installation:**
   ```sh
   python -c "import mlflow, numpy, pandas, sklearn; print('All dependencies installed successfully!')"
   ```

## Usage

This repository is primarily for personal use and course submission. Each directory contains relevant materials for different topics or assignments covered in the course.

## Code

Below is a simple example of how to log a model using MLflow:

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Generate sample data
X = np.random.rand(100, 1)
y = 3 * X.squeeze() + np.random.randn(100) * 0.1

# Train a model
model = LinearRegression()
model.fit(X, y)

# Log the model
mlflow.sklearn.log_model(model, "linear_regression_model")
print("Model logged successfully!")
```

Feel free to explore and modify the materials as needed!

