# Custom ML Model Evaluator

**Custom Model Evaluator** is a web application that allows users to **upload their own pre-trained binary classification machine learning models** (in ```.pkl``` or ```.pickle``` format) along with a dynamic input schema. The app automatically generates an input form based on the model's expected features, enabling users to test predictions with real-time input and view the model's output probabilities. Ideal for quick validation, demoing, or interactive evaluation of custom ML models — without writing extra code.

## Features
 
- Dynamically generates input fields for each feature.  
- Supports numeric (`int`, `float`) and `string` features.  
- Display prediction and probabilities interactively.  
- Easy to extend for binary classification ML model.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/chaanakyaaM/Custom_Model_Evaluator.git
cd Custom-Model-Evaluator
```
2. Install required packages:
```
pip install streamlit scikit-learn numpy 
```
or
```
pip install -r requirements.txt
```

3. Run the app:
```
streamlit run main.py
```

## Usage

### 1. IMPORTANT : Prepare your ML Model and Feature Schema 

Your model file must include both:
- A trained binary classification model.
- A feature schema describing the model’s expected input features.

You can use [feature-schema](https://pypi.org/project/feature-schema) package to get the feature schema.

Here’s an example using feature_schema for creating pickle ```(.plk)``` file:

```python
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from feature_schema import FeatureSchema

# Train your custom model
model = LogisticRegression()
model.fit(X, y)

# Create feature schema using only the input features (X)
fs = FeatureSchema(X)
# Note : Do not include the y variable in the feature-schema

# Bundle model and schema into a pickle
package = {
    "model": model,
    "schema": fs.to_dict()  
}

# create a pkl file of your pre-trained model
with open("model_with_schema.pkl", "wb") as f:
    pickle.dump(package, f)
```

### 2. Upload and Predict

- Open the Streamlit app.
- Upload model_with_schema.pkl.
- Enter feature values in the dynamically generated inputs.
- Click "Predict" to evaluate result along with probabilities

> Try uploading sample model ```model_with_schema.pkl``` for quick testing.

## Why Use This App?

- No more hardcoding feature names, types, or ranges.
- Works with only binary classification ML model that follows scikit-learn conventions.
- Automatically validates input based on feature schema.
- Instant predictions with probability insights.
- Perfect for demos and quick model evaluations.