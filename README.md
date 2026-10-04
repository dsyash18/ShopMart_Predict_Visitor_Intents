## Shopmart Predict Visitor Intents

A machine learning pipeline built to predict whether an online visitor will complete a purchase during their browsing session.

# The Problem
E-commerce platform "Amazon" & "FlipKart" tracks thousands of daily user sessions, recording online activity like page visits and session duration. However, without a way to identify which visitors actually intend to buy, marketing resources are spent inefficiently and revenue opportunities get missed.

Using a dataset of "12,330 unique sessions" collected over a year, this project builds a classification model to detect purchase intent from browsing metrics.

# What’s Inside

-> Data Preprocessing & Scaling: Used `ColumnTransformer` to standardise numerical metrics with `StandardScaler` and encode categorical factors with `OneHotEncoder`.
-> Leak-Free Architecture: Bundled feature transformation and modeling steps into an `sklearn.pipeline.Pipeline` to protect against data leakage.
-> Handling Imbalanced Classes: Applied `class_weight="balanced"` to ensure non-purchasing session volume doesn't overpower purchasing signals.
-> Pruned Decision Tree: Used explicit tree depth limits (`max_depth=6`, `min_samples_leaf=30`) to control decision boundaries and avoid overfitting.
-> Hyperparameter Optimization: Used `GridSearchCV` with 5-fold cross-validation tuned to maximize the "F1-Score".

---

## Pipeline Overview

--- Python
import pandas as pd,
from sklearn.model_selection import train_test_split, GridSearchCV,
from sklearn.compose import ColumnTransformer,
from sklearn.preprocessing import StandardScaler, OneHotEncoder,
from sklearn.pipeline import Pipeline,
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Load session data
df = pd.read_csv("Online_shoppers.csv")[cite: 3]

X = df.drop(columns=["Revenue"])[cite: 3]
y = df["Revenue"].astype(int)[cite: 3]

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),[cite: 3]
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# Decision tree with pre-pruning & balanced weights
dt = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)

# Full pipeline
pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", dt)
])
pipe.fit(X_train, y_train)

# Hyperparameter search
param_grid = {
    "model__max_depth": 
    "model__min_samples_leaf":
}

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)
