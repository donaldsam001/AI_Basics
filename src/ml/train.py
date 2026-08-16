import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from xgboost import XGBClassifier

FEATURES = [
    "semantic_score",
    "required_skill_score",
    "preferred_skill_score",
    "experience_score",
    "retrieval_score",
]


df = pd.read_csv(
    "data/training/cv_job_matches.csv"
)

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
)

model.fit(
    X_train,
    y_train,
)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print(
    classification_report(
        y_test,
        pred,
    )
)

print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        prob,
    )
)