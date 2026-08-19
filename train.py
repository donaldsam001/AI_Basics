import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

# 1. Load Data
df = pd.read_csv('data/ats_resume_dataset_elite_v3.csv')

# 2. Feature Engineering
df['certifications'] = df['certifications'].fillna('None')
df['has_certification'] = (df['certifications'] != 'None').astype(int)
df['num_resume_skills'] = df['resume_skills'].apply(lambda x: len(str(x).split(',')))
df['num_required_skills'] = df['required_skills'].apply(lambda x: len(str(x).split(',')))
df['experience_diff'] = df['experience_years'] - df['job_experience_required']

feature_cols = [
    'experience_years', 'job_experience_required', 'experience_diff',
    'skill_match_score', 'experience_match', 'education_match',
    'similarity_score', 'has_certification', 'num_resume_skills', 'num_required_skills',
    'education_level', 'job_role'
]

categorical_cols = ['education_level', 'job_role']
numeric_cols = [col for col in feature_cols if col not in categorical_cols]

X = df[feature_cols]
y = df['shortlisted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocessor & Classifier Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=250,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 4. Train and Serialize
pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'model_pipeline.joblib')
print("Model pipeline trained and saved to 'model_pipeline.joblib'.")