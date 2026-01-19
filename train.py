# Necessary Imports
import pickle
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

# Load Data
df = pd.read_csv('insurance.csv')

# Drop duplicates
df = df.drop_duplicates()

# Prepare X and y
target_col = 'charges'
X = df.drop(target_col, axis = 1)
y = df[target_col]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract numeric and categorical columns
numeric_cols = [c for c in X_train.columns if X_train[c].dtype != 'object']
cat_cols = [c for c in X_train.columns if X_train[c].dtype == 'object']

# Outlier Handling using clip method
Q1 = X_train['bmi'].quantile(0.25)
Q3 = X_train['bmi'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

X_train['bmi'] = X_train['bmi'].clip(lower, upper)
X_test['bmi'] = X_test['bmi'].clip(lower, upper)
outliers_train = X_train[(X_train['bmi'] < lower) | (X_train['bmi'] > upper)]
outliers_test = X_test[(X_test['bmi'] < lower) | (X_test['bmi'] > upper)]

num_pipeline = Pipeline([
    ('scalar', StandardScaler())
])

cat_pipeline = Pipeline([
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Define Model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42
)

# Train The Model
pipeline = Pipeline([
      ('preprocessor', preprocessor),
      ('model', model)
  ])

pipeline.fit(X_train, y_train)

# Evaluation
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"R2 Score : {r2}, RMSE : {rmse}, MAE: {mae}")

# Save The Model
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(f"Model saves as model.pkl")