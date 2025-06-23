import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.validation import check_is_fitted
import json

# Create directories if they don't exist
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
# Load the dataset
df = pd.read_csv('data/kc_house_data.csv')

# Feature Engineering
def feature_engineering(df):
    df['date'] = pd.to_datetime(df['date'])
    df['sale_year'] = df['date'].dt.year
    df['sale_month'] = df['date'].dt.month
    df['house_age'] = df['sale_year'] - df['yr_built']
    df['years_since_renov'] = np.where(df['yr_renovated'] == 0,
                                      df['house_age'],
                                      df['sale_year'] - df['yr_renovated'])
    df['total_area'] = df['sqft_living'] + df['sqft_basement']
    df['lot_ratio'] = df['sqft_living'] / df['sqft_lot']
    df['living_ratio'] = df['sqft_living'] / df['sqft_living15']
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
    df['rooms'] = df['bedrooms'] + df['bathrooms']
    df['is_waterfront'] = df['waterfront'].apply(lambda x: 1 if x > 0 else 0)
    df['has_view'] = df['view'].apply(lambda x: 1 if x > 0 else 0)
    df['price_per_sqft'] = df['price'] / df['sqft_living']
    df = df.drop(columns=['id', 'date', 'yr_built', 'yr_renovated'])
    return df

df = feature_engineering(df)

def remove_outliers(df, column, threshold=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']:
    df = remove_outliers(df, col)

X = df.drop('price', axis=1)
# Save feature list used in training
feature_columns = X.columns.tolist()

# Save it to file so you can use it later during prediction
with open('models/feature_columns.json', 'w') as f:
    json.dump(feature_columns, f)
  # keep the new safe feature
y = df['price']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
    ])

selector = SelectKBest(score_func=f_regression, k='all')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": {
        'model': Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('regressor', LinearRegression())
        ]),
        'params': {
            'selector__k': [10, 15, 'all'],
        }
    },
    "Ridge Regression": {
        'model': Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('regressor', Ridge())
        ]),
        'params': {
            'selector__k': [10, 15, 'all'],
            'regressor__alpha': [0.1, 1.0, 10.0]
        }
    },
    "Random Forest": {
        'model': Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        'params': {
            'selector__k': [10, 15, 'all'],
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
    },
    "XGBoost": {
        'model': Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('regressor', XGBRegressor(random_state=42))
        ]),
        'params': {
            'selector__k': [10, 15, 'all'],
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 6, 9],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0]
        }
    },
    "Gradient Boosting": {
        'model': Pipeline([
            ('preprocessor', preprocessor),
            ('selector', selector),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]),
        'params': {
            'selector__k': [10, 15, 'all'],
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 6],
            'regressor__min_samples_split': [2, 5]
        }
    }
}

results = {}
best_models = {}

for name, config in models.items():
    print(f"\nTraining and tuning {name}...")
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    preds = best_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    results[name] = {
        "R2 Score": round(r2, 4),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "Accuracy (%)": round(r2 * 100, 2),
        "Best Params": grid_search.best_params_
    }
    print(f"\n{name} Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Accuracy: {r2 * 100:.2f}%")
    print(f"Best Parameters: {grid_search.best_params_}")

results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df[['R2 Score', 'RMSE', 'MAE', 'Accuracy (%)']].sort_values('RMSE'))

best_model_name = results_df['RMSE'].idxmin()
best_model = best_models[best_model_name]

# === Save the best model pipeline explicitly ===
joblib.dump(best_model, os.path.join('models', 'best_house_price_pipeline.pkl'))
print(f"\nBest model pipeline saved as 'models/best_house_price_pipeline.pkl'")

# Also save preprocessing pipeline separately (optional)
joblib.dump(preprocessor, os.path.join('models', 'preprocessor.pkl'))

with open(os.path.join('results', 'model_performance.json'), 'w') as f:
    json.dump(results, f, indent=4)

if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
    importances = best_model.named_steps['regressor'].feature_importances_
    preprocessor_step = best_model.named_steps['preprocessor']
    numeric_features = preprocessor_step.transformers_[0][2]
    selector = best_model.named_steps['selector']
    check_is_fitted(selector)
    support_mask = selector.get_support()
    if len(support_mask) == len(numeric_features):
        selected_features = np.array(numeric_features)[support_mask]
        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join('results', 'feature_importance.png'))
        plt.close()
    else:
        print("Mismatch in number of features after preprocessing and feature selection.")
else:
    print(f"{type(best_model.named_steps['regressor']).__name__} does not support feature_importances_.")

plt.figure(figsize=(18, 12))
for i, (name, model) in enumerate(best_models.items(), 1):
    preds = model.predict(X_test)
    plt.subplot(2, 3, i)
    plt.scatter(y_test, preds, alpha=0.3, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'{name} Predictions vs Actual')
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('results', 'predictions_vs_actual.png'))
plt.close()

plt.figure(figsize=(18, 12))
for i, (name, model) in enumerate(best_models.items(), 1):
    preds = model.predict(X_test)
    residuals = y_test - preds
    plt.subplot(2, 3, i)
    plt.scatter(preds, residuals, alpha=0.3)
    plt.hlines(y=0, xmin=preds.min(), xmax=preds.max(), colors='red')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{name} Residual Plot')
    plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('results', 'residual_plots.png'))
plt.close()

plt.figure(figsize=(12, 10))
top_features = importance_df['Feature'].head(10).tolist() if 'importance_df' in locals() else X.columns[:15]
corr = df[top_features + ['price']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Heatmap of Top Features')
plt.tight_layout()
plt.savefig(os.path.join('results', 'correlation_heatmap.png'))
plt.close()

plt.figure(figsize=(10, 6))
preds = best_model.predict(X_test)
errors = y_test - preds
sns.histplot(errors, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.title('Distribution of Prediction Errors')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('results', 'error_distribution.png'))
plt.close()

print("\nFinal Evaluation of Best Model:")
print(f"Model: {best_model_name}")
print(f"R² Score: {results[best_model_name]['R2 Score']:.4f}")
print(f"RMSE: {results[best_model_name]['RMSE']:,.2f}")
print(f"MAE: {results[best_model_name]['MAE']:,.2f}")
print(f"Accuracy: {results[best_model_name]['Accuracy (%)']:.2f}%")
print(f"Best Parameters: {results[best_model_name]['Best Params']}")