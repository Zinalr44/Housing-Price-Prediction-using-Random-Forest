import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load your dataset
file_path = r'C:\Users\Dell.com\AppData\Local\Temp\f3e64102-c22b-41e0-85e2-40c3e12b7f48_archive (2).zip.f48\Housing.csv'
housing_data = pd.read_csv(file_path)

# Display the entire dataset
print("Entire Dataset:")
print(housing_data)

# Separate target variable (price) from features
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Define numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Combine actual and predicted prices into a DataFrame
result_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': predictions})

# Display actual and predicted prices
print("\nActual vs Predicted Prices:")
print(result_df)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy*100}")
