import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load datasets
train_path = "/content/train.csv"
test_path = "/content/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Select relevant features and target
features = ["GrLivArea", "BedroomAbvGr", "FullBath", "LotArea"]
target = "SalePrice"
X = train_df[features]
y = train_df[target]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Evaluate model performance
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae}, RMSE: {rmse}, R² Score: {r2}")

# Predict house prices on test data
X_test = test_df[features]
test_predictions = model.predict(X_test)

# Save predictions
test_df["PredictedPrice"] = test_predictions
submission = test_df[["Id", "PredictedPrice"]]
submission.to_csv("house_price_predictions.csv", index=False)
