import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# STEP 1: Load Dataset
# ==========================================

df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")

print("Original Dataset Shape:", df.shape)


# ==========================================
# STEP 2: Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())


# ==========================================
# STEP 3: Remove Rows with Missing Rating
# ==========================================

df = df.dropna(subset=["Rating"])


# ==========================================
# STEP 4: Fill Missing Categorical Values
# ==========================================

categorical_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")


# ==========================================
# STEP 5: Convert Rating to Numeric
# ==========================================

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df = df.dropna(subset=["Rating"])

print("\nFinal Dataset Shape:", df.shape)


# ==========================================
# STEP 6: Rating Statistics
# ==========================================

print("\nRating Statistics:")
print(df["Rating"].describe())


# ==========================================
# STEP 7: Select Features and Target
# ==========================================

X = df[
    ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
]

y = df["Rating"]


# ==========================================
# STEP 8: Convert Text Features to Numbers
# ==========================================

X = pd.get_dummies(
    X,
    columns=[
        "Genre",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3"
    ]
)

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ==========================================
# STEP 9: Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ==========================================
# STEP 10: Create Random Forest Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=30,
    max_features=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# STEP 11: Train Model
# ==========================================

print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model Training Completed!")


# ==========================================
# STEP 12: Make Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# STEP 13: Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("MAE  :", round(mae, 4))
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("R2 Score :", round(r2, 4))


# ==========================================
# STEP 14: Compare Actual vs Predicted
# ==========================================

results = pd.DataFrame({
    "Actual Rating": y_test.values,
    "Predicted Rating": np.round(y_pred, 2)
})

print("\nFirst 10 Predictions:")
print(results.head(10))
# ==========================================
# STEP 15: Predict Rating for a New Movie
# ==========================================

print("\n================================")
print("NEW MOVIE RATING PREDICTION")
print("================================")

genre = input("Enter Genre: ")
director = input("Enter Director: ")
actor1 = input("Enter Actor 1: ")
actor2 = input("Enter Actor 2: ")
actor3 = input("Enter Actor 3: ")

# Create new movie dataframe
new_movie = pd.DataFrame({
    "Genre": [genre],
    "Director": [director],
    "Actor 1": [actor1],
    "Actor 2": [actor2],
    "Actor 3": [actor3]
})

# Convert new movie data into dummy variables
new_movie = pd.get_dummies(
    new_movie,
    columns=[
        "Genre",
        "Director",
        "Actor 1",
        "Actor 2",
        "Actor 3"
    ]
)

# Match new movie columns with training columns
new_movie = new_movie.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Predict rating
predicted_rating = model.predict(new_movie)[0]

# Keep rating between 1 and 10
predicted_rating = max(1, min(10, predicted_rating))

print("\nPredicted Movie Rating:",
      round(predicted_rating, 2), "/ 10")
import matplotlib.pyplot as plt

# Actual vs Predicted Ratings
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")

plt.show()