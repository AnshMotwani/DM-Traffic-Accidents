import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# Load and preprocess data
def load_and_preprocess(file_path):
    # Load data with memory optimization
    df = pd.read_csv(file_path, parse_dates=["start_time"])

    # Feature Selection
    keep_columns = [
        "start_lat",
        "start_lng",
        "distance(mi)",
        "temperature(f)",
        "humidity(%)",
        "pressure(in)",
        "visibility(mi)",
        "wind_speed(mph)", s
        "precipitation(in)",
        "start_hour",
        "start_weekday",
        "junction",
        "traffic_signal",
        "weather_condition",
        "wind_direction",
        "sunrise_sunset",
        "state",
        "severity",
    ]

    df = df[keep_columns]

    # Convert binary features
    binary_features = ["junction", "traffic_signal"]
    df[binary_features] = df[binary_features].astype(int)

    # Process categorical features
    categorical_features = [
        "weather_condition",
        "wind_direction",
        "sunrise_sunset",
        "state",
    ]
    df = pd.get_dummies(df, columns=categorical_features)

    # Separate features and target
    X = df.drop("severity", axis=1).values
    y = df["severity"].values

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


# Neural Network Model
def create_model(input_shape):
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),  # Output layer for regression
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mae",
        metrics=["mae", "mse"],
    )
    return model


# Main execution
if __name__ == "__main__":
    # Load data (replace with your file path)
    X, y, scaler = load_and_preprocess("dataset/cleaned_us_accident_data.csv")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train model
    model = create_model(X_train.shape[1])

    early_stop = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1,
    )

    # Evaluation
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # Visualize training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Loss Progression")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Training MAE")
    plt.plot(history.history["val_mae"], label="Validation MAE")
    plt.title("MAE Progression")
    plt.ylabel("MAE")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.show()
