import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generating synthetic dataset for railway track crack detection and wear/tear
N = 1000
np.random.seed(42)

def generate_synthetic_data(N):
    # Crack detection data
    normal_data = np.sin(0.02 * np.arange(N)) + 0.5 * np.random.rand(N)  # Normal condition
    crack_data = 2 * (np.sin(0.02 * np.arange(N)) + np.random.rand(N))  # Crack condition

    labels_crack = np.zeros(N)
    labels_crack[N//2:N//2 + N//10] = 1  # Simulate cracks in the middle of the dataset

    data_crack = normal_data
    data_crack[N//2:N//2 + N//10] = crack_data[N//2:N//2 + N//10]

    # Wear and tear synthetic data
    wear_features = {
        "Elastic Modulus": np.random.uniform(209, 210, N),
        "Yield Strength": np.random.uniform(255, 260, N),
        "Ultimate Tensile Strength": np.random.uniform(870, 880, N),
        "Surface Hardness": np.random.uniform(250, 310, N),
        "Density": np.random.uniform(7850, 7850, N),
        "Thermal Conductivity": np.random.uniform(49.5, 50, N),
        "Coefficient of Thermal Expansion": np.full(N, 1.15e-5),
        "Fatigue Resistance": np.random.choice([1, 0.9], size=N, p=[0.8, 0.2])  # High vs slightly reduced
    }

    wear_labels = np.random.choice([0, 1], size=N, p=[0.7, 0.3])  # Simulated binary wear classification

    return data_crack, labels_crack, pd.DataFrame(wear_features), wear_labels

# Generate synthetic data
signal_data, crack_labels, wear_data, wear_labels = generate_synthetic_data(N)

# Combine data into a single DataFrame
df = pd.DataFrame({"signal": signal_data, "crack_label": crack_labels})
df = pd.concat([df, wear_data], axis=1)
df["wear_label"] = wear_labels

# Plot signal data with crack labels
plt.figure(figsize=(12, 6))
plt.plot(signal_data, label="Signal")
plt.scatter(np.arange(N), crack_labels * max(signal_data), color='red', label="Cracks (Label)", alpha=0.5)
plt.title("Synthetic Railway Track Data")
plt.xlabel("Time Step")
plt.ylabel("Signal Amplitude")
plt.legend()
plt.show()

# Insights: Average wear and tear values
wear_insights = wear_data.mean()
print("Wear and Tear Insights:")
print(wear_insights)

# Bar plot for wear insights
plt.figure(figsize=(10, 6))
wear_insights.plot(kind='bar', color='skyblue')
plt.title("Average Wear and Tear Feature Values")
plt.xlabel("Features")
plt.ylabel("Average Value")
plt.show()

# Prepare sequences for LSTM
step = 10  # Window size for sequences

def convert_to_sequences(data, labels, step):
    X, Y = [], []
    for i in range(len(data) - step):
        X.append(data[i:i + step])
        Y.append(labels[i + step - 1])
    return np.array(X), np.array(Y)

# Crack detection sequences
X_crack, y_crack = convert_to_sequences(df["signal"].values, df["crack_label"].values, step)
X_crack = X_crack.reshape((X_crack.shape[0], X_crack.shape[1], 1))

# Wear and tear sequences
wear_features = df.drop(columns=["signal", "crack_label", "wear_label"]).values
X_wear, y_wear = convert_to_sequences(wear_features, df["wear_label"].values, step)

# Split into training and testing sets
X_crack_train, X_crack_test, y_crack_train, y_crack_test = train_test_split(X_crack, y_crack, test_size=0.2, random_state=42)
X_wear_train, X_wear_test, y_wear_train, y_wear_test = train_test_split(X_wear, y_wear, test_size=0.2, random_state=42)

# Build combined model for crack detection and wear prediction
input_crack = Input(shape=(step, 1))
lstm_crack = LSTM(units=32, activation="relu")(input_crack)
output_crack = Dense(1, activation="sigmoid", name="crack_output")(lstm_crack)

input_wear = Input(shape=(step, wear_features.shape[1]))
lstm_wear = LSTM(units=32, activation="relu")(input_wear)
output_wear = Dense(1, activation="sigmoid", name="wear_output")(lstm_wear)

model = Sequential()
model = Sequential()
model.add(LSTM(units=32, input_shape=(step, 1), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_crack_train, y_crack_train, epochs=20, batch_size=16, verbose=2, validation_split=0.2)

# Evaluate model for wear
model_wear = Sequential()
model_wear.add(LSTM(units=32, input_shape=(step, wear_features.shape[1]), activation="relu"))
model_wear.add(Dense(16, activation="relu"))
model_wear.add(Dense(1, activation="sigmoid"))
model_wear.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_wear.summary()

history_wear = model_wear.fit(X_wear_train, y_wear_train, epochs=20, batch_size=16, verbose=2, validation_split=0.2)

# Plot accuracy and loss for crack detection
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy - Crack')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy - Crack')
plt.title("Training and Validation Accuracy (Crack Detection)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss - Crack')
plt.plot(history.history['val_loss'], label='Validation Loss - Crack')
plt.title("Training and Validation Loss (Crack Detection)")
plt.legend()
plt.show()

# Plot accuracy and loss for wear prediction
plt.figure(figsize=(12, 6))
plt.plot(history_wear.history['accuracy'], label='Training Accuracy - Wear')
plt.plot(history_wear.history['val_accuracy'], label='Validation Accuracy - Wear')
plt.title("Training and Validation Accuracy (Wear Prediction)")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history_wear.history['loss'], label='Training Loss - Wear')
plt.plot(history_wear.history['val_loss'], label='Validation Loss - Wear')
plt.title("Training and Validation Loss (Wear Prediction)")
plt.legend()
plt.show()
