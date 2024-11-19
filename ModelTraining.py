import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Constants
GRID_SIZE = 20  # Must match the grid size used in the game
NUM_FRAMES = 10  # Number of past frames to use as input
BATCH_SIZE = 1  # Batch size of 1 to preserve sequence order
EPOCHS = 25
INITIAL_LEARNING_RATE = 0.01

# Grid encoding
EMPTY_CELL = 0
SNAKE_CELL = 1
FOOD_CELL = 2

# Input encoding
INPUT_ENCODING = {
    'U': [1, 0, 0, 0, 0],  # Up
    'D': [0, 1, 0, 0, 0],  # Down
    'L': [0, 0, 1, 0, 0],  # Left
    'R': [0, 0, 0, 1, 0],  # Right
    'N': [0, 0, 0, 0, 1],  # None
}

# 1. Load the Recorded Data
def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    frames = []
    inputs = []
    for entry in data:
        frames.append(entry['frame'])
        inputs.append(entry['input'])
    return frames, inputs

# 2. Preprocess the Data
def preprocess_data(frames, inputs, stride=1):
    X_frames = []
    X_inputs = []
    y_frames = []

    num_samples = (len(frames) - NUM_FRAMES - 1) // stride

    for idx in range(0, num_samples * stride, stride):
        # Input frames: last NUM_FRAMES frames
        frame_sequence = frames[idx:idx+NUM_FRAMES]
        frame_sequence = np.array(frame_sequence)  # Shape: (NUM_FRAMES, GRID_SIZE, GRID_SIZE)
        X_frames.append(frame_sequence)

        # Input keyboard: input at frame idx+NUM_FRAMES
        input_key = inputs[idx+NUM_FRAMES]  # Shifted by one to align with the next frame
        input_encoded = INPUT_ENCODING.get(input_key, [0, 0, 0, 0, 1])
        X_inputs.append(input_encoded)

        # Output frame: frame at idx+NUM_FRAMES+1
        next_frame = frames[idx+NUM_FRAMES+1]
        y_frames.append(next_frame)

    # Convert to NumPy arrays
    X_frames = np.array(X_frames, dtype=np.float32)
    X_inputs = np.array(X_inputs, dtype=np.float32)
    y_frames = np.array(y_frames, dtype=np.float32)

    # Normalize frame data
    X_frames /= FOOD_CELL  # Assuming values are 0,1,2; normalize to 0-1
    y_frames /= FOOD_CELL

    # Reshape frames
    X_frames = X_frames.reshape(-1, NUM_FRAMES, GRID_SIZE, GRID_SIZE)
    y_frames = y_frames.reshape(-1, GRID_SIZE, GRID_SIZE, 1)

    return X_frames, X_inputs, y_frames

# Function to split data without shuffling
def split_data(X_frames, X_inputs, y_frames, validation_split=0.2):
    num_samples = len(X_frames)
    split_index = int(num_samples * (1 - validation_split))
    X_frames_train = X_frames[:split_index]
    X_frames_val = X_frames[split_index:]
    X_inputs_train = X_inputs[:split_index]
    X_inputs_val = X_inputs[split_index:]
    y_frames_train = y_frames[:split_index]
    y_frames_val = y_frames[split_index:]
    return X_frames_train, X_frames_val, X_inputs_train, X_inputs_val, y_frames_train, y_frames_val

# Create tf.data.Dataset
def create_dataset(X_frames, X_inputs, y_frames):
    dataset = tf.data.Dataset.from_tensor_slices(({'frames_input': X_frames, 'keyboard_input': X_inputs}, y_frames))
    return dataset

# Transformer Encoder Block
def transformer_encoder(inputs, num_heads, key_dim, ff_dim, dropout=0.1):
    # Multi-Head Attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention_output = layers.Dropout(dropout)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-Forward Network
    ffn = layers.Dense(ff_dim, activation='relu')(attention_output)
    ffn = layers.Dense(key_dim)(ffn)
    ffn_output = layers.Dropout(dropout)(ffn)
    ffn_output = LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
    return ffn_output

# 3. Define the Transformer Model
def create_transformer_model():
    # Input layers
    frames_input = layers.Input(shape=(NUM_FRAMES, GRID_SIZE, GRID_SIZE), name='frames_input')
    keyboard_input = layers.Input(shape=(5,), name='keyboard_input')  # Updated to match new input encoding

    # Flatten spatial dimensions
    frames_flat = layers.Reshape((NUM_FRAMES, GRID_SIZE * GRID_SIZE))(frames_input)

    # Frame Embeddings
    frames_embedded = layers.Dense(64)(frames_flat)

    # Positional Encoding
    positions = tf.range(start=0, limit=NUM_FRAMES, delta=1)
    position_embedding = layers.Embedding(input_dim=NUM_FRAMES, output_dim=64)(positions)
    position_embedding = tf.expand_dims(position_embedding, 0)  # Add batch dimension
    frames_embedded += position_embedding

    # Transformer Encoder Blocks
    x = frames_embedded
    for _ in range(2):  # Number of transformer encoder layers
        x = transformer_encoder(x, num_heads=4, key_dim=64, ff_dim=128, dropout=0.1)

    # Take the output corresponding to the last time step
    x = x[:, -1, :]  # Shape: (batch_size, 64)

    # Process keyboard input
    keyboard_embed = layers.Dense(64, activation='relu')(keyboard_input)

    # Combine transformer output and keyboard input
    x = layers.Concatenate()([x, keyboard_embed])

    # Decode to output frame
    x = layers.Dense(GRID_SIZE * GRID_SIZE, activation='sigmoid')(x)
    x = layers.Reshape((GRID_SIZE, GRID_SIZE, 1))(x)

    # Create model
    model = models.Model(inputs=[frames_input, keyboard_input], outputs=x)

    # Compile model with initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model

# Main function
def main():
    # Load data
    frames, inputs = load_data('snake_game_data.json')
    print("Data loaded.")

    # Preprocess data
    X_frames, X_inputs, y_frames = preprocess_data(frames, inputs)
    print("Data preprocessed.")

    # Split data into training and validation sets without shuffling
    X_frames_train, X_frames_val, X_inputs_train, X_inputs_val, y_frames_train, y_frames_val = split_data(
        X_frames, X_inputs, y_frames, validation_split=0.2)
    print("Data split into training and validation sets.")

    # Create datasets
    train_dataset = create_dataset(X_frames_train, X_inputs_train, y_frames_train)
    val_dataset = create_dataset(X_frames_val, X_inputs_val, y_frames_val)

    # Batch datasets without shuffling
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # Create model
    model = create_transformer_model()
    model.summary()

    # Define the learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=2, min_lr=1e-6, verbose=1)

    # Train the model with the learning rate scheduler
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        shuffle=False,  # Disable shuffling between epochs
        callbacks=[reduce_lr]
    )
    print("Training completed.")

    # Save the model
    model_save_path = 'snake_transformer_model.keras'
    model.save(model_save_path)
    print(f"Model saved to '{model_save_path}'.")

if __name__ == "__main__":
    main()
