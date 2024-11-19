import tkinter as tk
import numpy as np
import tensorflow as tf
from collections import deque

# Constants
GRID_SIZE = 20       # Must match the grid size used during training
CELL_SIZE = 20       # Size of each cell in pixels
NUM_FRAMES = 10      # Number of past frames to use as input
FOOD_CELL = 2        # The value used to represent food in the grid
SNAKE_CELL = 1       # The value used to represent the snake body in the grid
SPEED = 100          # Delay between moves in milliseconds

# Input encoding
INPUT_ENCODING = {
    'U': [1, 0, 0, 0, 0],
    'D': [0, 1, 0, 0, 0],
    'L': [0, 0, 1, 0, 0],
    'R': [0, 0, 0, 1, 0],
    'N': [0, 0, 0, 0, 1],  # Represents no input
}

class SnakeGameModel:
    def __init__(self, master, model_path):
        self.master = master
        self.master.title("Snake Game with Transformer Model")
        self.canvas = tk.Canvas(master, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE)
        self.canvas.pack()

        # Load the trained model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None  # Handle model loading failure appropriately

        # Initialize attributes
        self.after_id = None
        self.direction = 'R'  # Initial direction (encoded as 'R' to match INPUT_ENCODING)
        self.next_direction = 'R'  # Next direction
        self.last_input = 'R'  # Last input

        # Initialize game state
        self.reset_game()

        # Bind keys
        self.master.bind("<Key>", self.on_key_press)

        # Start game loop
        self.update()

    def reset_game(self):
        self.game_over = False

        # Starting position of the snake and food
        self.snake = [np.array([GRID_SIZE // 2, GRID_SIZE // 2])]
        self.direction = 'R'
        self.next_direction = 'R'
        self.last_input = 'R'
        self.place_food()

        # Initialize frame history
        self.frame_history = deque(maxlen=NUM_FRAMES)

        # Initialize the first few frames to provide context
        for _ in range(NUM_FRAMES):
            self.move_snake()
            frame = self.get_current_frame()
            self.frame_history.append(frame)

        # Draw the initial state
        self.draw()

    def place_food(self):
        while True:
            position = np.array([np.random.randint(0, GRID_SIZE), np.random.randint(0, GRID_SIZE)])
            if not any((position == segment).all() for segment in self.snake):
                self.food = position
                break

    def on_key_press(self, event):
        key = event.keysym
        key_map = {'Up': 'U', 'Down': 'D', 'Left': 'L', 'Right': 'R'}
        if key in key_map:
            if self.game_over:
                self.reset_game()
            else:
                # Prevent the snake from reversing direction
                opposite_directions = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}
                if key_map[key] != opposite_directions.get(self.direction, ''):
                    self.next_direction = key_map[key]
                    self.last_input = key_map[key]

    def update(self):
        if not self.game_over:
            # Prepare model input
            frames_input = np.array(self.frame_history, dtype=np.float32) / FOOD_CELL  # Normalize frames
            frames_input = frames_input.reshape(1, NUM_FRAMES, GRID_SIZE, GRID_SIZE)

            # Keyboard input
            keyboard_input = np.array(INPUT_ENCODING.get(self.last_input, INPUT_ENCODING['N']), dtype=np.float32)
            keyboard_input = keyboard_input.reshape(1, -1)

            # Predict next frame
            try:
                predicted_frame = self.model.predict([frames_input, keyboard_input])
                predicted_frame = predicted_frame.reshape(GRID_SIZE, GRID_SIZE)
                predicted_frame = (predicted_frame * FOOD_CELL).round().astype(int)  # Convert back to original scale
            except Exception as e:
                print(f"Error during model prediction: {e}")
                self.game_over = True
                return

            # Update game state based on predicted frame
            self.update_game_state(predicted_frame)

            # Update frame history
            self.frame_history.append(predicted_frame)

            # Reset last_input to 'None' until the next key press
            self.last_input = 'N'
            self.direction = self.next_direction

            # Draw the game
            self.draw()

            # Schedule the next update
            self.after_id = self.master.after(SPEED, self.update)
        else:
            self.canvas.create_text(
                GRID_SIZE * CELL_SIZE / 2,
                GRID_SIZE * CELL_SIZE / 2,
                text="Game Over",
                fill="red",
                font=("Arial", 24)
            )
            # Cancel any scheduled updates
            if self.after_id:
                self.master.after_cancel(self.after_id)
                self.after_id = None

    def update_game_state(self, predicted_frame):
        # Extract snake positions from predicted frame
        snake_positions = np.argwhere(predicted_frame == SNAKE_CELL)
        if snake_positions.size == 0:
            # No snake on the board, game over
            self.game_over = True
            return

        # Update snake
        self.snake = [np.array(pos) for pos in snake_positions]

        # Check for collisions
        head = self.snake[0]
        if any((head == segment).all() for segment in self.snake[1:]):
            # Snake collided with itself
            self.game_over = True
            return
        if (head < 0).any() or (head >= GRID_SIZE).any():
            # Snake collided with wall
            self.game_over = True
            return

        # Update food position
        food_positions = np.argwhere(predicted_frame == FOOD_CELL)
        if food_positions.size > 0:
            self.food = food_positions[0]
        else:
            # If no food is present, place a new one
            self.place_food()

    def move_snake(self):
        # Update the snake's position based on the current direction
        direction_vector = {
            'U': np.array([-1, 0]),
            'D': np.array([1, 0]),
            'L': np.array([0, -1]),
            'R': np.array([0, 1]),
        }
        move = direction_vector[self.direction]
        new_head = self.snake[0] + move

        # Move snake
        self.snake.insert(0, new_head)

        # Remove tail
        self.snake.pop()

        # Ensure the snake stays within bounds
        if (new_head < 0).any() or (new_head >= GRID_SIZE).any():
            self.game_over = True

    def draw(self):
        self.canvas.delete("all")

        # Draw snake
        for segment in self.snake:
            x = segment[1] * CELL_SIZE
            y = segment[0] * CELL_SIZE
            self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="green")

        # Draw food
        x = self.food[1] * CELL_SIZE
        y = self.food[0] * CELL_SIZE
        self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="red")

    def get_current_frame(self):
        frame = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        # Mark snake positions
        for segment in self.snake:
            row, col = int(segment[0]), int(segment[1])
            if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                frame[row, col] = SNAKE_CELL  # Snake body
        # Mark food position
        row, col = int(self.food[0]), int(self.food[1])
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            frame[row, col] = FOOD_CELL  # Food
        return frame

if __name__ == "__main__":
    root = tk.Tk()
    model_path = 'snake_transformer_model.keras'  # Update with your model's path
    game = SnakeGameModel(root, model_path)
    root.mainloop()
