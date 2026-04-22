import tensorflow as tf
import matplotlib.pyplot as plt

# Tell Python where your main folder is
data_dir = "datasheets/training_set/"

print("Step 1: Organizing the Textbook and the Quiz...")
# 1. The Textbook (Training Set) - 80% of your images
train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2, # Save 20% for the quiz
    subset="training",    # This is the training part
    seed=123,             # Keeps the random split fair
    image_size=(150, 150),# Resize all images to be the exact same size!
    batch_size=32
)

# 2. The Practice Quiz (Validation Set) - 20% of your images
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",  # This is the quiz part
    seed=123,
    image_size=(150, 150),
    batch_size=32
)

print("\nStep 2: Building and Training the AI Brain...")
# 3. Build a simple AI Brain (A Neural Network)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255), # Normalize image colors
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4) # Assuming you have 4 face shapes!
])

# Tell the brain how to score itself
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 4. Train the AI! 
# We save the results in a variable called "history" so we can graph it later
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20 # Read the textbook 20 times (just like your graph shows!)
)

print("\nStep 3: Drawing the Graph!")
# 5. Draw the exact graph you showed me!
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show() # This pops the graph up on your screen