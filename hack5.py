import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok

# Suppress unnecessary TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(_name_)

# Start an ngrok tunnel
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)  # 10 categories of clothing
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
print("Training the model...")
model.fit(train_images, train_labels, epochs=5)

# Create a dictionary to map label indices to clothing categories
clothing_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Outfit Recommender</title>
    </head>
    <body>
        <h1>Outfit Recommendation System</h1>
        <form action="/recommend" method="post">
            <label for="color">Preferred Color:</label>
            <input type="text" id="color" name="color" required><br><br>
            <label for="body_shape">Body Shape:</label>
            <input type="text" id="body_shape" name="body_shape" required><br><br>
            <button type="submit">Get Recommendations</button>
        </form>
    </body>
    </html>
    """

@app.route('/recommend', methods=['POST'])
def recommend_outfit():
    # Extract user input
    color_preference = request.form.get('color')
    body_shape = request.form.get('body_shape')

    # Generate a recommendation using the trained model
    random_image = test_images[np.random.randint(0, len(test_images))]
    prediction = model.predict(np.expand_dims(random_image, axis=0))
    category = clothing_labels[np.argmax(prediction)]

    # Customize the recommendation based on user input
    recommendation = {
        "item": category,
        "color_preference": color_preference.capitalize(),
        "body_shape": body_shape.capitalize()
    }
    return jsonify(recommendation)

if _name_ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)