import json
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load the model and dataset when the Lambda function initializes
# This ensures that the model and data are loaded only once when the Lambda function is cold-started.

def load_model():
    # Define the CNN architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Load the model and dataset once (cold start optimization)
model = load_model()

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images / 255.0  # Normalize pixel values to be between 0 and 1

def lambda_handler(event, context):
    try:
        # Evaluate the model on the test data
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        
        # Return the test accuracy as the response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'test_accuracy': test_acc
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
