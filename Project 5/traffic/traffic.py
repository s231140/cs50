import argparse
import os
import subprocess
import sys


def import_cv2():
    try:
        import cv2
        return cv2
    except ModuleNotFoundError:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        venv_python = os.path.abspath(
            os.path.join(base_dir, "..", "..", "..", ".venv", "Scripts", "python.exe")
        )
        current_python = os.path.abspath(sys.executable)

        if os.path.isfile(venv_python) and os.path.normcase(current_python) != os.path.normcase(venv_python):
            print("cv2 not found in the current Python interpreter.")
            print(f"Re-launching with project virtualenv: {venv_python}")
            subprocess.run([venv_python] + sys.argv)
            sys.exit(0)

        sys.exit(
            "Module 'cv2' is required but not installed. "
            "Install it with 'pip install opencv-python' in your active environment. "
            "If you are using the project virtualenv, run '.\\.venv\\Scripts\\python.exe -m pip install opencv-python'."
        )

cv2 = import_cv2()

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    parser = argparse.ArgumentParser(
        description="Train and evaluate a traffic sign classifier using the GTSRB dataset."
    )
    parser.add_argument("data_directory", nargs="?", default=None)
    parser.add_argument("model_file", nargs="?", default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})",
    )
    args = parser.parse_args()

    default_data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "gtsrb")
    )
    data_dir = args.data_directory or default_data_dir

    if not os.path.isdir(data_dir):
        sys.exit(f"Data directory not found: {data_dir}")

    print(f"Using data directory: {data_dir}")

    images, labels = load_data(data_dir)
    if not images:
        sys.exit("No images were loaded. Check the dataset path and image files.")

    print(f"Loaded {len(images)} images from {len(set(labels))} categories.")

    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    model = get_model()

    print("Starting training...")
    model.fit(x_train, y_train, epochs=args.epochs)

    print("Evaluating model...")
    results = model.evaluate(x_test, y_test, verbose=2)
    if isinstance(results, list) and len(results) >= 2:
        print(f"Evaluation result: loss={results[0]:.4f}, accuracy={results[1]:.4f}")
    else:
        print(f"Evaluation result: {results}")

    if args.model_file:
        model.save(args.model_file)
        print(f"Model saved to {args.model_file}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_dir):
            continue

        for filename in os.listdir(category_dir):
            file_path = os.path.join(category_dir, filename)
            if not os.path.isfile(file_path):
                continue

            image = cv2.imread(file_path)
            if image is None:
                continue

            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image = image.astype('float32') / 255.0
            images.append(image)
            labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)

