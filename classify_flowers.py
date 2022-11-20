import os
import PIL
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import dataset


def train_and_save_model(normalized_train_ds, normalized_valid_ds, classes, input_shape):
    model = Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(classes))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.build(input_shape)

    model.summary()

    epochs = 10
    history = model.fit(
        normalized_train_ds,
        validation_data=normalized_valid_ds,
        epochs=epochs
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    # saving the .h5 model

    model.save('/home/neha/ImageRecognitionInTensorFlow/experiment/image_classifier.h5')
    print('Model Saved!')


def test_model(model, class_names):
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(180, 180)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


def main(train=False):
    dataset_path, classes = dataset.get_dataset(
        url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz')

    flower = str(classes[0])
    list_of_items = os.listdir(flower)
    print(flower, os.path.join(flower, list_of_items[0]))
    list_of_items = os.listdir(flower)
    img = PIL.Image.open(os.path.join(flower, list_of_items[0]))
    # img.show()

    hparams = {
        'batch_size': 32,
        'image_height': 180,
        'image_width': 180
    }
    input_shape = (None, 180, 180, 3)
    # utilise dataloader
    train_ds = tf.keras.utils.image_dataset_from_directory(dataset_path,
                                                           validation_split=0.2,
                                                           subset='training',
                                                           seed=1,
                                                           image_size=(hparams['image_height'], hparams['image_width']))

    validation_ds = tf.keras.utils.image_dataset_from_directory(dataset_path,
                                                                validation_split=0.2,
                                                                subset='validation',
                                                                seed=1,
                                                                image_size=(
                                                                    hparams['image_height'], hparams['image_width']))

    AUTOTUNE = tf.data.AUTOTUNE
    classes = train_ds.class_names
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = validation_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    normalized_train_dataset = dataset.standardize_dataset(train_ds)
    normalized_validation_dataset = dataset.standardize_dataset(valid_ds)
    if train:
        train_and_save_model(normalized_train_dataset, normalized_validation_dataset, classes=classes,
                             input_shape=(None, 180, 180, 3))
    else:
        # load model
        saved_model = load_model('/home/neha/ImageRecognitionInTensorFlow/experiment/image_classifier.h5')
        saved_model.summary()

        test_model(saved_model, classes)


if __name__ == '__main__':
    main(False)
