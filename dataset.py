import os.path
import pathlib
import tensorflow as tf
from tensorflow.keras import layers


def get_dataset(url):
    dataset_url = url
    data_dir = '/home/neha/.keras/datasets/flower_photos'
    if not os.path.exists('/home/neha/.keras/datasets/flower_photos'):
        print('here')
        data_dir = tf.keras.utils.get_file('flower_photos',
                                           origin=dataset_url, untar=True)

    data_dir = pathlib.Path(data_dir)
    classes = list([x for x in data_dir.glob('*') if x.is_dir()])
    return data_dir, classes


def standardize_dataset(dataset):
    normalization = layers.Rescaling(1./255)
    normalized_ds = dataset.map(lambda x, y: (normalization(x), y))
    return normalized_ds
