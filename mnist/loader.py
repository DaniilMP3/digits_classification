import numpy as np
import gzip


def _training_images(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return [np.reshape(x, (row_count * column_count, 1)) / 255.0 for x in images]


def _training_labels(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        identity_matrix = np.eye(10)
        return [np.reshape(x, (10, 1)) for x in identity_matrix[labels]]


def get_data():
    return zip(_training_images('mnist/samples/train-images-idx3-ubyte.gz'),
               _training_labels('mnist/samples/train-labels-idx1-ubyte.gz'))


def get_test_data():
    return zip(_training_images('mnist/samples/t10k-images-idx3-ubyte.gz'),
               _training_labels('mnist/samples/t10k-labels-idx1-ubyte.gz'))













