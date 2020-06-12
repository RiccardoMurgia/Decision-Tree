# Import the necessary modules and libraries
from emnist import extract_training_samples
from emnist import extract_test_samples
import numpy as np
from numpy import save



def manageDataSet(dimension, X, y):
    labels = [y[i] for i in range(0, dimension)]
    images = np.zeros((dimension, 784))
    for letter in range(0, dimension):
        count = 0
        for i in range(0, 28):
            for j in range(0, 28):
                images[letter][count] = X[letter][i][j]
                count += 1
    return images, labels


def balanceDataSet(X, y):
    data_set = np.hstack((X, y.reshape(len(y), 1)))
    np.random.shuffle(data_set)
    return data_set[:, :-1], data_set[:, -1]


def saveDataSet(dataSetType):

    if dataSetType == 'digits':
        # Extract Dataset
        print('Extraction Dataset')
        X_train, y_train = extract_training_samples('digits')
        X_test, y_test = extract_test_samples('digits')

        # Reshape Dataset
        print('Reshaping Dataset ')
        images_train, labels_train = manageDataSet(len(y_train), X_train, y_train)
        images_test, labels_test = manageDataSet(len(y_test), X_test, y_test)

        # Save the Dataset
        print('Saving Dataset')
        save("images_numbers_train.npy", images_train)
        save("labels_numbers_train.npy", labels_train)
        save("images_numbers_test.npy", images_test)
        save("labels_numbers_test.npy", labels_test)

    if dataSetType == 'letters':
        # Extract Dataset
        print('Extraction Dataset')
        X_train, y_train = extract_training_samples('letters')
        X_test, y_test = extract_test_samples('letters')

        # Reshape Dataset
        print('Reshaping Dataset ')
        imgs_train, labels_train = manageDataSet(len(y_train), X_train, y_train)
        imgs_test, labels_test = manageDataSet(len(y_test), X_test, y_test)

        # Save reshape Dataset
        print('Extraction Dataset')
        save("images_letters_train.npy", imgs_train)
        save("labels_letters_train.npy", labels_train)
        save("images_letters_test.npy", imgs_test)
        save("labels_letters_test.npy", labels_test)








