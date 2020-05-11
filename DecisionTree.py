# Import the necessary modules and libraries
from sklearn import tree
from sklearn.metrics import accuracy_score
from emnist import extract_training_samples
from emnist import extract_test_samples
from matplotlib import pyplot as plt
import numpy as np
from numpy import save
from numpy import load
from random import *


def manage_data_set(dimension, X, y):
    labels = [y[i] for i in range(0, dimension)]
    images = np.zeros((dimension, 784))
    for letter in range(0, dimension):
        count = 0
        for i in range(0, 28):
            for j in range(0, 28):
                images[letter][count] = X[letter][i][j]
                count += 1
    return images, labels

def balance_data_set(X, y):
    data_set = np.hstack((X, y.reshape(len(y), 1)))
    np.random.shuffle(data_set)
    return data_set[:, :-1], data_set[:, -1]

def numbers_test():
    dimensions = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 240000]
    test_accuracy = []
    train_accuracy = []

    print("Digits classifier")
    '''
    # Extract Dataset
    print('Extrating Dataset')
    X_train, y_train = extract_training_samples('digits')
    X_test, y_test = extract_test_samples('digits')


    # Reshape Dataset
    print('Rashaping Dataset ')
    images_train, labels_train = manage_data_set(240000, X_train, y_train)
    images_test, labels_test = manage_data_set(40000, X_test, y_test)


    # Save the Dataset
    print('Saving Dataset')
    save("images_numbers_train.npy", images_train)
    save("labels_numbers_train.npy", labels_train)
    save("images_numbers_test.npy", images_test)
    save("labels_numbers_test.npy", labels_test)
    '''

    # Extract the balance Dataset
    images_train_init, labels_train_init = balance_data_set(load("images_numbers_train.npy"),
                                                             load("labels_numbers_train.npy"))

    for dimension in dimensions:
        print("The Decision Tree are training with " + str(dimension) + " elements")

        # Balancing
        images_train, labels_train = images_train_init[0:dimension:], labels_train_init[0:dimension]

        digits_clf =  tree.DecisionTreeClassifier()
        digits_clf.fit(images_train, labels_train)

        test_accuracy.append(
            accuracy_score(load("labels_numbers_test.npy"), digits_clf.predict(load("images_numbers_test.npy"))))
        print(str(test_accuracy))

    # train_accuracy.append(
    #    accuracy_score(load("labels_numbers_train.npy"), digits_clf.predict(load("images_numbers_train.npy"))))

    plt.plot(dimensions, test_accuracy, marker="o", color='blue')
    # plt.plot(dimensions, train_accuracy, marker="o", color='green', lable='TrainAccuracy')
    # plt.legend(loc='upper left')
    plt.title("Decision Tree Performance")
    plt.xlabel("Training-Set Dimension")
    plt.ylabel("Test Accuracy")
    plt.show()

def letters_test():
    dimensions = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 124800]
    test_accuracy = []
    train_accuracy = []

    print("Letters classifier")

    # Extract Dataset
    '''
    X_train, y_train = extract_training_samples('letters')
    X_test, y_test = extract_test_samples('letters')
    '''

    # Reshape Dataset
    '''
    imgs_train, labels_train = manage_data_set(124800, X_train, y_train)
    imgs_test, labels_test = manage_data_set(20800, X_test, y_test)
    '''

    # Save reshape Dataset
    '''
    save("images_letters_train.npy", imgs_train)
    save("labels_letters_train.npy", labels_train)
    save("images_letters_test.npy", imgs_test)
    save("labels_letters_test.npy", labels_test)
    '''

    # Extract the reshape Dataset

    images_train_init, labels_train_init = balance_data_set(load("images_letters_train.npy"),load("labels_letters_train.npy"))

    for dimension in dimensions:
        print("The Decision Tree are training with " + str(dimension) + " elements")

        images_train, labels_train = images_train_init[0:dimension:], labels_train_init[0:dimension]

        # training with numbers dataset
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(images_train, labels_train)

        # testing dataset
        test_accuracy.append(
            accuracy_score(load("labels_letters_test.npy"), clf.predict(load("images_letters_test.npy"))))
        train_accuracy.append(
            accuracy_score(load("labels_letters_train.npy"), clf.predict(load("images_letters_train.npy"))))

        print(str(test_accuracy))

    plt.plot(dimensions, test_accuracy, marker="o", color='blue', label='TestAccuracy')
    plt.plot(dimensions, train_accuracy, marker="o", color='green', label='TrainAccuracy')
    plt.legend(loc='upper left')
    plt.title("Decision tree Performance in letters classifications")
    plt.xlabel("Training-Set Dimension")
    plt.ylabel("Test Accuracy")

    plt.show()


letters_test()
