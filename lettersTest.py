from  processesDataSet import *
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from numpy import load

def lettersTest():
    dimensions = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 124800]
    test_accuracy = []
    train_accuracy = []

    print("Letters classifier")
    saveDataSet('letters')

    # Extract the reshape Dataset
    images_train_init, labels_train_init = balanceDataSet(load("images_letters_train.npy"),
                                                          load("labels_letters_train.npy"))

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

    plt.plot(dimensions, test_accuracy, marker="o", color='blue', label='TestAccuracy')
    plt.plot(dimensions, train_accuracy, marker="o", color='green', label='TrainAccuracy')
    plt.legend(loc='upper left')
    plt.title("Decision tree Performance in letters classifications")
    plt.xlabel("Training-Set Dimension")
    plt.ylabel("Test Accuracy")

    plt.show()

lettersTest()