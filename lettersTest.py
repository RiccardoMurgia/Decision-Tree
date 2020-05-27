from processesDataSet import *
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from numpy import load


def lettersTest():
    dimensions = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 124800]
    testAccuracy = []
    trainAccuracy = []

    print("Letters classifier")
    '''saveDataSet('letters')'''

    
    images_train_init, labels_train_init = balanceDataSet(load("images_letters_train.npy"),
                                                          load("labels_letters_train.npy"))

    for dimension in dimensions:
        print("The Decision Tree is training with " + str(dimension) + " elements")

        images_train, labels_train = images_train_init[0:dimension:], labels_train_init[0:dimension]

         
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(images_train, labels_train)

        
        testAccuracy.append(
            accuracy_score(load("labels_letters_test.npy"), clf.predict(load("images_letters_test.npy"))))
        trainAccuracy.append(
            accuracy_score(load("labels_letters_train.npy"), clf.predict(load("images_letters_train.npy"))))

    plt.plot(dimensions, testAccuracy, marker="o", color='blue', label='TestAccuracy')
    plt.plot(dimensions, trainAccuracy, marker="o", color='green', label='TrainAccuracy')
    plt.legend(loc='upper left')
    plt.title("Decision tree Performance in letters classifications")
    plt.xlabel("Training-Set Dimension")
    plt.ylabel("Accuracy")

    plt.savefig('LettersAccuracy.png')
    plt.show()


lettersTest()
