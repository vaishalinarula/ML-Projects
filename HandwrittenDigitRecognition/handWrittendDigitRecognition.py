from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def checkAccuracy(test_X, test_Y, clf):
    predict_Y = clf.predict(test_X)
    print("Accuracy of our model is: ", 100*accuracy_score(test_Y, predict_Y))

def predictValue(test_X, test_Y, clf):
    val = int(input("Enter any value between (1 to 1000) for demo of our model : "))
    digit = test_X[val]
    digit_display = digit.reshape(28, 28)
    plt.imshow(digit_display, interpolation = "nearest")
    plt.show()
    print("Your handwritten digit image is recognizing... ")
    print("Digit is : ", clf.predict([test_X[val]]))
    isCheckAccuracy = int(input("Press 1 for checking accuracy of our model else Press 0 : "))
    if isCheckAccuracy == 1:
        checkAccuracy(test_X, test_Y, clf)
    else:
        print("Thank You for using our model !")


def selectClassifier(train_X, train_Y):
    print("Now, Select any of the given classifier: ")
    print("Press 1 for Support Vector Machine and Press 2 for K-nearest Neighbour: ")
    clacfyr = int(input())
    if clacfyr == 1:
        clf = SVC()
        clf.fit(train_X, train_Y)
    else:
        clf = KNeighborsClassifier()
        clf.fit(train_X, train_Y)
    return clf


def fetchData():
    print("Please wait, dataset is fetching from MNIST dataset.")
    mnist = fetch_openml('mnist_784')
    print("Dataset has been fetched!")
    print("Please wait, dataset is splitting in training and testing data.")
    x = mnist['data']
    y = mnist['target']
    train_X = x[:6000]                          # training data
    train_Y = y[:6000]                          # training data
    test_X = x[6000:7000]                       # testing data
    test_Y = y[6000:7000]                       # testing data
    print("Dataset has been split.")
    clf = selectClassifier(train_X, train_Y)
    predictValue(test_X, test_Y, clf)

def main():
    print("Hey, this is Handwritten Digit Recognition project : ")
    print("Press Enter to continue: ")
    input()
    fetchData()

if True:
    main()
