import numpy as np
from scipy import stats
import sklearn

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    # test case 1
    np.random.seed(0) 
    X_train = [[val] for val in range(100)] # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train] # 1D

    lin_reg = MySimpleLinearRegressionClassifier(None)
    lin_reg.fit(X_train, y_train)

    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)

    # assert against sklearn
    assert np.isclose(lin_reg.slope, sklearn_lin_reg.coef_[0])
    assert np.isclose(lin_reg.intercept, sklearn_lin_reg.intercept_)

    # test case 2
    np.random.seed(0)
    X_train = [[val] for val in range(500)] # 2D
    y_train = [row[0] / 2 + np.random.normal(0,50) for row in X_train] # 1D

    if(y_train[0] >= 250):
        y_train[0] = 1
    elif(y_train[0] < 250):
        y_train[0] = 0

    lin_reg = MySimpleLinearRegressionClassifier(None)
    lin_reg.fit(X_train, y_train)

    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)

    # assert against sklearn
    assert np.isclose(lin_reg.slope, sklearn_lin_reg.coef_[0])
    assert np.isclose(lin_reg.intercept, sklearn_lin_reg.intercept_)

def test_simple_linear_regression_classifier_predict():
    # test case 1
    np.random.seed(0) 
    X_train = [[val] for val in range(100)] # 2D
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train] # 1D
    X_test = [[150], [-150], [0], [50], [1000]]

    for index, y in enumerate(y_train):
        if y >= 100:
            y_train[index] = 1 # 1 for high
        elif y < 100:
            y_train[index] = 0 # 0 for low

    lin_reg = MySimpleLinearRegressionClassifier(None)
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)

    # assert against sklearn
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)

    assert np.allclose(y_predicted, sklearn_y_predicted)

    # test case 2
    np.random.seed(0)
    X_train = [[val] for val in range(500)] # 2D
    y_train = [row[0] / 2 + np.random.normal(0,50) for row in X_train] # 1D
    X_Test = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    for index, y in enumerate(y_train):
        if y >= 250:
            y_train[index] = 1 # 1 for high
        elif y < 250:
            y_train[index] = 0 # 0 for low

    lin_reg = MySimpleLinearRegressionClassifier(None)
    lin_reg.fit(X_train, y_train)
    y_predicted = lin_reg.predict(X_test)

    # assert against sklearn
    sklearn_lin_reg = LinearRegression()
    sklearn_lin_reg.fit(X_train, y_train)
    sklearn_y_predicted = sklearn_lin_reg.predict(X_test)

    assert np.allclose(y_predicted, sklearn_y_predicted)

def test_kneighbors_classifier_kneighbors():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test_class_example1 = [.33, 1]
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_class_example2 = [2.6, 3]

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]    
    X_test_bramer_example = [11.5, 9.5]

    # test case 1
    kNN = MyKNeighborsClassifier(n_neighbors = 3)
    kNN.fit(X_train_class_example1, y_train_class_example1)
    myDistances, myIndices = kNN.kneighbors(X_test_class_example1)
    
    sklearn_kNN = KNeighborsClassifier()
    sklearn_kNN.fit(X_train_class_example1, y_train_class_example1)
    sklearn_distances, sklearn_indices = sklearn_kNN.kneighbors([X_test_class_example1], n_neighbors = 3)

    assert np.allclose(myDistances, sklearn_distances)
    assert np.allclose(myIndices, sklearn_indices)

    # test case 2
    kNN = MyKNeighborsClassifier(n_neighbors = 3)
    kNN.fit(X_train_class_example2, y_train_class_example2)
    myDistances, myIndices = kNN.kneighbors(X_test_class_example2)

    sklearn_kNN = KNeighborsClassifier()
    sklearn_kNN.fit(X_train_class_example2, y_train_class_example2)
    sklearn_distances, sklearn_indices = sklearn_kNN.kneighbors([X_test_class_example2], n_neighbors = 3)

    assert np.allclose(myDistances, sklearn_distances)
    assert np.allclose(myIndices, sklearn_indices)

    # test case 3
    kNN = MyKNeighborsClassifier(n_neighbors = 3)
    kNN.fit(X_train_bramer_example, y_train_bramer_example)
    myDistances, myIndices = kNN.kneighbors(X_test_bramer_example)

    sklearn_kNN = KNeighborsClassifier()
    sklearn_kNN.fit(X_train_bramer_example, y_train_bramer_example)
    sklearn_distances, sklearn_indices = sklearn_kNN.kneighbors([X_test_bramer_example], n_neighbors = 3)

    assert np.allclose(myDistances, sklearn_distances)
    assert np.allclose(myIndices, sklearn_indices)

def test_kneighbors_classifier_predict():
    # from in-class #1  (4 instances)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test_class_example1 = [3, 2]
    example_1_solution = "bad"
    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test_class_example2 = [2.6, 3]
    example_2_solution = "no"
    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]    
    X_test_bramer_example = [11.5, 9.5]
    example_3_solution = "+"
    # test case 1
    kNN = MyKNeighborsClassifier(n_neighbors = 3)
    kNN.fit(X_train_class_example1, y_train_class_example1)
    kNN.kneighbors(X_test_class_example1)
    myPredictions = kNN.predict(X_test_class_example1)

    assert np.array_equal(myPredictions[0], example_1_solution)
   
    # test case 2
    kNN = MyKNeighborsClassifier(n_neighbors = 3)
    kNN.fit(X_train_class_example2, y_train_class_example2)
    kNN.kneighbors(X_test_class_example2)
    myPredictions = kNN.predict(X_test_class_example2)

    assert np.array_equal(myPredictions[0], example_2_solution)
    
    # test case 3
    kNN = MyKNeighborsClassifier(n_neighbors = 3)
    kNN.fit(X_train_bramer_example, y_train_bramer_example)
    kNN.kneighbors(X_test_bramer_example)
    myPredictions = kNN.predict(X_test_bramer_example)

    assert np.array_equal(myPredictions[0], example_3_solution)


def test_dummy_classifier_fit():
    #test case 1
    X_train = [i for i in range(100)]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7,0.3]))

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "yes"

    #test case 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "no"

    #case 3
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.8, 0.1, 0.1]))

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "yes"

def test_dummy_classifier_predict():
     #test case 1
    X_train = [i for i in range(100)]
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7,0.3]))
    X_Test = [10, 20]
    solution = ["yes", "yes"]

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    predictions = dummy.predict(X_Test)

    assert np.array_equal(predictions, solution)

    #test case 2
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    solution = ["no", "no"]

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train) 
    predictions = dummy.predict(X_Test)

    assert np.array_equal(predictions, solution)
    #case 3
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.8, 0.1, 0.1]))
    solution = ["yes", "yes"]

    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)
    predictions = dummy.predict(X_Test)

    assert np.array_equal(predictions, solution)

   
