from svm_basic import svm_basic

def get_data(filename):
    datamatrix = []
    labelmatrix = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        datamatrix.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelmatrix.append(int(lineArr[2]))
    return datamatrix, labelmatrix

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def plot_fit(fit_line, datamatrix, labelmatrix):
    import matplotlib.pyplot as plt
    import numpy as np

    weights = fit_line
    print(len(weights))
    dataarray = np.asarray(datamatrix)
    n = dataarray.shape[0]

    # Keep track of the two classes in different arrays so they can be plotted later...
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelmatrix[i]) == 1:
            xcord1.append(dataarray[i, 1])
            ycord1.append(dataarray[i, 2])
        else:
            xcord2.append(dataarray[i, 1])
            ycord2.append(dataarray[i, 2])
    fig = plt.figure()

    # Plot the data as points with different colours
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    # Plot the best-fit line
    x = np.arange(-1.0, 6.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    print(y)
    ax.plot(x, y)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def accuracy(labels, hypotheses):
    count = 0.0
    correct = 0.0

    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == h:
            correct += 1.0
    return correct / count

def print_confusion_matrix(labels, hypotheses):
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    count = 1.0
    for l, h in zip(labels, hypotheses):
        count += 1.0
        if l == 1 and h == 1:
            tp += 1.0
        elif l == 1 and h == -1:
            fn += 1.0
        elif l == -1 and h == -1:
            tn += 1.0
        else:
            fp += 1
    print ('-----------------------------')
    print ('\tConfusion Matrix')
    print ('-----------------------------')
    print ('\t\tPredicted')
    print ('\tActual\tNO\tYES')
    print ('-----------------------------')
    print ('\tNO\t', tn, '\t', fp)
    print ('-----------------------------')
    print ('\tYES\t', fn, '\t', tp)
    print ('-----------------------------')
    
X, Y = get_data('linearly_separable_train.csv')
dataArr, labelArr = loadDataSet('linearly_separable_train.csv')

clf = svm_basic()
w = clf.fit(dataArr, labelArr)
print ('Weights:', w)
plot_fit(w, X, Y)

verify_x, verify_y = get_data('linearly_separable_verify.csv')
hypotheses = clf.predict(verify_x)

print ('Accuracy:', accuracy(verify_y, hypotheses))

print_confusion_matrix(verify_y, hypotheses)

