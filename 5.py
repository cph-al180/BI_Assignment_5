import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pylab import polyfit, poly1d

traindata = 'training2.json'
testdata = 'testing.json'
  
def getData():
    global training_data
    global testing_data
    with open(traindata) as json_data_1:
        training_data = json.load(json_data_1, encoding = 'ISO-8859-1')
    with open(testdata) as json_data_2:
        testing_data = json.load(json_data_2, encoding = 'ISO-8859-1')

def formatData():
    global train_karma
    global train_created
    global test_karma
    global test_created
    global TRAIN_CREATED
    
    train_karma = []
    train_created = []
    test_karma = []
    test_created = []
    TRAIN_CREATED = []
    
    for i in training_data:
        train_karma.append(i["karma"])
        train_created.append(i["created"])
        
    for j in testing_data:
        test_karma.append(j["karma"])
        test_created.append(j["created"])
        
    TRAIN_CREATED = np.array([train_created])
    TRAIN_CREATED = TRAIN_CREATED.T    

#Part 1
def trainAndPlot():
    global model
    
    X,y = TRAIN_CREATED, train_karma
    x = train_created
    
    model = linear_model.LinearRegression()
    model.fit(X, y)
    
    fit = np.polyfit(x, y, deg=1)
    fit_fn = np.poly1d(fit)
    plt.plot(X, y, 'ro', X, fit_fn(X), 'b')
    plt.show()

def run():
    getData()
    formatData()
    trainAndPlot()
    print('done')

run()
