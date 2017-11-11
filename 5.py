import json
import matplotlib.pyplot as plt
from sklearn import linear_model

traindata = 'training.json'
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
    train_karma = []
    train_created = []
    test_karma = []
    test_created = []
    for i in training_data:
        train_karma.append(i["karma"])
        train_created.append(i["created"])
    for j in testing_data:
        test_karma.append(j["karma"])
        test_created.append(j["created"])

def train():
    x,y = train_created, train_karma
    regr = linear_model.LinearRegression()
    regr.fit(x, y)

def run():
    getData()
    formatData()
    train()
    print('done')

run()



    

