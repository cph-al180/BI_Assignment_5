# Business Intelligence Assignment 5 - Machine Learning

## Part 1   - Training Model and Scatter Plot
Write 4 lines about a and b  
![Scatter Plot of Data](HackerNewsPlot.png)  
`a: -4.3054089103938141e-05`  
`b: 63661.886865183958`  
```python
def trainAndPlot():
    X,y = TRAIN_CREATED, train_karma
    x = train_created
    model = linear_model.LinearRegression()
    model.fit(X, y)
    fit = np.polyfit(x, y, deg=1)
    fit_fn = np.poly1d(fit)
    plt.plot(X, y, 'ro', X, fit_fn(X), 'b')
    plt.show()
    print("a: ", model.coef_[0])
    print("b: ", model.intercept_)
```
## Part 2 - Mean Absolute Error
Write about MAE  
`MAE (training): 4535.2278195244253`   
`MAE (testing): 4363.9936837520208`  
```python
def calcMAE():
    train_karma_pred = model.predict(TRAIN_CREATED)
    test_karma_pred = model.predict(TEST_CREATED)
    
    train_MAE = mean_absolute_error(train_karma, train_karma_pred)
    test_MAE = mean_absolute_error(test_karma, test_karma_pred)
    
    print('Train MAE: ', train_MAE)
    print('Test MAE: ', test_MAE)
```
## Part 3 - Mean Squared Error

## Part 4 - Pearson's r

## Part 5 - Predictions
