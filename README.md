# Business Intelligence Assignment 5 - Machine Learning

## Part 1   - Training Model and Scatter Plot
`y = ax + b`  
What does the `a` value mean: The `a` value is the value of `y` when `x` is equal to 0. If `x` is never zero, then this value is not relevant.   
What does the `b` value mean: `b` is the coeffecient of `x`. It determines how much `y` changes for each one-unit change to `x`.  
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
The result numbers tell us that the average difference between the data-set containing actual data (training and testing), and the data-set containing predicted karma is about `4535`.  
The difference between the 2 results; `4535` and `4363` comes due to them containing different data. If both data-sets were identical, then the results would also be.  
  
`MAE (training): 4535.2278195244253`   
`MAE (testing): 4363.9936837520208`  
```python
def calcMAE():
    train_karma_pred = model.predict(TRAIN_CREATED)
    test_karma_pred = model.predict(TEST_CREATED)
    train_MAE = mean_absolute_error(train_karma, train_karma_pred)
    test_MAE = mean_absolute_error(test_karma, test_karma_pred)
```
## Part 3 - Mean Squared Error  
The MSE results that I've ended up with are very confusing to me, considering they're supposed to be between `0` and `1` (best to worst), and I've been unable to figure out why they don't make sense.  
  
`MSE (training): 10230.170612147558`   
`MSE (testing): 7858.119559984959`  
```python
def calcMSE():
    train_karma_pred = model.predict(TRAIN_CREATED)
    test_karma_pred = model.predict(TEST_CREATED)
    train_MSE = mean_squared_error(train_karma, train_karma_pred)
    test_MSE = mean_squared_error(test_karma, test_karma_pred)
	train_MSE = math.sqrt(train_MSE)
    test_MSE = math.sqrt(test_MSE)
```
## Part 4 - Pearson's r
Pearson's r value describes the correlation between 2 variables, which in our case is `created`and `karma`. The result is between `-1` and `1`. This value is representated by the linear regression line in the scatter plot image. Pearson's r is different from MAE and MSE because it calculates the correlation between `x`and `y` values throughout an entire data-set, and does not take into account any predicted values.  
  
`r (training): -0.35941580366452558`   
`r (testing): -0.36569769103632976`  
```python
def calcPR():
    train_PR = pearsonr(train_created, train_karma)
    test_PR = pearsonr(test_created, test_karma)
```
## Part 5 - Predictions
We can use the model to find out how long it would take to reach `1000` karma, which is about `1454813038`. I couldn't think of another way to calculate that number, without creating a new model, other than using `model.predict()` in a python shell untill the result was close to `1000`. I am sure there is a better way and more accurate way to achieve the most fitting result.  
Since of all of the prediction calculations are based on the linear model, which is based purely on the json data we were provided, the best way that I can think of, is to feed it even more data. There are some users in the data-set which heavily deviate from the average user, which could scew the results.
