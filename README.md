# Business Assignment 5 - Machine Learning

## Part 1   

![Scatter Plot of Data](HackerNewsPlot.png.png)
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
## Part 2

## Part 3

## Part 4

## Part 5
