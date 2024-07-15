# Boston Housing Price Prediction

### Housing Price in Boston
The medv variable is the target variable.

### Data Description
The Boston data frame has 506 rows and 14 columns.
This data frame contains the following columns:

- crim: per capita crime rate by town

- zn: proportion of residential land zoned for lots over 25,000 sq.ft

- indus: proportion of non-retail business acres per town

- chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

- nox: nitrogen oxides concentration (parts per 10 million)

- rm: average number of rooms per dwelling

- age: proportion of owner-occupied units built prior to 1940

- dis: weighted mean of distances to five Boston employment centres

- rad: index of accessibility to radial highways

- tax: full-value property-tax rate per \$10,000

- ptratio: pupil-teacher ratio by town

- black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

- lstat: lower status of the population (percent)

- medv: median value of owner-occupied homes in \$1000s


## Data preprocessing
- remove 'ID'
- check for missing data
- no 'medv' in testing data

## One-hot encoding
Transform 'chas' into 'chas_0' and 'chas_1', and under each new feature, use 0 and 1 to represent the presence or absence of the category

## Standardization
Using the mean and standard deviation to standardize the data, making the numerical values of each data item fall within the range of 0 to 1

## Bagging
From the training dataset, extract K samples, then train K classifiers (trees in this case) using these K samples. Each time, the K samples are put back into the population, so there is some data overlap among these K samples. However, because the samples for each tree are still different, the trained classifiers (trees) have diversity. The final result is obtained by majority vote, where each classifier has equal weight

## Training
Split data to 60% for trianig and 40% for validation, randomly

### Linear Regression

Linear regression is a regression analysis method that models the relationship between one or more independent variables and a dependent variable using the least squares function, known as the linear regression equation.

### Random Forest

1. Define a random sample of size n (here, using the bagging method), which involves randomly selecting n data points from the dataset with replacement

2. From the selected n data points, train a decision tree. For each node:
  - Randomly select d features
  - Use these features to split the node based on information gain
3. Repeat steps 1 to 2 k times

4. Aggregate the predictions of all decision trees and determine the classification result by majority vote

