'''
Linear Regression
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import train_test_split

'''
Exploring the Data
'''

# Dataset: Sales and advertising channels
adv = pd.read_csv('../data/Advertising.csv')
adv.head()

# Split data into train and test
train, test = train_test_split(adv, test_size=0.3, random_state=1)

# Convert them back into dataframes, for convenience
train = pd.DataFrame(data=train, columns=adv.columns)
test = pd.DataFrame(data=test, columns=adv.columns)

# Plot the Radio spending against the Sales
plt.figure(figsize=(10,9))
plt.subplot(221)
plt.scatter(train.Radio, train.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("Radio"); plt.ylabel("Sales")

# Plot the Newspaper spending against the Sales
plt.subplot(222)
plt.scatter(train.Newspaper, train.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("Newspaper"); plt.ylabel("Sales")

# Plot the TV spending against the Sales
plt.subplot(223)
plt.scatter(train.TV, train.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("TV"); plt.ylabel("Sales")

# Plot the Region against the Sales
plt.subplot(224)
plt.scatter(train.Region, train.Sales, alpha=0.7)  # Plot the raw data
plt.xlabel("Region"); plt.ylabel("Sales")

'''
ESTIMATING THE COEFFICIENTS
'''

# Fit a linear regression model using OLS
from sklearn.linear_model import LinearRegression
slm = LinearRegression()
slm.fit(train['Radio'][:,np.newaxis], train['Sales'])

# Evaluate the output
slm.intercept_
slm.coef_

'''
EXERCISE:
1) Given coefficient estimates, predict the y-value for train.Radio.min()
& train.Radio.max()
2) Create a scatter plot that also shows the data and the line of best fit

Hint: Use the following convention plt.plot([x_min, x_max], [y_min, y_max])
'''
# Calculate the ymin, ymax
ymin = slm.intercept_ + slm.coef_ * train.Radio.min()
ymax = slm.intercept_ + slm.coef_ * train.Radio.max()

# Plotting
plt.plot([train.Radio.min(), train.Radio.max()], [ymin, ymax])
plt.scatter(train.Radio, train.Sales)

# This is equivalent
slm.predict(train.Radio.min())
slm.predict(train.Radio.max())

'''
Model Evaluation
'''
# Evaluate the fit of the model based off of the training set
preds = slm.predict(test['Radio'][:,np.newaxis])
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test['Sales'],preds))

# Evaluate the model fit based off of cross validation
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(slm, adv['Radio'][:,np.newaxis], adv['Sales'], cv=5, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

'''
EXERCISE:
1) Run a multiple regression with Radio and TV. 
        Which coefficient has a higher value?
        What does this suggest practically?
2) Calculate the 5-fold CV RMSE. Is it better or worse than before?
'''

# 1) 
mlm = LinearRegression()
mlm.fit(train[['Radio', 'TV']], train['Sales'])
mlm.intercept_
mlm.coef_

for i in zip(['Radio', 'TV'], mlm.coef_):
    print i[0], np.round(i[1], 4)

# 2)
preds = mlm.predict(test[['Radio', 'TV']])
scores = cross_val_score(mlm, adv[['Radio', 'TV']], adv['Sales'], cv=5, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))

'''
Common Transformations
'''


'''
INTERACTION TERMS
What: Interaction measure the combined effect of two variables acting together
Why:  This is a way to make predictions and inferences when 
two input variables are correlated with each other.
Notes: Whenever you include an interaction terms, it is conventional to 
include the main effects as well.
'''

train['Radio_TV'] = train['Radio'] * train['TV']
lmi = LinearRegression()
lmi.fit(train[['TV', 'Radio', 'Radio_TV']], train['Sales'])
lmi.coef_ 

'''
DUMMY VARIABLES
What: Get k -1 binary "dummy" variables to represent all possible combinations of a
categorical variable (k is levels in the categorical variable).
Why: It's the standard way to represent non-ordinal categorical data for Linear Regression.
'''
# Categorical Variables
# create three dummy variables using get_dummies, then exclude the 1st dummy column
region_dummies = pd.get_dummies(adv.Region, prefix='Region').iloc[:, 1:]

# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)
adv = pd.concat([adv, region_dummies], axis=1)
adv.head()

'''
LOG TRANSFORM:
What: Take the logarithm of one or more of your variables (in this case the output).
Why: If the underlying distribution is lognormal or if the residuals are heteroskedastic.
'''

# Return to the case of the simple Linear regression
# Fit a multiple linear regression model using OLS
slm = LinearRegression()
slm.fit(train['TV'][:,np.newaxis], train['Sales'])

# Plot the residuals across the range of predicted values for the lin-lin model
pred = slm.predict(test['TV'][:,np.newaxis])
resid = pred - test['Sales']
plt.scatter(pred, resid, alpha=0.7)
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")

# Log Transformation (e.g., for when your data, heteroskedastic and positive, or log-linear)
sllm = LinearRegression()
sllm.fit(train['TV'][:,np.newaxis], np.log(train['Sales']))
sllm.coef_ 

# For a unit change in x, the beta coefficient corresponds to the % change in y
sllm.coef_ * 100
(np.exp(sllm.predict(30)) - np.exp(sllm.predict(29))) / np.exp(sllm.predict(29))*100

'''
Overview of the StatsModels package

The StatsModels packages has some nice features for linear modeling. However
we will be using the scikit-learn package for much of the remainder of 
the course because of its extensibility to other techniques.
'''

import statsmodels.formula.api as smf

# Create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=adv).fit()

# Print the coefficients
lm.params

# Print the confidence intervals for the model coefficients
lm.conf_int()

# Print the p-values for the model coefficients
lm.pvalues

# Print a summary of the fitted model
lm.summary()

# Include interaction terms
lm = smf.ols(formula='Sales ~ TV + Radio*Newspaper', data=adv).fit()
lm.summary()

# Include dummy variables
lm = smf.ols(formula='Sales ~ TV + C(Region)', data=adv).fit()
lm.summary()

# Include the log response
lm = smf.ols(formula='np.log(Sales) ~ TV', data=adv).fit()
lm.summary()


'''
Regression by Hand - Toy DataSet
'''

import numpy as np

# Create a toy data set
X0 = [1, 1, 1, 1, 1]
X1 = [5, 1, 23.5, 5, 1.512]
X2 = [4, 5.1, 7, 8.7, 9]

X = np.matrix([X0, X1, X2]).T
Y = np.matrix([10.0, 33.1, 5.7, 200, 105.3]).T

# Estimate coefficients with matrix algebra
beta_hat = (X.T * X).I * X.T * Y
beta_hat

# Estimate coefficients with statsmodels
import statsmodels.api as sm
est = sm.OLS(Y, X).fit()
est.params

# Estimate coefficients with scikit learn
est = LinearRegression(fit_intercept=False).fit(X,Y)
est.coef_

'''
Regression by Hand - Advertising DataSet
'''
adv['constant'] = 1
X = np.matrix(adv[['constant', 'TV', 'Radio', 'Newspaper']])
Y = np.matrix(adv['Sales']).T

# Estimate coefficients with matrix algebra
beta_hat = (X.T * X).I * X.T * Y
beta_hat

# Estimate coefficients with statsmodels
est = sm.OLS(Y, X).fit()
est.params

# Estimate coefficients with scikit learn
est = LinearRegression(fit_intercept=False).fit(X,Y)
est.coef_