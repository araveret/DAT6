#############################################
#
#     CLASSIFICATION & REGRESSION TREES 
#                 (CART)
#
#############################################


# ---------------------------------------------------------------
# ---------------- NOTES FOR NEW R USERS ------------------------
# ---------------------------------------------------------------
#    R was built by Data Scientists for Data Scientists. 
#    R "thinks" in terms of Data Frames / Matrices.
#    R has a lot more built-in Data Science functionality than Python (e.g., ols(), read.csv())
#    R indexes from 1, not 0
#    R help is called with ?<function name> (e.g., ?str)
#    Much of R data-wrangling is similar to interacting with Pandas Dataframes (e.g., Logical Filtering)
#    R has many packages, 5000+, so often there are many packages to do the same function
# ---------------------------------------------------------------


# ===========================================
#     Set up the workspace and get the data
# ===========================================

rm(list=ls()); gc()     # clear the workspace
set.seed(973487)        # Ensures you can repeat the results
library(rpart)          # For creating the tree
setwd("C:/Users/josdavis/Documents/Personal/GitHub/DAT6/data")

# Get the data
data <- read.csv("titanic.csv", header = TRUE)

# Split into training and testing sets
idxs <- runif(nrow(data)) < 0.7   # Random Indices
train <- data[idxs, ]             # Training set
test  <- data[!idxs, ]            # Testing set
rm(idxs, data)

# summarize the data
summary(train)
str(train)
head(train, 7)

# ===========================================
#       Create the tree
# ===========================================

tree <- rpart(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch, 
              data = train, 
              method = "class")

# View the tree
tree

# View the details of each node
summary(tree)

# View the importance scores (avg. decrease in gini coefficient)
tree$variable.importance

# ===========================================
#       Plot the tree
# ===========================================

# Simple plot is ugly and uninformative
plot(tree)
text(tree)

# Good quick alternative is to convert the rpart object to a binary tree 
# using the partykit package
library(partykit)  
plot(as.party(tree))

# This package provides some additional options for plotting
library(rpart.plot)
prp(tree) 
# Check out documention (?prp) for more plotting options
# http://www.milbo.org/rpart-plot/prp.pdf is a very thorough user manual for prp()

# The 'rattle' package allows for some pretty plotting
library(rattle) 
fancyRpartPlot(tree)

# ===========================================
#       Control the parameters of the tree
# ===========================================

# The control argument allows you to limit how large the tree grows
# For example: minsplit = 30 stops splitting once a node has 30 or less data points
tree <- rpart(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch, 
              data = train,
              method = "class",
              control = rpart.control(minsplit = 30))

# Another example: maxdepth = 4 limits the depth of the tree to 4 levels (inlcuding terminal node)
tree <- rpart(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch, 
              data = train,
              method = "class",
              control = rpart.control(maxdepth = 4))

# See the documentation for default values and more options
?rpart.control

# Remove records with missing response or ALL missing inputs (DEFALUT)
tree <- rpart(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch, 
              data = train,
              method = "class",
              na.action = na.rpart)

# Missing values (remove rows with any missing values)
tree <- rpart(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch, 
              data = train,
              method = "class",
              na.action = na.omit)

# ===========================================
#       Evaluate the accuracy of the tree
# ===========================================

# Generate predictions (both probabilities and class predictions)
test$predict_proba <- predict(tree, type = "prob", newdata = test)[,2]
test$prediction <- predict_proba > 0.5

# Acccuracy in terms of classification rate (with 0.5 threshhold)
sum(test$prediction == test$survived) / nrow(test)

# Confusion Matrix (rows are predictions, colums are actuals)
table(test$prediction, test$survived)
prop.table(table(test$prediction, test$survived), 2)

# Sensitivity: When the person survived, how often did it predict survival?
# A.K.A. True Positive Rate
test_lived = test[test$survived,]
sum(test_lived$prediction == test_lived$survived) / nrow(test_lived)

# Specificty: When the person died, how often did it predict death?
# A.K.A. True Negative Rate
test_died = test[!test$survived,]
sum(test_died$prediction == test_died$survived) / nrow(test_died)