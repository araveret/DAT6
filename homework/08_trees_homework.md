## Class 8 Homework: Trees and Ensembles

You should try to build upon the ideas we implemented in class and create your own [kaggle submission](https://www.kaggle.com/c/titanic-gettingStarted).

* Ideas:
    * Try to come up with a more sophisticated method for filling in missing age values. For example, use the mean age by class and gender instead of the mean overall age.
    * Tune across a different parameter besides the one presented in class.
    * Try to tune across two tuning parameters at the same time.
	* Create the compliment of the Spouse Variable (e.g., Sibling Variable)
    * See if you can split apart Parents and Children variable into two variables. If you make assumptions, make sure to state them explicitly.
    * Look up the formula for entropy. Change the splitting criterion to entropy. Does this change the results?
	* Look at the documentation for [Extremely Randomized Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier)
	* **Advanced:** Create your own ensemble with three different models, a Random Forest, KNN, and Logistic Regression. Experiment with different ways of combining them together (e.g., majority vote, average, stacking). Does this appear to improve accuracy? Why or why not?
