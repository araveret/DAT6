## DAT6 Course Repository

Course materials for [General Assembly's Data Science course](https://generalassemb.ly/education/data-science/washington-dc/) in Washington, DC (2/21/14 - 5/2/15). View student work in the [student repository](https://github.com/sinanuozdemir/DAT6-students).

**Instructors:** Sinan Ozdemir and Josiah Davis.

**Office hours:** 5-7pm on Tuesday and 5-7pm on Saturday at General Assembly

**Machine Learning Overview:** See [here](images/ML.jpg) for an overview of the machine learning models! Please only use this as a *broad* general overview and shouldn't be taken as law. There are some generalizations that I am making.

**[Course Project information](project.md)**

Saturday | Topic | Project Milestone
--- | --- | ---
2/21:  | [Introduction / Pandas](#class-1-introduction-and-pandas)
2/28:| [Git(hub) / Getting Data](#class-2-github-and-getting-data) | 
3/7:| [Advanced Pandas / Machine Learning](#class-3-advanced-pandas-and-machine-learning) | [One Page Write-up with Data](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#march-7-one-page-write-up-with-data)
π  == τ/2 day  | [Model Evaluation / Logistic Regression](#class-4-model-evaluation-and-logistic-regression) | 
3/21: | [Linear Regression](#class-5-linear-regression) | [2-3 Minute Presentation](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#march-21-2-3-minute-presentation) 
3/28: | [Data Problem / Clustering and Visualization](#class-6-data-problem-and-clustering-and-visualization) | 
4/2 | Naive Bayes / Natural Language Processing | [Deadline for Topic Changes](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#april-2-deadline-for-topic-changes)
4/11 | Decision Trees / Ensembles | [First Draft Due (Peer Review)](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#april-11-first-draft-due-peer-review)
4/18 | PCA / Databases / MapReduce | 
4/25 | Recommendation Engines | 
5/2 | Project Presentations | [Presentation](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#may-2-presentation)


### Installation and Setup
* Install the [Anaconda distribution](http://continuum.io/downloads) of Python 2.7x.
* Install [Git](http://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and create a [GitHub](https://github.com/) account.
* Once you receive an email invitation from [Slack](https://slack.com/), join our "DAT6 team" and add your photo!

------
### Class 1: Introduction and Pandas
**Agenda:**
* Introduction to General Assembly
* Course overview: our philosophy and expectations ([slides](slides/01_course_overview.pdf))
* Data science overview ([slides](slides/01_intro_to_data_science.pdf))
* Data Analysis in Python ([code](code/01_pandas.py))
* Tools: check for proper setup of Anaconda, overview of Slack

**Homework:**
* [Pandas Homework](homework/01_pandas_homework.py)

**Optional:**
* Review your base python ([code](code/00_base_python_refresher.py))

------
### Class 2: Git(hub) and Getting Data

**Agenda:**
* Github: ([slides](slides/02_git_github.pdf))
* Getting Data ([slides](slides/02_getting_data.pdf))
* Regular Expressions ([code](code/02_re_example.py))
* Getting Data ([code](code/02_getting_data.py))

**Homework:**
* Complete the first [Project Milestone](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#march-7-one-page-write-up-with-data) (Submit on the [Dat6-students](https://github.com/sinanuozdemir/DAT6-students) repo via a pull request)

**Resources:**
* [Forbes: The Facebook Experiment](http://www.forbes.com/sites/dailymuse/2014/08/04/the-facebook-experiment-what-it-means-for-you/)
* [Hacking OkCupid](http://www.wired.com/2014/01/how-to-hack-okcupid/all/)
* [Videos](http://www.dataschool.io/git-and-github-videos-for-beginners/) on Git and GitHub. Created by one of our very own General Assembly Instructors, Kevin Markham.
* [Reference](http://www.dataschool.io/git-quick-reference-for-beginners/) for common Git commands (created by Kevin Markham).
* Solutions to last week's pandas homework assignment ([code](homework/01_pandas_solutions.py))

------
### Class 3: Advanced Pandas and Machine Learning
**Agenda:**
* Advanced pandas ([code](code/03_pandas.py))
* Iris exploration exercise ([exercise](code/03_iris_prework.py), [solutions](code/03_iris_solutions.py))
* Intro. to Machine Learning ([slides](slides/03_ml_knn.pdf), [code](code/03_sklearn_knn.py))

**Homework:**
* Complete the advanced [Pandas homework](homework/03_pandas_homework.md) (Submit on the [Dat6-students](https://github.com/sinanuozdemir/DAT6-students) repo via a pull request)
* Continue to develop your project. If you have a dataset, explore it with pandas. If you don't have a dataset yet, you should prioritize getting the data.(Nothing to turn in for next week).
* Read this excellent article, [Understanding the Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html), and be prepared to discuss it next class. (You can ignore sections 4.2 and 4.3.) Here are some questions to think about while you read:
    * In the Party Registration example, what are the features? What is the response? Is this a regression or classification problem?
    * In the interactive visualization, try using different values for K across different sets of training data. What value of K do you think is "best"? How do you define "best"?
    * In the visualization, what do the lighter colors versus the darker colors mean? How is the darkness calculated?
    * How does the choice of K affect model bias? How about variance?
    * As you experiment with K and generate new training data, how can you "see" high versus low variance? How can you "see" high versus low bias?
    * Why should we care about variance at all? Shouldn't we just minimize bias and ignore variance?
    * Does a high value for K cause over-fitting or under-fitting?

**Resources:**
* For more on Pandas plotting, read the [visualization page](http://pandas.pydata.org/pandas-docs/stable/visualization.html) from the official Pandas documentation.
* To learn how to customize your plots further, browse through this [notebook on matplotlib](http://nbviewer.ipython.org/github/fonnesbeck/Bios366/blob/master/notebooks/Section2_4-Matplotlib.ipynb) (long!) and check out the [matplotlib documentation](http://matplotlib.org/faq/usage_faq.html).
* To explore different types of visualizations and when to use them, Columbia's Data Mining class has an excellent [slide deck](http://www2.research.att.com/~volinsky/DataMining/Columbia2011/Slides/Topic2-EDAViz.ppt).
* For a more in-depth look at machine learning, read section 2.1 (14 pages) of Hastie and Tibshirani's excellent book, [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). (It's a free PDF download!)
* To learn about NumPy check out this [reference code](code/03_numpy.py)

------
### Class 4: Model Evaluation and Logistic Regression
**Agenda:**
* [The Bias-Variance Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
* Model Evaluation Procedures ([slides](slides/04_model_evaluation_procedures.pdf), [code](code/04_model_evaluation.py))
* Logistic Regression ([slides](slides/04_logistic_regression.pdf), [exercise](code/04_logistic_regression_exercise.py), [solutions](code/04_logistic_regression_solutions.py))
* Model Evalutation Metrics ([slides](slides/04_model_evaluation_metrics.pdf), [code](code/04_confusion_roc.py))

**Homework:**
* Your Homework for this week is to prepare a [2-3 minute presenation](https://github.com/sinanuozdemir/DAT6/blob/master/project.md#march-21-2-3-minute-presentation) and submit a pull request before class.

**Resources:**
* For more on the ROC Curve / AUC, watch the [video](http://www.dataschool.io/roc-curves-and-auc-explained/) (14 minutes total) created by one of our very own GA instructors, Kevin Markham.
* For more on logistic regression, watch the [first three videos](https://www.youtube.com/playlist?list=PL5-da3qGB5IC4vaDba5ClatUmFppXLAhE) (30 minutes total) from Chapter 4 of An Introduction to Statistical Learning.
* UCLA's IDRE has a handy table to help you remember the [relationship between probability, odds, and log-odds](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/odds_ratio.htm).
* Better Explained has a very friendly introduction (with lots of examples) to the [intuition behind "e"](http://betterexplained.com/articles/an-intuitive-guide-to-exponential-functions-e/).
* Here are some useful lecture notes on [interpreting logistic regression coefficients](http://www.unm.edu/~schrader/biostat/bio2/Spr06/lec11.pdf).

------
### Class 5: Linear Regression
**Agenda:**
* Project Status Updates
* Linear Regression and Evaluation ([slides](slides/05_linear_regression.pdf), [code](code/05_linear_regression.py))

**Homework:**
* Your Homework for this week is to continue to develop your project. April 2nd is the deadline for project changes.

**Resources:**
* To go much more in-depth on linear regression, read Chapter 3 of  [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/). Alternatively, watch the [related videos](http://www.dataschool.io/15-hours-of-expert-machine-learning-videos/) that covers the key points from that chapter.
* The  [introduction to linear regression](http://people.duke.edu/~rnau/regintro.htm) is much more mathmatical and thorough, and includes a lot of practical advice.
* The aforementioned article has a particularly helpful section on the [assumptions of linear regression](http://people.duke.edu/~rnau/testing.htm).

### Class 6: Data Problem and Clustering and Visualization
* Today we will work on a real world data problem! Our [data](data/ZYX_prices.csv) is stock data over 7 months of a fictional company ZYX including twitter sentiment, volume and stock price. Our goal is to create a predictive model that predicts forward returns.
* Today we will also be covering our first unsupervised machine learning algorithm, clustering. Our scope will be to explore the kmeans algorithm ([slides](slides/06_clustering.pdf), [code](code/06_kmeans_clustering.py)). In particular we will address:
   * What are the applications of cluster analysis?
   * How does the kmeans algorithm work on a conceptual level?
   * How we can create our own kmeans clustering routine in python. 
   * What are the different options for visualizing the output of Kmeans clustering?
   * How do we measure the quality of our cluster analysis and tune our modeling procedure? ([additional code](code/06_evaluating_cluster_validation.py))
   * What are some of the limitations of  cluster analysis using Kmeans? ([additional code](code/06_kmeans_limitations.py))

* Project overview ([documentation](slides/06_GA_Stocks.pdf))
    * Be sure to read documentation thoroughly and ask questions! We may not have included all of the information you want...
    * Remember, the goal is prediction. We are given labeled data and we must build a supervised model in order to predict forward stock return. When building your models, be sure to use examples from previous classes to build and evaluate them.
    * Metrics are key! Be sure to know which metrics are relevant to the model you chose. For example RMSE only makes sense for regression and ROC/AUC only works for classification.

**Homework:**

* Read Paul Graham's [A Plan for Spam](http://www.paulgraham.com/spam.html) and be prepared to **discuss it in class next time**. Here are some questions to think about while you read:
    * Should a spam filter optimize for sensitivity or specificity, in Paul's opinion?
    * Before he tried the "statistical approach" to spam filtering, what was his approach?
    * How exactly does his statistical filtering system work?
    * What did Paul say were some of the benefits of the statistical approach?
    * How good was his prediction of the "spam of the future"?
* Below are the foundational topics upon which Monday's class will depend. Please review these materials before class:
    * **Confusion matrix:** [guide](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/) roughly mirrors the lecture from class.
    * **Sensitivity and specificity:** Rahul Patwari has an [excellent video](https://www.youtube.com/watch?v=U4_3fditnWg&list=PL41ckbAGB5S2PavLIXUETzAmi5reIod23) (9 minutes).
    * **Basics of probability:** These [slides](https://docs.google.com/presentation/d/1cM2dVbJgTWMkHoVNmYlB9df6P2H8BrjaqAcZTaLe9dA/edit#slide=id.gfc3caad2_00) are very good. Pay specific attention to these terms: probability, mutually exclusive, independent. You may also find videos of Sinan teaching similar ideas in the class videos section of Slack.
* Complete the [kmeans clustering exercise](homework/06_clustering_homework.py) on the [UN dataset](data/UNdata.csv) and submit a pull request to the GitHub repo.
* Conduct kmeans clustering on your own dataset and submit a pull request to the GitHub repository
* Download all of the NLTK collections.
   * In Python, use the following commands to bring up the download menu.
   * ```import nltk```
   * ```nltk.download()```
   * Choose "all".
   * Alternatively, just type ```nltk.download('all')```
* Install two new packages:  ```textblob``` and ```lda```.
   * Open a terminal or command prompt.
   * Type ```pip install textblob``` and ```pip install lda```.   
 
**Deadline for topic changes for your final project is next week!**

**Resources:**

* [Introduction to Data Mining](http://www-users.cs.umn.edu/~kumar/dmbook/index.php) has a great [chapter on cluster analysis](http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf).
* The scikit-learn user guide has a [section on clustering](http://scikit-learn.org/stable/modules/clustering.html).
