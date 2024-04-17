![spam-filter](https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/3d609d19-8f64-45ac-ab76-df20bb29db4b)

----

# **Credit Risk Classification with Logistic Regression using Scikit-learn**

## **Overview of the Analysis**

The purpose of this analysis is to improve the e-mail filtering system at an Internet Service Provider (ISP).  With a dataset containing the frequency of tokens and a spam/not spam designation from 2,788 normal and 1,813 spam e-mails for a total of 4,601, I develop a supervised machine learning model that accurately detects spam e-mails.  Although the instructions stipulate only Logistic Regression and Random Forest models, due to the binary classification nature of the problem, I expanded the analysis to also include Decision Tree, Support Vector Machine, K-Nearest Neighbor, and Gaussian Naive Bayes.  Furthermore, the discrepancy between the numbers of normal and spam e-mails convinced me to examine the effect of various techniques for random and synthetic sampling: random undersampling, random oversampling, cluster centroids, synthetic minority oversampling (SMOTE), and synthetic minority oversampling with edited nearest neighbors (SMOTEEN).

When predicting whether the Logistic Regression or Random Forest would be a better choice, I gravitated towards Logistic Regression because of the numerical nature of the features.  To accomplish the analysis, I used the following process:

1. Created 36 optimized models (6 classifiers X 6 sampling methodologies) in the IPython Notebook, spam_detector_optimization.ipynb, and wrote them to files in the folder, resources.
2. Read the spam data into a dataframe in the IPython Notebook, spam_detector.ipynb.
3. Separated the data into features and labels, checked the labels value count, and split the features and labels variables into training and testing data sets.
4. Read the optimized models into the IPython Notebook, spam_detector.ipynb.
6. Fit the models by using the training data.
7. Evaluated the each model’s performance using the testing data to find the accuracy, precision, and recall scores in a confusion matrix.

The purpose of this analysis is to evaluate the performance of Logistic Regression and Random Forest for predicting whether an e-mail is spam. Here are the results of those two models and the ranking of all the models.







## **Summary**

The first Logistic Regression model does an excellent job predicting healthy loans with a small number of false positives and negatives leading to a precision score of 100%, a recall score of 100%, and an f1-score of 100%.  Nevertheless, this model less accurately predicts high-risk loans with a precision of 87%, a recall of 92%, and an f1-score of 90%. The balanced accuracy, 99%, is higher than the actual accuracy, 96%, because of the significant difference in labels's value counts. The first model's potential for an increase in accuracy and the comparatively inadequate performance in predicting high-risk loans vs. health loans are concerning. Thus, the first model warrants further optimization either by closing the value count gap with additional data or random oversampling: the second model uses the latter to solve this problem.

In terms of accuracy, the second Logistic Regression model with random oversampling matches the first model for predicting healthy loans and outperforms it for high-risk loans. For instance, the number of accepted healthy loans falls (18,642 to 18,632); the number of rejected high-risk loans expands (604 to 650); the number of false positives increases slightly (89 to 99); and the number of false negatives significantly drops (49 to 3). Moreover, using random oversampling to generate additional synthetic samples for the minority label class eliminates the labels's value count discrepancy leading to, among other things, the balanced accuracy score matching the overall accuracy score, 99%. For healthy loans, both models have 100% precision, 99% recall, and 100% f1-scores; for high-risk loans, although the precision, 87%, remains the same, the recall, 92%, increases by 8% to 100%, and the f1-score, 90%, increases by 3% to 93%. Consequently, using random oversampling with the Logistic Regression model maintains its identification of healthy loans while improving its identification of high-risk loans.
