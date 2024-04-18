![spam-filter](https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/3d609d19-8f64-45ac-ab76-df20bb29db4b)

----

# **Credit Risk Classification with Logistic Regression using Scikit-learn**

## **Overview of the Analysis**

The purpose of this analysis is to improve the e-mail filtering system at an Internet Service Provider (ISP).  With a dataset containing the frequency of tokens and a spam/not spam designation from 2,788 normal and 1,813 spam e-mails, I developed a supervised machine learning model that accurately detects spam e-mails. When predicting whether the Logistic Regression or Random Forest would be a better choice, I gravitated towards Logistic Regression because of the numerical nature of the features.  Although the instructions stipulated only Logistic Regression and Random Forest models, due to the binary classification nature of the problem, I expanded the analysis to include Decision Tree, Support Vector Machine, K-Nearest Neighbor, and Gaussian Naive Bayes.  Furthermore, the discrepancy between the numbers of normal and spam e-mails convinced me to examine the effect of various techniques for random and synthetic sampling: random undersampling, random oversampling, cluster centroids, synthetic minority oversampling (SMOTE), and synthetic minority oversampling with edited nearest neighbors (SMOTEEN).

To accomplish the analysis, I used the following process:

1. Created 36 optimized hyperparameter models (6 classifiers X 6 sampling methodologies) in the IPython Notebook, spam_detector_optimization.ipynb, and wrote them to files in the folder, resources.
2. Read the spam data into a dataframe in the IPython Notebook, spam_detector.ipynb.
3. Separated the data into features and labels, checked the labels value count, and split the features and labels variables into training and testing data sets.
4. Read the optimized hyperparameters from the files in the folder, resources.
6. Fit the models by using the optimized parameters and the training data.
7. Evaluated the each model’s performance using the testing data to find the accuracy, precision, and recall scores in a confusion matrix.

The purpose of this analysis is to evaluate the performance of Logistic Regression and Random Forest for predicting whether an e-mail is spam. Here are the results of those two models and the ranking of all the models.

<img width="432" alt="Screenshot 2024-04-17 at 3 51 23 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/3009deba-873d-4fec-8320-86d1e3210d41"><img width="431" alt="Screenshot 2024-04-17 at 3 52 02 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/a59bf1a5-625d-4951-9cb7-150780118771">
<img width="436" alt="Screenshot 2024-04-17 at 5 03 33 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/931bd3a0-8970-46bc-8c1d-0726fc544f7f">

<img width="347" alt="Screenshot 2024-04-17 at 3 52 44 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/33020891-cd3c-434e-bbd0-4cac5ffc8de5">

## **Summary**

My prediction of Logistic Regression producing better results than Random Forest is incorrect.  Not only is the accuracy of the Random Forest model superior to the Logistic Regression's metric, the Random Forest model with random oversampling has better accuracy than the other 35 models surveyed.  The most notable observation from these results is the varying effects that oversampling methods can have on any classifier.  

----

### Copyright

Nicholas J. George © 2024. All Rights Reserved.
