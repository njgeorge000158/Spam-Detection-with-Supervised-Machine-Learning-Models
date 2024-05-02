# **Spam Detection with Supervised Machine Learning Models**

----

### **Installation:**

----

This project only requires running the Google Colab Notebook, spam_detector_colab.ipynb.

----

### **Usage:**

----

The Google Colab Notebook, spam_detector_colab.ipynb, requires the following Python scripts with it in the same folder:

classificationsx.py

logx.py

pandasx.py

spam_detector_constants.py

timex.py

If the folders, logs and images, are not present, the Google Colab Notebook will create them.  The Google Colab Notebook, spam_detector_colab.ipynb, requires the csv file, spam_data.csv, found in the link, https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv, and grid search model files found in the folder, resources, to execute. To place the Google Colab Notebook in Log Mode or Image Mode set the parameter for the appropriate function in the Google Colab Notebook's second coding cell to True. In Log Mode, it writes designated information to the log file in the folder, logs. If the program is in Image Mode, it writes all DataFrames, hvplot maps, and matplotlib plots to PNG and HTML files to the folder, images.

----

### **Resource Summary:**

----

#### Source code

spam_detector_colab.ipynb, classificationsx.py, logx.py, pandasx.py, spam_detector_constants.py, timex.py

#### Input files

spam_data.csv, dt_CLUSTER_CENTROIDS_grid_search_model.sav, dt_grid_search_model.sav, dt_OVERSAMPLED_grid_search_model.sav, dt_SMOTE_grid_search_model.sav, dt_SMOTEENN_grid_search_model.sav, dt_UNDERSAMPLED_grid_search_model.sav, knn_CLUSTER_CENTROIDS_grid_search_model.sav, knn_grid_search_model.sav, knn_OVERSAMPLED_grid_search_model.sav, knn_SMOTE_grid_search_model.sav, knn_SMOTEENN_grid_search_model.sav, knn_UNDERSAMPLED_grid_search_model.sav, lr_CLUSTER_CENTROIDS_grid_search_model.sav, lr_grid_search_model.sav, lr_OVERSAMPLED_grid_search_model.sav, lr_SMOTE_grid_search_model.sav, lr_SMOTEENN_grid_search_model.sav, lr_UNDERSAMPLED_grid_search_model.sav, rf_CLUSTER_CENTROIDS_grid_search_model.sav, rf_grid_search_model.sav, rf_OVERSAMPLED_grid_search_model.sav, rf_SMOTE_grid_search_model.sav, rf_SMOTEENN_grid_search_model.sav, rf_UNDERSAMPLED_grid_search_model.sav, svm_CLUSTER_CENTROIDS_grid_search_model.sav, svm_grid_search_model.sav, svm_OVERSAMPLED_grid_search_model.sav, svm_SMOTE_grid_search_model.sav, svm_SMOTEENN_grid_search_model.sav, svm_UNDERSAMPLED_grid_search_model.sav, 

#### Output files

dt_CLUSTER_CENTROIDS_model.sav, dt_model.sav, dt_OVERSAMPLED_model.sav, dt_SMOTE_model.sav, dt_SMOTEENN_model.sav, dt_UNDERSAMPLED_model.sav, gnb_CLUSTER_CENTROIDS_model.sav, gnb_model.sav, gnb_OVERSAMPLED_model.sav, gnb_SMOTE_model.sav, gnb_SMOTEENN_model.sav, gnb_UNDERSAMPLED_model.sav, knn_CLUSTER_CENTROIDS_model.sav, knn_model.sav, knn_OVERSAMPLED_model.sav, knn_SMOTE_model.sav, knn_SMOTEENN_model.sav, knn_UNDERSAMPLED_model.sav, lr_CLUSTER_CENTROIDS_model.sav, lr_model.sav, lr_OVERSAMPLED_model.sav, lr_SMOTE_model.sav, lr_SMOTEENN_model.sav, lr_UNDERSAMPLED_model.sav, rf_CLUSTER_CENTROIDS_model.sav, rf_model.sav, rf_OVERSAMPLED_model.sav, rf_SMOTE_model.sav, rf_SMOTEENN_model.sav, rf_UNDERSAMPLED_model.sav, svm_CLUSTER_CENTROIDS_model.sav, svm_model.sav, svm_OVERSAMPLED_model.sav, svm_SMOTE_model.sav, svm_SMOTEENN_model.sav, svm_UNDERSAMPLED_model.sav

#### SQL script

n/a

#### Software

Matplotlib, Numpy, Pandas, Python 3.11.4, scikit-learn

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

----

### **GitHub Repository Branches:**

----

#### main branch 

|&rarr; [./classificationsx.py](./classificationsx.py)

|&rarr; [./logx.py](./logx.py)

|&rarr; [./pandasx.py](./pandasx.py)

|&rarr; [./spam_detector_constants.py](./spam_detector_constants.py)

|&rarr; [./spam_detector_colab.ipynb](./spam_detector_colab.ipynb)

|&rarr; [./README.TECHNICAL.md](./README.TECHNICAL.md)

|&rarr; [./README.md](./README.md)

|&rarr; [./timex.py](./timex.pyy)

|&rarr; [./images/](./images/)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable12SpamDataTable.png](./images/spam_detector_hyperparameters_optimization_colabTable12SpamDataTable.png)
  
  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable131SpamTargetSeries.png](./images/spam_detector_hyperparameters_optimization_colabTable131SpamTargetSeries.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable132SpamFeaturesDataFrame.png](./images/spam_detector_hyperparameters_optimization_colabTable132SpamFeaturesDataFrame.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable151SpamScaledFeaturesTrainingData.png](./images/spam_detector_hyperparameters_optimization_colabTable151SpamScaledFeaturesTrainingData.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable152SpamScaledFeaturesTestData.png](./images/spam_detector_hyperparameters_optimization_colabTable152SpamScaledFeaturesTestData.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable271ScaledFeaturesTrainingUndersampledData.png](./images/spam_detector_hyperparameters_optimization_colabTable271ScaledFeaturesTrainingUndersampledData.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable272ScaledFeaturesTrainingOversampledData.png](./images/spam_detector_hyperparameters_optimization_colabTable272ScaledFeaturesTrainingOversampledData.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable273ScaledFeaturesTrainingClusterCentroidsData.png](./images/spam_detector_hyperparameters_optimization_colabTable273ScaledFeaturesTrainingClusterCentroidsData.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable275ScaledFeaturesTrainingSMOTEENData.png](./images/spam_detector_hyperparameters_optimization_colabTable275ScaledFeaturesTrainingSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detector_hyperparameters_optimization_colabTable275XTrainingScaledSMOTEENData.png](./images/spam_detector_hyperparameters_optimization_colabTable275XTrainingScaledSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable12SpamDataTable.png](./images/spam_detector_colabTable12SpamDataTable.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable131SpamTargetSeries.png](./images/spam_detector_colabTable131SpamTargetSeries.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable132SpamFeaturesDataFrame.png](./images/spam_detector_colabTable132SpamFeaturesDataFrame.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable151SpamScaledFeaturesTrainingData.png](./images/spam_detector_colabTable151SpamScaledFeaturesTrainingData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable152SpamScaledFeaturesTestData.png](./images/spam_detector_colabTable152SpamScaledFeaturesTestData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable271ScaledFeaturesTrainingUndersampledData.png](./images/spam_detector_colabTable271ScaledFeaturesTrainingUndersampledData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable272ScaledFeaturesTrainingOversampledData.png](./images/spam_detector_colabTable272ScaledFeaturesTrainingOversampledData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable273ScaledFeaturesTrainingClusterCentroidsData.png](./images/spam_detector_colabTable273ScaledFeaturesTrainingClusterCentroidsData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable274ScaledFeaturesTrainingSMOTEData.png](./images/spam_detector_colabTable274ScaledFeaturesTrainingSMOTEData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable275ScaledFeaturesTrainingSMOTEENData.png](./images/spam_detector_colabTable275ScaledFeaturesTrainingSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable971ModelPerformanceMatrix.png](./images/spam_detector_colabTable971ModelPerformanceMatrix.png)

  &emsp; |&rarr; [./images/spam_detector_colabTable972ModelPerformanceRankings.png](./images/spam_detector_colabTable972ModelPerformanceRankings.png)
  
  &emsp; |&rarr; [./images/README.md](./images/README.md)
  
|&rarr; [./logs/](./logs/)

  &emsp; |&rarr; [./logs/20240415spam_detector_colab_log.txt](./logs/20240415spam_detector_colab_log.txt)

  &emsp; |&rarr; [./logs/20240415spam_detector_hyperparameters_optimization_colab_log.txt](./logs/20240415spam_detector_hyperparameters_optimization_colab_log.txt)

  &emsp; |&rarr; [./logs/README.md](./logs/README.md)

|&rarr; [./models/](./models/)

  &emsp; |&rarr; [./models/dt_CLUSTER_CENTROIDS_grid_search_model.sav](./models/dt_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./models/dt_CLUSTER_CENTROIDS_model.sav](./models/dt_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./models/dt_grid_search_model.sav](./models/dt_grid_search_model.sav)

  &emsp; |&rarr; [./models/dt_model.sav](./models/dt_model.sav)

  &emsp; |&rarr; [./models/dt_OVERSAMPLED_grid_search_model.sav](./models/dt_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/dt_OVERSAMPLED_model.sav](./models/dt_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/dt_SMOTE_grid_search_model.sav](./models/dt_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./models/dt_SMOTE_model.sav](./models/dt_SMOTE_model.sav)

  &emsp; |&rarr; [./models/dt_SMOTEENN_grid_search_model.sav](./models/dt_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./models/dt_SMOTEENN_model.sav](./models/dt_SMOTEENN_model.sav)

  &emsp; |&rarr; [./models/dt_UNDERSAMPLED_grid_search_model.sav](./models/dt_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/dt_UNDERSAMPLED_model.sav](./models/dt_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/gnb_CLUSTER_CENTROIDS_model.sav](./models/gnb_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./models/gnb_model.sav](./models/gnb_model.sav)

  &emsp; |&rarr; [./models/gnb_OVERSAMPLED_model.sav](./models/gnb_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/gnb_SMOTE_model.sav](./models/gnb_SMOTE_model.sav)

  &emsp; |&rarr; [./models/gnb_SMOTEENN_model.sav](./models/gnb_SMOTEENN_model.sav)

  &emsp; |&rarr; [./models/gnb_UNDERSAMPLED_model.sav](./models/gnb_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/knn_CLUSTER_CENTROIDS_grid_search_model.sav](./models/knn_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./models/knn_CLUSTER_CENTROIDSs_model.sav](./models/knn_CLUSTER_CENTROIDSs_model.sav)

  &emsp; |&rarr; [./models/knn_grid_search_model.sav](./models/knn_grid_search_model.sav)

  &emsp; |&rarr; [./models/knn_model.sav](./models/knn_model.sav)

  &emsp; |&rarr; [./models/knn_OVERSAMPLED_grid_search_model.sav](./models/knn_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/knn_OVERSAMPLED_model.sav](./models/knn_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/knn_SMOTE_grid_search_model.sav](./models/knn_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./models/knn_SMOTE_model.sav](./models/knn_SMOTE_model.sav)

  &emsp; |&rarr; [./models/knn_SMOTEENN_grid_search_model.sav](./models/knn_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./models/knn_SMOTEENN_model.sav](./models/knn_SMOTEENN_model.sav)

  &emsp; |&rarr; [./models/knn_UNDERSAMPLED_grid_search_model.sav](./models/knn_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/knn_UNDERSAMPLED_model.sav](./models/knn_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/lending_data.csv](./models/lending_data.csv)

  &emsp; |&rarr; [./models/lr_CLUSTER_CENTROIDS_grid_search_model.sav](./models/lr_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./models/lr_CLUSTER_CENTROIDS_model.sav](./models/lr_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./models/lr_grid_search_model.sav](./models/lr_grid_search_model.sav)

  &emsp; |&rarr; [./models/lr_model.sav](./models/lr_model.sav)

  &emsp; |&rarr; [./models/lr_OVERSAMPLED_grid_search_model.sav](./models/lr_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/lr_OVERSAMPLED_model.sav](./models/lr_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/lr_SMOTE_grid_search_model.sav](./models/lr_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./models/lr_SMOTE_model.sav](./models/lr_SMOTE_model.sav)

  &emsp; |&rarr; [./models/lr_SMOTEENN_grid_search_model.sav](./models/lr_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./models/lr_SMOTEENN_model.sav](./models/lr_SMOTEENN_model.sav)

  &emsp; |&rarr; [./models/lr_UNDERSAMPLED_grid_search_model.sav](./models/lr_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/lr_UNDERSAMPLED_model.sav](./models/lr_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/README.md](./models/README.md)

  &emsp; |&rarr; [./models/rf_CLUSTER_CENTROIDS_grid_search_model.sav](./models/rf_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./models/rf_CLUSTER_CENTROIDS_model.sav](./models/rf_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./models/rf_grid_search_model.sav](./models/rf_grid_search_model.sav)

  &emsp; |&rarr; [./models/rf_model.sav](./models/rf_model.sav)

  &emsp; |&rarr; [./models/rf_OVERSAMPLED_grid_search_model.sav](./models/rf_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/rf_OVERSAMPLED_model.sav](./models/rf_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/rf_SMOTE_grid_search_model.sav](./models/rf_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./models/rf_SMOTE_model.sav](./models/rf_SMOTE_model.sav)

  &emsp; |&rarr; [./models/rf_SMOTEENN_grid_search_model.sav](./models/rf_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./models/rf_SMOTEENN_model.sav](./models/rf_SMOTEENN_model.sav)

  &emsp; |&rarr; [./models/rf_UNDERSAMPLED_grid_search_model.sav](./models/rf_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/rf_UNDERSAMPLED_model.sav](./models/rf_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/svm_CLUSTER_CENTROIDS_grid_search_model.sav](./models/svm_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./models/svm_CLUSTER_CENTROIDS_model.sav](./models/svm_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./models/svm_grid_search_model.sav](./models/svm_grid_search_model.sav)

  &emsp; |&rarr; [./models/svm_model.sav](./models/svm_model.sav)
  
  &emsp; |&rarr; [./models/svm_OVERSAMPLED_grid_search_model.sav](./models/svm_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/svm_OVERSAMPLED_model.sav](./models/svm_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/svm_SMOTE_grid_search_model.sav](./models/svm_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./models/svm_SMOTE_model.sav](./models/svm_SMOTE_model.sav)

  &emsp; |&rarr; [./models/svm_SMOTEENN_grid_search_model.sav](./models/svm_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./models/svm_SMOTEENN_model.sav](./models/svm_SMOTEENN_model.sav)

  &emsp; |&rarr; [./models/svm_UNDERSAMPLED_grid_search_model.sav](./models/svm_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./models/svm_UNDERSAMPLED_model.sav](./models/svm_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./models/README.md](./models/README.md)

----

### **References:**

----

[imbalanced-learn documentation](https://imbalanced-learn.org/stable/)

[Matplotlib Documentation](https://matplotlib.org/stable/index.html)

[Numpy documentation](https://numpy.org/doc/1.26/)

[Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

[Python Documentation](https://docs.python.org/3/contents.html)

[scikit-learn Documentation](https://scikit-learn.org/stable/)

----

### **Authors and Acknowledgment:**

----

### Copyright

Nicholas J. George © 2024. All Rights Reserved.
