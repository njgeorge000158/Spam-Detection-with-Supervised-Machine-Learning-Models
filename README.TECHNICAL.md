# **Spam Detection with Supervised Machine Learning Models**

----

### **Installation:**

----

This project only requires running the Google Colab Notebook, spam_detector.ipynb.

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

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable12SpamDataTable.png](./images/spam_detector_optimization_colabTable12SpamDataTable.png)
  
  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable131SpamTargetSeries.png](./images/spam_detector_optimization_colabTable131SpamTargetSeries.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable132SpamFeaturesDataFrame.png](./images/spam_detector_optimization_colabTable132SpamFeaturesDataFrame.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable151SpamScaledFeaturesTrainingData.png](./images/spam_detector_optimization_colabTable151SpamScaledFeaturesTrainingData.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable152SpamScaledFeaturesTestData.png](./images/spam_detector_optimization_colabTable152SpamScaledFeaturesTestData.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable271ScaledFeaturesTrainingUndersampledData.png](./images/spam_detector_optimization_colabTable271ScaledFeaturesTrainingUndersampledData.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable272ScaledFeaturesTrainingOversampledData.png](./images/spam_detector_optimization_colabTable272ScaledFeaturesTrainingOversampledData.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable273ScaledFeaturesTrainingClusterCentroidsData.png](./images/spam_detector_optimization_colabTable273ScaledFeaturesTrainingClusterCentroidsData.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable275ScaledFeaturesTrainingSMOTEENData.png](./images/spam_detector_optimization_colabTable275ScaledFeaturesTrainingSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detector_optimization_colabTable275XTrainingScaledSMOTEENData.png](./images/spam_detector_optimization_colabTable275XTrainingScaledSMOTEENData.png)

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

  &emsp; |&rarr; [./logs/20240415spam_detector_optimization_colab_log.txt](./logs/20240415spam_detector_optimization_colab_log.txt)

  &emsp; |&rarr; [./logs/README.md](./logs/README.md)

|&rarr; [./resources/](./resources/)

  &emsp; |&rarr; [./resources/dt_CLUSTER_CENTROIDS_grid_search_model.sav](./resources/dt_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_CLUSTER_CENTROIDS_model.sav](./resources/dt_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./resources/dt_grid_search_model.sav](./resources/dt_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_model.sav](./resources/dt_model.sav)

  &emsp; |&rarr; [./resources/dt_OVERSAMPLED_grid_search_model.sav](./resources/dt_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_OVERSAMPLED_model.sav](./resources/dt_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/dt_SMOTE_grid_search_model.sav](./resources/dt_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_SMOTE_model.sav](./resources/dt_SMOTE_model.sav)

  &emsp; |&rarr; [./resources/dt_SMOTEENN_grid_search_model.sav](./resources/dt_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_SMOTEENN_model.sav](./resources/dt_SMOTEENN_model.sav)

  &emsp; |&rarr; [./resources/dt_UNDERSAMPLED_grid_search_model.sav](./resources/dt_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_UNDERSAMPLED_model.sav](./resources/dt_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/gnb_CLUSTER_CENTROIDS_model.sav](./resources/gnb_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./resources/gnb_model.sav](./resources/gnb_model.sav)

  &emsp; |&rarr; [./resources/gnb_OVERSAMPLED_model.sav](./resources/gnb_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/gnb_SMOTE_model.sav](./resources/gnb_SMOTE_model.sav)

  &emsp; |&rarr; [./resources/gnb_SMOTEENN_model.sav](./resources/gnb_SMOTEENN_model.sav)

  &emsp; |&rarr; [./resources/gnb_UNDERSAMPLED_model.sav](./resources/gnb_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/knn_CLUSTER_CENTROIDS_grid_search_model.sav](./resources/knn_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_CLUSTER_CENTROIDSs_model.sav](./resources/knn_CLUSTER_CENTROIDSs_model.sav)

  &emsp; |&rarr; [./resources/knn_grid_search_model.sav](./resources/knn_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_model.sav](./resources/knn_model.sav)

  &emsp; |&rarr; [./resources/knn_OVERSAMPLED_grid_search_model.sav](./resources/knn_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_OVERSAMPLED_model.sav](./resources/knn_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/knn_SMOTE_grid_search_model.sav](./resources/knn_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_SMOTE_model.sav](./resources/knn_SMOTE_model.sav)

  &emsp; |&rarr; [./resources/knn_SMOTEENN_grid_search_model.sav](./resources/knn_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_SMOTEENN_model.sav](./resources/knn_SMOTEENN_model.sav)

  &emsp; |&rarr; [./resources/knn_UNDERSAMPLED_grid_search_model.sav](./resources/knn_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_UNDERSAMPLED_model.sav](./resources/knn_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/lending_data.csv](./resources/lending_data.csv)

  &emsp; |&rarr; [./resources/lr_CLUSTER_CENTROIDS_grid_search_model.sav](./resources/lr_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_CLUSTER_CENTROIDS_model.sav](./resources/lr_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./resources/lr_grid_search_model.sav](./resources/lr_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_model.sav](./resources/lr_model.sav)

  &emsp; |&rarr; [./resources/lr_OVERSAMPLED_grid_search_model.sav](./resources/lr_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_OVERSAMPLED_model.sav](./resources/lr_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/lr_SMOTE_grid_search_model.sav](./resources/lr_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_SMOTE_model.sav](./resources/lr_SMOTE_model.sav)

  &emsp; |&rarr; [./resources/lr_SMOTEENN_grid_search_model.sav](./resources/lr_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_SMOTEENN_model.sav](./resources/lr_SMOTEENN_model.sav)

  &emsp; |&rarr; [./resources/lr_UNDERSAMPLED_grid_search_model.sav](./resources/lr_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_UNDERSAMPLED_model.sav](./resources/lr_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/README.md](./resources/README.md)

  &emsp; |&rarr; [./resources/rf_CLUSTER_CENTROIDS_grid_search_model.sav](./resources/rf_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_CLUSTER_CENTROIDS_model.sav](./resources/rf_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./resources/rf_grid_search_model.sav](./resources/rf_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_model.sav](./resources/rf_model.sav)

  &emsp; |&rarr; [./resources/rf_OVERSAMPLED_grid_search_model.sav](./resources/rf_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_OVERSAMPLED_model.sav](./resources/rf_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/rf_SMOTE_grid_search_model.sav](./resources/rf_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_SMOTE_model.sav](./resources/rf_SMOTE_model.sav)

  &emsp; |&rarr; [./resources/rf_SMOTEENN_grid_search_model.sav](./resources/rf_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_SMOTEENN_model.sav](./resources/rf_SMOTEENN_model.sav)

  &emsp; |&rarr; [./resources/rf_UNDERSAMPLED_grid_search_model.sav](./resources/rf_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_UNDERSAMPLED_model.sav](./resources/rf_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/svm_CLUSTER_CENTROIDS_grid_search_model.sav](./resources/svm_CLUSTER_CENTROIDS_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_CLUSTER_CENTROIDS_model.sav](./resources/svm_CLUSTER_CENTROIDS_model.sav)

  &emsp; |&rarr; [./resources/svm_grid_search_model.sav](./resources/svm_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_model.sav](./resources/svm_model.sav)
  
  &emsp; |&rarr; [./resources/svm_OVERSAMPLED_grid_search_model.sav](./resources/svm_OVERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_OVERSAMPLED_model.sav](./resources/svm_OVERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/svm_SMOTE_grid_search_model.sav](./resources/svm_SMOTE_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_SMOTE_model.sav](./resources/svm_SMOTE_model.sav)

  &emsp; |&rarr; [./resources/svm_SMOTEENN_grid_search_model.sav](./resources/svm_SMOTEENN_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_SMOTEENN_model.sav](./resources/svm_SMOTEENN_model.sav)

  &emsp; |&rarr; [./resources/svm_UNDERSAMPLED_grid_search_model.sav](./resources/svm_UNDERSAMPLED_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_UNDERSAMPLED_model.sav](./resources/svm_UNDERSAMPLED_model.sav)

  &emsp; |&rarr; [./resources/README.md](./resources/README.md)

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
