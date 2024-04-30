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

  &emsp; |&rarr; [./images/spam_detector_optimizationTable12SpamDataTable.png](./images/spam_detector_optimizationTable12SpamDataTable.png)
  
  &emsp; |&rarr; [./images/spam_detector_optimizationTable131SpamTargetSeries.png](./images/spam_detector_optimizationTable131SpamTargetSeries.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable132SpamFeaturesDataFrame.png](./images/spam_detector_optimizationTable132SpamFeaturesDataFrame.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable151SpamScaledFeaturesTrainingData.png](./images/spam_detector_optimizationTable151SpamScaledFeaturesTrainingData.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable152SpamScaledFeaturesTestData.png](./images/spam_detector_optimizationTable152SpamScaledFeaturesTestData.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable271ScaledFeaturesTrainingUndersampledData.png](./images/spam_detector_optimizationTable271ScaledFeaturesTrainingUndersampledData.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable272ScaledFeaturesTrainingOversampledData.png](./images/spam_detector_optimizationTable272ScaledFeaturesTrainingOversampledData.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable273ScaledFeaturesTrainingClusterCentroidsData.png](./images/spam_detector_optimizationTable273ScaledFeaturesTrainingClusterCentroidsData.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable275ScaledFeaturesTrainingSMOTEENData.png](./images/spam_detector_optimizationTable275ScaledFeaturesTrainingSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detector_optimizationTable275XTrainingScaledSMOTEENData.png](./images/spam_detector_optimizationTable275XTrainingScaledSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detectorTable12SpamDataTable.png](./images/spam_detectorTable12SpamDataTable.png)

  &emsp; |&rarr; [./images/spam_detectorTable131SpamTargetSeries.png](./images/spam_detectorTable131SpamTargetSeries.png)

  &emsp; |&rarr; [./images/spam_detectorTable132SpamFeaturesDataFrame.png](./images/spam_detectorTable132SpamFeaturesDataFrame.png)

  &emsp; |&rarr; [./images/spam_detectorTable151SpamScaledFeaturesTrainingData.png](./images/spam_detectorTable151SpamScaledFeaturesTrainingData.png)

  &emsp; |&rarr; [./images/spam_detectorTable152SpamScaledFeaturesTestData.png](./images/spam_detectorTable152SpamScaledFeaturesTestData.png)

  &emsp; |&rarr; [./images/spam_detectorTable271ScaledFeaturesTrainingUndersampledData.png](./images/spam_detectorTable271ScaledFeaturesTrainingUndersampledData.png)

  &emsp; |&rarr; [./images/spam_detectorTable272ScaledFeaturesTrainingOversampledData.png](./images/spam_detectorTable272ScaledFeaturesTrainingOversampledData.png)

  &emsp; |&rarr; [./images/spam_detectorTable273ScaledFeaturesTrainingClusterCentroidsData.png](./images/spam_detectorTable273ScaledFeaturesTrainingClusterCentroidsData.png)

  &emsp; |&rarr; [./images/spam_detectorTable274ScaledFeaturesTrainingSMOTEData.png](./images/spam_detectorTable274ScaledFeaturesTrainingSMOTEData.png)

  &emsp; |&rarr; [./images/spam_detectorTable275ScaledFeaturesTrainingSMOTEENData.png](./images/spam_detectorTable275ScaledFeaturesTrainingSMOTEENData.png)

  &emsp; |&rarr; [./images/spam_detectorTable971ModelPerformanceMatrix.png](./images/spam_detectorTable971ModelPerformanceMatrix.png)

  &emsp; |&rarr; [./images/spam_detectorTable972ModelPerformanceRankings.png](./images/spam_detectorTable972ModelPerformanceRankings.png)
  
  &emsp; |&rarr; [./images/README.md](./images/README.md)
  
|&rarr; [./logs/](./logs/)

  &emsp; |&rarr; [./logs/20240415spam_detector_log.txt](./logs/20240415spam_detector_log.txt)

  &emsp; |&rarr; [./logs/20240415spam_detector_optimization_log.txt](./logs/20240415spam_detector_optimization_log.txt)

  &emsp; |&rarr; [./logs/README.md](./logs/README.md)

|&rarr; [./resources/](./resources/)

  &emsp; |&rarr; [./resources/dt_centroids_grid_search_model.sav](./resources/dt_centroids_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_centroids_model.sav](./resources/dt_centroids_model.sav)

  &emsp; |&rarr; [./resources/dt_grid_search_model.sav](./resources/dt_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_model.sav](./resources/dt_model.sav)

  &emsp; |&rarr; [./resources/dt_oversampled_grid_search_model.sav](./resources/dt_oversampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_smote_grid_search_model.sav](./resources/dt_smote_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_smote_model.sav](./resources/dt_smote_model.sav)

  &emsp; |&rarr; [./resources/dt_smoteen_grid_search_model.sav](./resources/dt_smoteen_grid_search_model.sav)

  &emsp; |&rarr; [./resources/dt_smoteen_model.sav](./resources/dt_smoteen_model.sav)

  &emsp; |&rarr; [./resources/dt_undersampled_grid_search_model.sav](./resources/dt_undersampled_grid_search_model.sav)
  
  &emsp; |&rarr; [./resources/dt_undersampled_model.sav](./resources/dt_undersampled_model.sav)

  &emsp; |&rarr; [./resources/gnb_centroids_model.sav](./resources/gnb_centroids_model.sav)

  &emsp; |&rarr; [./resources/gnb_model.sav](./resources/gnb_model.sav)

  &emsp; |&rarr; [./resources/gnb_oversampled_model.sav](./resources/gnb_oversampled_model.sav)

  &emsp; |&rarr; [./resources/gnb_smote_model.sav](./resources/gnb_smote_model.sav)

  &emsp; |&rarr; [./resources/gnb_smoteen_model.sav](./resources/gnb_smoteen_model.sav)

  &emsp; |&rarr; [./resources/gnb_undersampled_model.sav](./resources/gnb_undersampled_model.sav)

  &emsp; |&rarr; [./resources/knn_centroids_grid_search_model.sav](./resources/knn_centroids_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_centroids_model.sav](./resources/knn_centroids_model.sav)

  &emsp; |&rarr; [./resources/knn_grid_search_model.sav](./resources/knn_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_model.sav](./resources/knn_model.sav)

  &emsp; |&rarr; [./resources/knn_oversampled_grid_search_model.sav](./resources/knn_oversampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_oversampled_model.sav](./resources/knn_oversampled_model.sav)

  &emsp; |&rarr; [./resources/knn_smote_grid_search_model.sav](./resources/knn_smote_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_smote_model.sav](./resources/knn_smote_model.sav)

  &emsp; |&rarr; [./resources/knn_smoteen_grid_search_model.sav](./resources/knn_smoteen_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_smoteen_model.sav](./resources/knn_smoteen_model.sav)

  &emsp; |&rarr; [./resources/knn_undersampled_grid_search_model.sav](./resources/knn_undersampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/knn_undersampled_model.sav](./resources/knn_undersampled_model.sav)

  &emsp; |&rarr; [./resources/lr_centroids_grid_search_model.sav](./resources/lr_centroids_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_centroids_model.sav](./resources/lr_centroids_model.sav)

  &emsp; |&rarr; [./resources/lr_grid_search_model.sav](./resources/lr_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_model.sav](./resources/lr_model.sav)

  &emsp; |&rarr; [./resources/lr_oversampled_grid_search_model.sav](./resources/lr_oversampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_oversampled_model.sav](./resources/lr_oversampled_model.sav)

  &emsp; |&rarr; [./resources/lr_smote_grid_search_model.sav](./resources/lr_smote_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_smote_model.sav](./resources/lr_smote_model.sav)

  &emsp; |&rarr; [./resources/lr_smoteen_grid_search_model.sav](./resources/lr_smoteen_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_smoteen_model.sav](./resources/lr_smoteen_model.sav)

  &emsp; |&rarr; [./resources/lr_undersampled_grid_search_model.sav](./resources/lr_undersampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/lr_undersampled_model.sav](./resources/lr_undersampled_model.sav)

  &emsp; |&rarr; [./resources/rf_centroids_grid_search_model.sav](./resources/rf_centroids_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_centroids_model.sav](./resources/rf_centroids_model.sav)

  &emsp; |&rarr; [./resources/rf_grid_search_model.sav](./resources/rf_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_model.sav](./resources/rf_model.sav)

  &emsp; |&rarr; [./resources/rf_oversampled_grid_search_model.sav](./resources/rf_oversampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_oversampled_model.sav](./resources/rf_oversampled_model.sav)

  &emsp; |&rarr; [./resources/rf_smote_grid_search_model.sav](./resources/rf_smote_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_smote_model.sav](./resources/rf_smote_model.sav)

  &emsp; |&rarr; [./resources/rf_smoteen_grid_search_model.sav](./resources/rf_smoteen_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_smoteen_model.sav](./resources/rf_smoteen_model.sav)

  &emsp; |&rarr; [./resources/rf_undersampled_grid_search_model.sav](./resources/rf_undersampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/rf_undersampled_model.sav](./resources/rf_undersampled_model.sav)

  &emsp; |&rarr; [./resources/svm_centroids_grid_search_model.sav](./resources/svm_centroids_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_centroids_model.sav](./resources/svm_centroids_model.sav)

  &emsp; |&rarr; [./resources/svm_grid_search_model.sav](./resources/svm_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_model.sav](./resources/svm_model.sav)
  
  &emsp; |&rarr; [./resources/svm_oversampled_grid_search_model.sav](./resources/svm_oversampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_oversampled_model.sav](./resources/svm_oversampled_model.sav)

  &emsp; |&rarr; [./resources/svm_smote_grid_search_model.sav](./resources/svm_smote_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_smote_model.sav](./resources/svm_smote_model.sav)

  &emsp; |&rarr; [./resources/svm_smoteen_grid_search_model.sav](./resources/svm_smoteen_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_smoteen_model.sav](./resources/svm_smoteen_model.sav)

  &emsp; |&rarr; [./resources/svm_undersampled_grid_search_model.sav](./resources/svm_undersampled_grid_search_model.sav)

  &emsp; |&rarr; [./resources/svm_undersampled_model.sav](./resources/svm_undersampled_model.sav)

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
