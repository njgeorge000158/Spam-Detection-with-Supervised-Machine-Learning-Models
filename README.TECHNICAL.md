# **Spam Detection with Supervised Machine Learning Models Using Scikit-Learn**

----

### **Installation:**

----

If the computer has Anaconda, Jupyter Notebook, and a recent version of Python, the IPython notebook already has the following dependencies installed: datetime, io, json, matplotlib, numpy, pandas, pathlib, os, pandas, requests, requests_html, scipy.

In addition to those modules, the IPython notebook needs the following to execute: holoviews, hvplot, geoviews, geopy, aspose-words, dataframe-image, sklearn.

Here are the requisite Terminal commands for the installation of these peripheral modules:

pip3 install -U holoviews

pip3 install -U hvplot

pip3 install -U geoviews

pip3 install -U geopy

pip3 install -U aspose-words

pip3 install -U dataframe-image

pip3 install -U scikit-learn

----

### **Usage:**

----

The IPython notebook, spam_detector.ipynb, requires the following Python scripts with it in the same folder:

classificationsx.py

logx_constants.py

logx.py

pandas_processx.py

If the folders, logs and images, are not present, the IPython notebook will create them.  The IPython notebook, spam_detector.ipynb, requires the csv file, spam_data.csv, found in the link, https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv, to execute. To place the IPython notebook in Log Mode or Image Mode set the parameter for the appropriate function in the IPython notebook's second coding cell to True. If the program is in Log Mode, it writes designated information to the log file in the folder, logs. If the program is in Image Mode, it writes all DataFrames, hvplot maps, and matplotlib plots to PNG and HTML files to the folder, images.

----

### **Resource Summary:**

----

#### Source code

spam_detector.ipynb, classificationsx.py, logx_constants.py, logx.py, log_subroutines.py, pandas_processx.py

#### Input files

spam_data.csv

#### Output files

n/a

#### SQL script

n/a

#### Software

Jupyter Notebook, Matplotlib, Numpy, Pandas, Python 3.11.4, scikit-learn

![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

----

### **GitHub Repository Branches:**

----

#### main branch 

|&rarr; [./classificationsx.py](./classificationsx.pyn)

|&rarr; [./logx_constants.py](./logx_constants.py)

|&rarr; [./logx.py](./logx.py)

|&rarr; [./pandas_processx.py](./pandas_processx.py)

|&rarr; [./spam_detector.ipynb](./spam_detector.ipynb)

|&rarr; [./README.TECHNICAL.md](./README.TECHNICAL.md)

|&rarr; [./README.md](./README.md)

|&rarr; [./images/](./images/)

  &emsp; |&rarr; [./images/spam_detectorTable131SpamYSeries.png](./images/spam_detectorTable131SpamYSeries.png)
  
  &emsp; |&rarr; [./images/spam_detectorTable231LogisticRegressionTrainingPredictions.png](./images/spam_detectorTable231LogisticRegressionTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable232LogisticRegressionTestPredictions.png](./images/spam_detectorTable232LogisticRegressionTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable331DecisionTreeTrainingPredictions.png](./images/spam_detectorTable331DecisionTreeTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable332DecisionTreeTestPredictions.png](./images/spam_detectorTable332DecisionTreeTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable431RandomForestTrainingPredictions.png](./images/spam_detectorTable431RandomForestTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable432RandomForestTestPredictions.png](./images/spam_detectorTable432RandomForestTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable531SVCTrainingPredictions.png](./images/spam_detectorTable531SVCTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable532SVCTestPredictions.png](./images/spam_detectorTable532SVCTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable631KNNTrainingPredictions.png](./images/spam_detectorTable631KNNTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable632KNNTestPredictions.png](./images/spam_detectorTable632KNNTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable831LogisticRegressionResampledTrainingPredictions.png](./images/spam_detectorTable831LogisticRegressionResampledTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable832LogisticRegressionResampledTestPredictions.png](./images/spam_detectorTable832LogisticRegressionResampledTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable931DecisionTreeResampledTrainingPredictions.png](./images/spam_detectorTable931DecisionTreeResampledTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable932DecisionTreeResampledTestPredictions.png](./images/spam_detectorTable932DecisionTreeResampledTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable1031RandomForestResampledTrainingPredictions.png](./images/spam_detectorTable1031RandomForestResampledTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable1032RandomForestResampledTestPredictions.png](./images/spam_detectorTable1032RandomForestResampledTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable1131SVMResampledTrainingPredictions.png](./images/spam_detectorTable1131SVMResampledTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable1132SVMResampledTestPredictions.png](./images/spam_detectorTable1132SVMResampledTestPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable1231KNNResampledTrainingPredictions.png](./images/spam_detectorTable1231KNNResampledTrainingPredictions.png)

  &emsp; |&rarr; [./images/spam_detectorTable1232KNNResampledTestPredictions.png](./images/spam_detectorTable1232KNNResampledTestPredictions.png)

  &emsp; |&rarr; [./images/README.md](./images/README.md)
  
|&rarr; [./logs/](./logs/)

  &emsp; |&rarr; [./logs/20240411spam_detector_log.txt](./logs/20240411spam_detector_log.txt)

  &emsp; |&rarr; [./logs/README.md](./logs/README.md)

|&rarr; [./resources/](./resources/)

  &emsp; |&rarr; [./resources/dt_model.sav](./resources/dt_model.sav)

  &emsp; |&rarr; [./resources/dt_resampled_model.sav](./resources/dt_resampled_model.sav)

  &emsp; |&rarr; [./resources/knn_model.sav](./resources/knn_model.sav)

  &emsp; |&rarr; [./resources/knn_resampled_model.sav](./resources/knn_resampled_model.sav)

  &emsp; |&rarr; [./resources/lr_model.sav](./resources/lr_model.sav)

  &emsp; |&rarr; [./resources/lr_resampled_model.sav](./resources/lr_resampled_model.sav)

  &emsp; |&rarr; [./resources/rf_model.sav](./resources/rf_model.sav)

  &emsp; |&rarr; [./resources/rf_resampled_model.sav](./resources/rf_resampled_model.sav)

  &emsp; |&rarr; [./resources/svm_model.sav](./resources/svm_model.sav)

  &emsp; |&rarr; [./resources/svm_resampled_model.sav](./resources/svm_resampled_model.sav)

  &emsp; |&rarr; [./resources/README.md](./resources/README.md)

----

### **References:**

----

[Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/)

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
