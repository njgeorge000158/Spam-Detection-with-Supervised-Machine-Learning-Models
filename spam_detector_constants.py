#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  spam_detector_constants.py
 #
 #  File Description:
 #      This Python script, spam_detector_constants.py, contains generic Python constants
 #      for spam detector machine learning models.
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/11/2024      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'spam_detector_constants.py'


# In[3]:


CONSTANT_INPUT_FILE_PATH = 'https://static.bc-edx.com/mbc/ai/m4/datasets/spam-data.csv'


CONSTANT_ML_RANDOM_STATE_1 = 21

CONSTANT_ML_RANDOM_STATE_2 = 9

CONSTANT_ML_LR_MAX_ITERATIONS = 10000

CONSTANT_ML_RF_N_ESTIMATORS = 200

CONSTANT_ML_SVM_PROBABILITY = True

CONSTANT_ML_KNN_LEAF_SIZE = 2


# In[4]:


CONSTANT_LR_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_grid_search_model.sav'

CONSTANT_LR_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_UNDERSAMPLED_grid_search_model.sav'

CONSTANT_LR_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_OVERSAMPLED_grid_search_model.sav'

CONSTANT_LR_CLUSTER_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_CLUSTER_CENTROIDS_grid_search_model.sav'

CONSTANT_LR_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_SMOTE_grid_search_model.sav'

CONSTANT_LR_SMOTEENN_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_SMOTEENN_grid_search_model.sav'


CONSTANT_DT_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_grid_search_model.sav'

CONSTANT_DT_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_UNDERSAMPLED_grid_search_model.sav'

CONSTANT_DT_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_OVERSAMPLED_grid_search_model.sav'

CONSTANT_DT_CLUSTER_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_CLUSTER_CENTROIDS_grid_search_model.sav'

CONSTANT_DT_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_SMOTE_grid_search_model.sav'

CONSTANT_DT_SMOTEENN_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_SMOTEENN_grid_search_model.sav'


CONSTANT_RF_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_grid_search_model.sav'

CONSTANT_RF_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_UNDERSAMPLED_grid_search_model.sav'

CONSTANT_RF_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_OVERSAMPLED_grid_search_model.sav'

CONSTANT_RF_CLUSTER_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_CLUSTER_CENTROIDS_grid_search_model.sav'

CONSTANT_RF_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_SMOTE_grid_search_model.sav'

CONSTANT_RF_SMOTEENN_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_SMOTEENN_grid_search_model.sav'


CONSTANT_SVM_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_grid_search_model.sav'

CONSTANT_SVM_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_UNDERSAMPLED_grid_search_model.sav'

CONSTANT_SVM_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_OVERSAMPLED_grid_search_model.sav'

CONSTANT_SVM_CLUSTER_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_CLUSTER_CENTROIDS_grid_search_model.sav'

CONSTANT_SVM_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_SMOTE_grid_search_model.sav'

CONSTANT_SVM_SMOTEENN_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_SMOTEENN_grid_search_model.sav'


CONSTANT_KNN_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_grid_search_model.sav'

CONSTANT_KNN_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_UNDERSAMPLED_grid_search_model.sav'

CONSTANT_KNN_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_OVERSAMPLED_grid_search_model.sav'

CONSTANT_KNN_CLUSTER_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_CLUSTER_CENTROIDS_grid_search_model.sav'

CONSTANT_KNN_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_SMOTE_grid_search_model.sav'

CONSTANT_KNN_SMOTEENN_GRID_SEARCH_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_SMOTEENN_grid_search_model.sav'


# In[5]:


CONSTANT_LR_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_model.sav'

CONSTANT_LR_UNDERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_UNDERSAMPLED_model.sav'

CONSTANT_LR_OVERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_OVERSAMPLED_model.sav'

CONSTANT_LR_CLUSTER_CENTROIDS_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_CLUSTER_CENTROIDS_model.sav'

CONSTANT_LR_SMOTE_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_SMOTE_model.sav'

CONSTANT_LR_SMOTEENN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/lr_SMOTEENN_model.sav'


CONSTANT_DT_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_model.sav'

CONSTANT_DT_UNDERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_UNDERSAMPLED_model.sav'

CONSTANT_DT_OVERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_OVERSAMPLED_model.sav'

CONSTANT_DT_CLUSTER_CENTROIDS_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_CLUSTER_CENTROIDS_model.sav'

CONSTANT_DT_SMOTE_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_SMOTE_model.sav'

CONSTANT_DT_SMOTEENN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/dt_SMOTEENN_model.sav'


CONSTANT_RF_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_model.sav'

CONSTANT_RF_UNDERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_UNDERSAMPLED_model.sav'

CONSTANT_RF_OVERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_OVERSAMPLED_model.sav'

CONSTANT_RF_CLUSTER_CENTROIDS_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_CLUSTER_CENTROIDS_model.sav'

CONSTANT_RF_SMOTE_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_SMOTE_model.sav'

CONSTANT_RF_SMOTEENN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/rf_SMOTEENN_model.sav'


CONSTANT_SVM_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_model.sav'

CONSTANT_SVM_UNDERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_UNDERSAMPLED_model.sav'

CONSTANT_SVM_OVERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_OVERSAMPLED_model.sav'

CONSTANT_SVM_CLUSTER_CENTROIDS_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_CLUSTER_CENTROIDS_model.sav'

CONSTANT_SVM_SMOTE_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_SMOTE_model.sav'

CONSTANT_SVM_SMOTEENN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/svm_SMOTEENN_model.sav'


CONSTANT_KNN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_model.sav'

CONSTANT_KNN_UNDERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_UNDERSAMPLED_model.sav'

CONSTANT_KNN_OVERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_OVERSAMPLED_model.sav'

CONSTANT_KNN_CLUSTER_CENTROIDS_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_CLUSTER_CENTROIDS_model.sav'

CONSTANT_KNN_SMOTE_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_SMOTE_model.sav'

CONSTANT_KNN_SMOTEENN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/knn_SMOTEENN_model.sav'


CONSTANT_GNB_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/gnb_model.sav'

CONSTANT_GNB_UNDERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/gnb_UNDERSAMPLED_model.sav'

CONSTANT_GNB_OVERSAMPLED_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/gnb_OVERSAMPLED_model.sav'

CONSTANT_GNB_CLUSTER_CENTROIDS_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/gnb_CLUSTER_CENTROIDS_model.sav'

CONSTANT_GNB_SMOTE_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/gnb_SMOTE_model.sav'

CONSTANT_GNB_SMOTEENN_MODEL_FILE_PATH \
    = './gdrive/MyDrive/spam_detection/resources/gnb_SMOTEENN_model.sav'


# In[ ]:




