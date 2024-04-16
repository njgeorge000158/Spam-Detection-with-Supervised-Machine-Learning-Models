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
    = './resources/lr_grid_search_model.sav'

CONSTANT_LR_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/lr_undersampled_grid_search_model.sav'

CONSTANT_LR_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/lr_oversampled_grid_search_model.sav'

CONSTANT_LR_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/lr_centroids_grid_search_model.sav'

CONSTANT_LR_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/lr_smote_grid_search_model.sav'

CONSTANT_LR_SMOTEEN_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/lr_smoteen_grid_search_model.sav'


CONSTANT_DT_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/dt_grid_search_model.sav'

CONSTANT_DT_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/dt_undersampled_grid_search_model.sav'

CONSTANT_DT_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/dt_oversampled_grid_search_model.sav'

CONSTANT_DT_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/dt_centroids_grid_search_model.sav'

CONSTANT_DT_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/dt_smote_grid_search_model.sav'

CONSTANT_DT_SMOTEEN_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/dt_smoteen_grid_search_model.sav'


CONSTANT_RF_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/rf_grid_search_model.sav'

CONSTANT_RF_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/rf_undersampled_grid_search_model.sav'

CONSTANT_RF_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/rf_oversampled_grid_search_model.sav'

CONSTANT_RF_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/rf_centroids_grid_search_model.sav'

CONSTANT_RF_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/rf_smote_grid_search_model.sav'

CONSTANT_RF_SMOTEEN_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/rf_smoteen_grid_search_model.sav'


CONSTANT_SVM_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/svm_grid_search_model.sav'

CONSTANT_SVM_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/svm_undersampled_grid_search_model.sav'

CONSTANT_SVM_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/svm_oversampled_grid_search_model.sav'

CONSTANT_SVM_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/svm_centroids_grid_search_model.sav'

CONSTANT_SVM_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/svm_smote_grid_search_model.sav'

CONSTANT_SVM_SMOTEEN_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/svm_smoteen_grid_search_model.sav'


CONSTANT_KNN_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/knn_grid_search_model.sav'

CONSTANT_KNN_UNDERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/knn_undersampled_grid_search_model.sav'

CONSTANT_KNN_OVERSAMPLED_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/knn_oversampled_grid_search_model.sav'

CONSTANT_KNN_CENTROIDS_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/knn_centroids_grid_search_model.sav'

CONSTANT_KNN_SMOTE_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/knn_smote_grid_search_model.sav'

CONSTANT_KNN_SMOTEEN_GRID_SEARCH_MODEL_FILE_PATH \
    = './resources/knn_smoteen_grid_search_model.sav'


# In[5]:


CONSTANT_LR_MODEL_FILE_PATH \
    = './resources/lr_model.sav'

CONSTANT_LR_UNDERSAMPLED_MODEL_FILE_PATH \
    = './resources/lr_undersampled_model.sav'

CONSTANT_LR_OVERSAMPLED_MODEL_FILE_PATH \
    = './resources/lr_oversampled_model.sav'

CONSTANT_LR_CENTROIDS_MODEL_FILE_PATH \
    = './resources/lr_centroids_model.sav'

CONSTANT_LR_SMOTE_MODEL_FILE_PATH \
    = './resources/lr_smote_model.sav'

CONSTANT_LR_SMOTEEN_MODEL_FILE_PATH \
    = './resources/lr_smoteen_model.sav'


CONSTANT_DT_MODEL_FILE_PATH \
    = './resources/dt_model.sav'

CONSTANT_DT_UNDERSAMPLED_MODEL_FILE_PATH \
    = './resources/dt_undersampled_model.sav'

CONSTANT_DT_OVERSAMPLED_MODEL_FILE_PATH \
    = './resources/dt_oversampled_model.sav'

CONSTANT_DT_CENTROIDS_MODEL_FILE_PATH \
    = './resources/dt_centroids_model.sav'

CONSTANT_DT_SMOTE_MODEL_FILE_PATH \
    = './resources/dt_smote_model.sav'

CONSTANT_DT_SMOTEEN_MODEL_FILE_PATH \
    = './resources/dt_smoteen_model.sav'


CONSTANT_RF_MODEL_FILE_PATH \
    = './resources/rf_model.sav'

CONSTANT_RF_UNDERSAMPLED_MODEL_FILE_PATH \
    = './resources/rf_undersampled_model.sav'

CONSTANT_RF_OVERSAMPLED_MODEL_FILE_PATH \
    = './resources/rf_oversampled_model.sav'

CONSTANT_RF_CENTROIDS_MODEL_FILE_PATH \
    = './resources/rf_centroids_model.sav'

CONSTANT_RF_SMOTE_MODEL_FILE_PATH \
    = './resources/rf_smote_model.sav'

CONSTANT_RF_SMOTEEN_MODEL_FILE_PATH \
    = './resources/rf_smoteen_model.sav'


CONSTANT_SVM_MODEL_FILE_PATH \
    = './resources/svm_model.sav'

CONSTANT_SVM_UNDERSAMPLED_MODEL_FILE_PATH \
    = './resources/svm_undersampled_model.sav'

CONSTANT_SVM_OVERSAMPLED_MODEL_FILE_PATH \
    = './resources/svm_oversampled_model.sav'

CONSTANT_SVM_CENTROIDS_MODEL_FILE_PATH \
    = './resources/svm_centroids_model.sav'

CONSTANT_SVM_SMOTE_MODEL_FILE_PATH \
    = './resources/svm_smote_model.sav'

CONSTANT_SVM_SMOTEEN_MODEL_FILE_PATH \
    = './resources/svm_smoteen_model.sav'


CONSTANT_KNN_MODEL_FILE_PATH \
    = './resources/knn_model.sav'

CONSTANT_KNN_UNDERSAMPLED_MODEL_FILE_PATH \
    = './resources/knn_undersampled_model.sav'

CONSTANT_KNN_OVERSAMPLED_MODEL_FILE_PATH \
    = './resources/knn_oversampled_model.sav'

CONSTANT_KNN_CENTROIDS_MODEL_FILE_PATH \
    = './resources/knn_centroids_model.sav'

CONSTANT_KNN_SMOTE_MODEL_FILE_PATH \
    = './resources/knn_smote_model.sav'

CONSTANT_KNN_SMOTEEN_MODEL_FILE_PATH \
    = './resources/knn_smoteen_model.sav'


CONSTANT_GNB_MODEL_FILE_PATH \
    = './resources/gnb_model.sav'

CONSTANT_GNB_UNDERSAMPLED_MODEL_FILE_PATH \
    = './resources/gnb_undersampled_model.sav'

CONSTANT_GNB_OVERSAMPLED_MODEL_FILE_PATH \
    = './resources/gnb_oversampled_model.sav'

CONSTANT_GNB_CENTROIDS_MODEL_FILE_PATH \
    = './resources/gnb_centroids_model.sav'

CONSTANT_GNB_SMOTE_MODEL_FILE_PATH \
    = './resources/gnb_smote_model.sav'

CONSTANT_GNB_SMOTEEN_MODEL_FILE_PATH \
    = './resources/gnb_smoteen_model.sav'


# In[ ]:




