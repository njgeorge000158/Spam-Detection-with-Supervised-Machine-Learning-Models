#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  classificationsx.py
 #
 #  File Description:
 #      This Python script, classificationsx.py, contains generic Python functions 
 #      for completing common tasks in classification models. Here is the list:
 #
 #      return_binary_classification_confusion_matrix
 #      
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/11/2023      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/

import logx

import pandas as pd

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'classificationsx.py'


# In[3]:


#*******************************************************************************************
 #
 #  Function Name:  return_binary_classification_confusion_matrix
 #
 #  Function Description:
 #      This function evaluates y series and predictions of a binary classification model
 #      and returns its confusion matrix matrix.
 #
 #
 #  Return Type: float, dataframe, string
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  series  y_test_series   The parameter is test data used for evaluation.
 #  nparray predictions_nparray
 #                          The parameter is the predictions used for evaluation.
 #  string  caption_string  The parameter is the confusion matrix title.
 #  string  true_outcome_string  
 #                          The parameter is the true outcome name.
 #  string  false_outcome_string  
 #                          The parameter is the false outcome name.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_binary_classification_confusion_matrix \
        (y_test_series, 
         predictions_nparray, 
         caption_string,
         true_outcome_string,
         false_outcome_string):

    try:
    
        confusion_matrix_nparray = confusion_matrix(y_test_series, predictions_nparray)
        
        
        index_string_list = ['Actual ' + true_outcome_string, 'Actual ' + false_outcome_string]

        column_string_list = ['Predicted ' + true_outcome_string, 'Predicted ' + false_outcome_string]

        target_names_string_list = [true_outcome_string.lower(), false_outcome_string.lower()]


        accuracy_score_float = balanced_accuracy_score(y_test_series, predictions_nparray)
        
        confusion_matrix_dataframe \
            = pd.DataFrame \
                (confusion_matrix_nparray,
                 index = index_string_list,
                 columns = column_string_list)

        classification_report_string \
            = classification_report \
                (y_test_series, 
                 predictions_nparray, 
                 target_names = target_names_string_list)

        
        logx.print_and_log_text \
            ('\033[1m' + f'{caption_string}\n' + '\033[0m\n'
             + '1) '
             + '\033[1m' + 'Overall Accuracy Score: ' + '\033[0m'
             + f'{round(accuracy_score_float * 100, 2)}%\n\n'
             + '2) '
             + '\033[1m' + 'Confusion Matrix:\n' + '\033[0m\n'
             + f'{confusion_matrix_dataframe}\n\n'
             + f'3) '
             + '\033[1m' + 'Classification Report:\n' + '\033[0m\n'
             + f'{classification_report_string}\n')
    
    
        return accuracy_score_float, confusion_matrix_dataframe, classification_report_string
    
    except:
        
        logx.print_and_log_text \
            ('The function, return_model_performance_evaluation, '
             + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
             + 'was unable to return a confusion matrix.')
    
        return None


# In[ ]:




