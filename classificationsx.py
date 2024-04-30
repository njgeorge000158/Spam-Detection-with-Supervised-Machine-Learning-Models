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
 #      return_optimal_hyperparameters_logistic_regression
 #      return_optimal_hyperparameters_decision_tree
 #      return_optimal_hyperparameters_random_forest
 #      return_optimal_hyperparameters_svm
 #      return_optimal_hyperparameters_knn
 #
 #      return_predictions_dataframe
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

from IPython.display import clear_output

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'classificationsx.py'


# In[3]:


#*******************************************************************************************
 #
 #  Function Name:  return_optimal_hyperparameters_logistic_regression
 #
 #  Function Description:
 #      This function returns the best hyperparameters from overall accuracy 
 #      for a logistic regression model.
 #
 #
 #  Return Type: dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          x_train_scaled_dataframe        
 #                          The parameter is the scaled training model features 
 #                          or x-variable.
 #  dataframe
 #          x_test_scaled_dataframe        
 #                          The parameter is the scaled test model features 
 #                          or x-variable.
 #  series  y_train_series  The parameter is the training target or y-variable.
 #  series  y_test_series   The parameter is the test target or y-variable.
 #  integer max_iterations_integer
 #                          The parameter is the maximum number of iterations 
 #                          for the model to converge.
 #  integer random_state_integer  
 #                          The parameter is the random state for the model.
 #  boolean
 #          display_progress_boolean
 #                          The parameter indicates whether the function will 
 #                          display progress.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_optimal_hyperparameters_logistic_regression \
        (x_train_scaled_dataframe,
         x_test_scaled_dataframe,
         y_train_series,
         y_test_series,
         max_iterations_integer = 10000,
         random_state_integer = 21,
         display_progress_boolean = True):

    best_hyperparameters_dictionary = {'parameters': None, 'accuracy': 0.0}

        
    solver_string_list \
        = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']

    class_weight_list = ['balanced', None]

    multi_class_string_list = ['auto', 'ovr', 'multinomial']

    boolean_list = [True, False]

        
    index_integer = 0
        
    upper_limit_integer \
        = len(solver_string_list) \
            * len(class_weight_list) \
            * len(multi_class_string_list) \
            * len(boolean_list) * 2

        
    for current_solver in solver_string_list:

        for current_class_weight in class_weight_list:

            for current_multi_class in multi_class_string_list:

                for current_dual in boolean_list:

                    for current_fit_intercept in boolean_list:

                        if display_progress_boolean == True:

                            clear_output(wait = True)

                            index_integer += 1
                                
                            logx.print_and_log_text \
                                (f'Progress: {index_integer}/{upper_limit_integer}')
                            

                        try:
            
                            lr_model \
                                = LogisticRegression \
                                    (solver = current_solver,
                                     class_weight = current_class_weight,
                                     multi_class = current_multi_class,
                                     dual = current_dual,
                                     fit_intercept = current_fit_intercept,
                                     max_iter = max_iterations_integer,
                                     random_state = random_state_integer) \
                                        .fit(x_train_scaled_dataframe, y_train_series)

                        except:

                            continue

                        test_predictions_nparray \
                            = lr_model.predict(x_test_scaled_dataframe)

                        accuracy_score_float \
                            = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
                        if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

                            best_hyperparameters_dictionary['parameters'] \
                                = lr_model.get_params()
                
                            best_hyperparameters_dictionary['accuracy'] \
                                = accuracy_score_float

               
    if display_progress_boolean == True:
            
        logx.print_and_log_text('\n')

        
    return best_hyperparameters_dictionary


# In[4]:


#*******************************************************************************************
 #
 #  Function Name:  return_optimal_hyperparameters_decision_tree
 #
 #  Function Description:
 #      This function returns the best hyperparameters from overall accuracy 
 #      for a decision tree model.
 #
 #
 #  Return Type: dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          x_train_scaled_dataframe        
 #                          The parameter is the scaled training model features 
 #                          or x-variable.
 #  dataframe
 #          x_test_scaled_dataframe        
 #                          The parameter is the scaled test model features 
 #                          or x-variable.
 #  series  y_train_series  The parameter is the training target or y-variable.
 #  series  y_test_series   The parameter is the test target or y-variable.
 #  integer random_state_integer  
 #                          The parameter is the random state for the model.
 #  boolean
 #          display_progress_boolean
 #                          The parameter indicates whether the function will 
 #                          display progress.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_optimal_hyperparameters_decision_tree \
        (x_train_scaled_dataframe,
         x_test_scaled_dataframe,
         y_train_series,
         y_test_series,
         random_state_integer = 21,
         display_progress_boolean = True):

    best_hyperparameters_dictionary = {'parameters': None, 'accuracy': 0.0}
        
        
    criterion_string_list = ['gini', 'entropy', 'log_loss']

    splitter_string_list = ['best', 'random']

    class_weight_list = ['balanced', None]


    index_integer = 0
        
    upper_limit_integer \
        = len(criterion_string_list) \
            * len(splitter_string_list) \
            * len(class_weight_list)
        
        
    for current_criterion in criterion_string_list:

        for current_splitter in splitter_string_list:

            for current_class_weight in class_weight_list:

                if display_progress_boolean == True:

                    clear_output(wait = True)

                    index_integer += 1
                                
                    logx.print_and_log_text \
                        (f'Progress: {index_integer}/{upper_limit_integer}')

                    
                try:
                        
                    dt_model \
                        = DecisionTreeClassifier \
                            (criterion = current_criterion,
                             splitter = current_splitter,
                             class_weight = current_class_weight,
                             random_state = random_state_integer) \
                                .fit(x_train_scaled_dataframe, y_train_series)

                except:
                        
                    continue
                        
                test_predictions_nparray \
                    = dt_model.predict(x_test_scaled_dataframe)

                accuracy_score_float \
                    = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
                if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

                    best_hyperparameters_dictionary['parameters'] = dt_model.get_params()
                
                    best_hyperparameters_dictionary['accuracy'] = accuracy_score_float

        
    if display_progress_boolean == True:
            
        logx.print_and_log_text('\n')
        
        
    return best_hyperparameters_dictionary


# In[5]:


#*******************************************************************************************
 #
 #  Function Name:  return_optimal_hyperparameters_random_forest
 #
 #  Function Description:
 #      This function returns the best hyperparameters from overall accuracy 
 #      for a random forest model.
 #
 #
 #  Return Type: dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          x_train_scaled_dataframe        
 #                          The parameter is the scaled training model features 
 #                          or x-variable.
 #  dataframe
 #          x_test_scaled_dataframe        
 #                          The parameter is the scaled test model features 
 #                          or x-variable.
 #  series  y_train_series  The parameter is the training target or y-variable.
 #  series  y_test_series   The parameter is the test target or y-variable.
 #  integer random_state_integer  
 #                          The parameter is the random state for the model.
 #  boolean
 #          display_progress_boolean
 #                          The parameter indicates whether the function will 
 #                          display progress.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_optimal_hyperparameters_random_forest \
        (x_train_scaled_dataframe,
         x_test_scaled_dataframe,
         y_train_series,
         y_test_series,
         random_state_integer = 21,
         display_progress_boolean = True):
            
    best_hyperparameters_dictionary = {'parameters': None, 'accuracy': 0.0}
        
        
    criterion_string_list = ['gini', 'entropy', 'log_loss']

    class_weight_string_list = ['balanced', 'balanced_subsample', None]

    monotonic_cst_list = [1, 0, -1, None]

    boolean_list = [True, False]


    index_integer = 0
        
    upper_limit_integer \
        = len(criterion_string_list) \
            * len(class_weight_string_list) \
            * len(monotonic_cst_list) \
            * len(boolean_list) * 2


    for current_criterion in criterion_string_list:

        for current_class_weight in class_weight_string_list:

            for current_monotonic_cst in monotonic_cst_list:

                for current_bootstrap in boolean_list:

                    for current_oob_score in boolean_list:

                        if display_progress_boolean == True:
                                
                            clear_output(wait = True)

                            index_integer += 1
                                
                            logx.print_and_log_text \
                                (f'Progress: {index_integer}/{upper_limit_integer}')

                        try:
                
                            rf_model \
                                = RandomForestClassifier \
                                    (criterion = current_criterion,
                                     class_weight = current_class_weight,
                                     monotonic_cst = current_monotonic_cst,
                                     bootstrap = current_bootstrap,
                                     oob_score = current_oob_score,
                                     random_state = random_state_integer) \
                                        .fit(x_train_scaled_dataframe, y_train_series)

                        except:

                            continue

                        test_predictions_nparray \
                            = rf_model.predict(x_test_scaled_dataframe)

                        accuracy_score_float \
                            = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
                        if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

                            best_hyperparameters_dictionary['parameters'] \
                                = rf_model.get_params()
                
                            best_hyperparameters_dictionary['accuracy'] \
                                = accuracy_score_float

        
    upper_limit_integer = (len(x_train_scaled_dataframe.columns) * 10) // 2

    if upper_limit_integer < 100:

        upper_limit_integer = 100
 
        
    for n in range(10, upper_limit_integer + 1):

        if display_progress_boolean == True:
            
            clear_output(wait = True)
            
            logx.print_and_log_text(f'Progress: {n}/{upper_limit_integer}')
            
        rf_model \
            = RandomForestClassifier \
                (criterion = best_hyperparameters_dictionary['parameters']['criterion'],
                 class_weight = best_hyperparameters_dictionary['parameters']['class_weight'],
                 monotonic_cst = best_hyperparameters_dictionary['parameters']['monotonic_cst'],
                 bootstrap = best_hyperparameters_dictionary['parameters']['bootstrap'],
                 oob_score = best_hyperparameters_dictionary['parameters']['oob_score'],
                 n_estimators = n,
                 random_state = random_state_integer) \
                    .fit(x_train_scaled_dataframe, y_train_series)

        test_predictions_nparray \
            = rf_model.predict(x_test_scaled_dataframe)

        accuracy_score_float \
            = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
        if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

            best_hyperparameters_dictionary['parameters'] = rf_model.get_params()
                
            best_hyperparameters_dictionary['accuracy'] = accuracy_score_float

        
    if display_progress_boolean == True:
            
        logx.print_and_log_text('\n')
        
        
    return best_hyperparameters_dictionary


# In[6]:


#*******************************************************************************************
 #
 #  Function Name:  return_optimal_hyperparameters_svm
 #
 #  Function Description:
 #      This function returns the best hyperparameters from overall accuracy 
 #      for a support vector machine model.
 #
 #
 #  Return Type: dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          x_train_scaled_dataframe        
 #                          The parameter is the scaled training model features 
 #                          or x-variable.
 #  dataframe
 #          x_test_scaled_dataframe        
 #                          The parameter is the scaled test model features 
 #                          or x-variable.
 #  series  y_train_series  The parameter is the training target or y-variable.
 #  series  y_test_series   The parameter is the test target or y-variable.
 #  integer random_state_integer  
 #                          The parameter is the random state for the model.
 #  boolean
 #          display_progress_boolean
 #                          The parameter indicates whether the function will 
 #                          display progress.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_optimal_hyperparameters_svm \
        (x_train_scaled_dataframe,
         x_test_scaled_dataframe,
         y_train_series,
         y_test_series,
         random_state_integer = 21,
         display_progress_boolean = True):

    best_hyperparameters_dictionary = {'parameters': None, 'accuracy': 0.0}
        
       
    kernel_string_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

    gamma_string_list = ['scale', 'auto']

    class_weight_list = ['balanced', None]

    decision_function_shape_string_list = ['ovo', 'ovr']

    boolean_list = [True, False]


    index_integer = 0
        
    upper_limit_integer \
        = len(kernel_string_list) * len(gamma_string_list) \
            * len(class_weight_list) * len(decision_function_shape_string_list) \
            * len(boolean_list) * 2

    for current_kernel in kernel_string_list:

        for current_gamma in gamma_string_list:

            for current_class_weight in class_weight_list:

                for current_decision_function_shape in decision_function_shape_string_list:

                    for current_shrinking in boolean_list:
    
                        for current_probablity in boolean_list:

                            if display_progress_boolean == True:

                                clear_output(wait = True)

                                index_integer += 1
                                
                                logx.print_and_log_text \
                                    (f'Progress: {index_integer}/{upper_limit_integer}')

                            try:
                                svm_model \
                                    = SVC \
                                        (kernel = current_kernel,
                                         gamma = current_gamma,
                                         class_weight = current_class_weight,
                                         decision_function_shape = current_decision_function_shape,
                                         shrinking = current_shrinking,
                                         probability = current_probablity,
                                         random_state = random_state_integer) \
                                            .fit(x_train_scaled_dataframe, y_train_series)

                            except:

                                continue

                            test_predictions_nparray = svm_model.predict(x_test_scaled_dataframe)

                            accuracy_score_float \
                                = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
                            if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

                                best_hyperparameters_dictionary['parameters'] = svm_model.get_params()
                
                                best_hyperparameters_dictionary['accuracy'] = accuracy_score_float

  
    if best_hyperparameters_dictionary['parameters']['kernel'] == 'poly':

        max_polynomial_degree_integer = 9

        for i in range (1, max_polynomial_degree_integer + 1):

            if display_progress_boolean == True:

                clear_output(wait = True)

                index_integer += 1
                                
                logx.print_and_log_text(f'Progress: {i}/{max_polynomial_degree_integer}')

            try:

                svm_model \
                    = SVC \
                        (kernel = best_hyperparameters_dictionary['parameters']['kernel'],
                         gamma = best_hyperparameters_dictionary['parameters']['gamma'],
                         class_weight = best_hyperparameters_dictionary['parameters']['class_weight'],
                         decision_function_shape = best_hyperparameters_dictionary['parameters']['decision_function_shape'],
                         shrinking = best_hyperparameters_dictionary['parameters']['shrinking'],
                         probability = best_hyperparameters_dictionary['parameters']['probablity'],
                         degree = i,
                         random_state = random_state_integer) \
                            .fit(x_train_scaled_dataframe, y_train_series)

            except:

                continue

            test_predictions_nparray \
                = svm_model.predict(x_test_scaled_dataframe)

            accuracy_score_float \
                = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
            if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

                best_hyperparameters_dictionary['parameters'] = svm_model.get_params()
                
                best_hyperparameters_dictionary['accuracy'] = accuracy_score_float

        
    if display_progress_boolean == True:
            
        logx.print_and_log_text('\n')

        
    return best_hyperparameters_dictionary


# In[7]:


#*******************************************************************************************
 #
 #  Function Name:  return_optimal_hyperparameters_knn
 #
 #  Function Description:
 #      This function returns the best hyperparameters from overall accuracy 
 #      for a k-nearest neighbor machine model.
 #
 #
 #  Return Type: dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          x_train_scaled_dataframe        
 #                          The parameter is the scaled training model features 
 #                          or x-variable.
 #  dataframe
 #          x_test_scaled_dataframe        
 #                          The parameter is the scaled test model features 
 #                          or x-variable.
 #  series  y_train_series  The parameter is the training target or y-variable.
 #  series  y_test_series   The parameter is the test target or y-variable.
 #  boolean
 #          display_progress_boolean
 #                          The parameter indicates whether the function will 
 #                          display progress.
 #  list n_neighbors_integer_list
 #                          The parameter is the range of the number of k neighbors.
 #  integer leaf_size_integer_list
 #                          The parameter is the range of leaf sizes.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_optimal_hyperparameters_knn \
        (x_train_scaled_dataframe,
         x_test_scaled_dataframe,
         y_train_series,
         y_test_series,
         display_progress_boolean = True,
         n_neighbors_integer_list = [5, 10],
         leaf_size_integer_list = [2, 5]):

    best_hyperparameters_dictionary = {'parameters': None, 'accuracy': 0.0}
        
       
    algorithm_string_list = ['auto', 'ball_tree', 'kd_tree', 'brute']

    weights_list = ['uniform', 'distance', None]

    p_integer_list = [1, 2]
        

    index_integer = 0
        
    upper_limit_integer \
        = len(algorithm_string_list) * len(weights_list) * len(p_integer_list) \
            * (n_neighbors_integer_list[1] - n_neighbors_integer_list[0] + 1)  \
            * (leaf_size_integer_list[1] - leaf_size_integer_list[0] + 1)

        
    for current_algorithm in algorithm_string_list:

        for current_weights in weights_list:

            for current_p in p_integer_list:

                for current_n_neighbors \
                        in range(n_neighbors_integer_list[0], n_neighbors_integer_list[1] + 1):

                    for current_leaf_size \
                            in range(leaf_size_integer_list[0], leaf_size_integer_list[1] + 1):
                    
                        if display_progress_boolean == True:

                            index_integer += 1

                            if index_integer % 100 == 0 \
                                or index_integer >= upper_limit_integer:
                                                                    
                                clear_output(wait = True)
                                
                                logx.print_and_log_text \
                                    (f'Progress: {index_integer}/{upper_limit_integer}')

                        try:
                                
                            knn_model \
                                = KNeighborsClassifier \
                                    (algorithm = current_algorithm,
                                     weights = current_weights,
                                     p = current_p,
                                     n_neighbors = current_n_neighbors,
                                     leaf_size = current_leaf_size) \
                                        .fit(x_train_scaled_dataframe, y_train_series)
                            
                        except:

                            continue

                        test_predictions_nparray \
                                = knn_model.predict(x_test_scaled_dataframe)

                        accuracy_score_float \
                            = balanced_accuracy_score(y_test_series, test_predictions_nparray)

        
                        if accuracy_score_float > best_hyperparameters_dictionary['accuracy']:

                            best_hyperparameters_dictionary['parameters'] = knn_model.get_params()
                
                            best_hyperparameters_dictionary['accuracy'] = accuracy_score_float
                                
        
    if display_progress_boolean == True:
            
        logx.print_and_log_text('\n')

        
    return best_hyperparameters_dictionary


# In[8]:


#*******************************************************************************************
 #
 #  Function Name:  return_predictions_dataframe
 #
 #  Function Description:
 #      This function returns the predictions dataframe from the x-y variables 
 #      and machine model.
 #
 #
 #  Return Type: dataframe
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  ML model  
 #          ml_model        The parameter is the scikit-learn machine learning model.
 #  dataframe
 #          x_scaled_dataframe
 #                          The parameter is the scaled model features or x-variable.
 #  series  y_series        The parameter is the target or y-variable.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2023          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_predictions_dataframe \
        (ml_model,
         x_scaled_dataframe,
         y_series):

    predictions_nparray = ml_model.predict(x_scaled_dataframe)

    predictions_dictionary = {'prediction': predictions_nparray, 'actual': y_series}

    return pd.DataFrame(predictions_dictionary)


# In[9]:


#*******************************************************************************************
 #
 #  Function Name:  return_binary_classification_confusion_matrix
 #
 #  Function Description:
 #      This function evaluates the y-variable test series and model predictions 
 #      of a binary classification model and returns its confusion matrix matrix.
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

    confusion_matrix_nparray = confusion_matrix(y_test_series, predictions_nparray)
        
        
    index_string_list \
        = ['Actual ' + true_outcome_string, 'Actual ' + false_outcome_string]

    column_string_list \
        = ['Predicted ' + true_outcome_string, 'Predicted ' + false_outcome_string]

    target_names_string_list \
        = [true_outcome_string.lower(), false_outcome_string.lower()]


    accuracy_score_float \
        = balanced_accuracy_score(y_test_series, predictions_nparray)
        
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


# In[ ]:




