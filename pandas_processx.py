#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  pandas_processx.py
 #
 #  File Description:
 #      This Python script, pandas_processxs.py, contains Python functions for processing 
 #      Pandas data structures. Here is the list:
 #
 #      return_standard_format_styler
 #      save_image_and_return_styler
 #      return_formatted_table
 #      return_summary_statistics_as_dataframe
 #      return_formatted_rows
 #      return_dataframe_description
 #      return_formatted_description
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/11/2024      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/

import logx_constants
import logx

import dataframe_image

import pandas as pd

pd.options.mode.chained_assignment = None


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'pandas_processx.py'


# In[3]:


#*******************************************************************************************
 #
 #  Function Name:  return_standard_format_styler
 #
 #  Function Description:
 #      This function returns a styler object in standard format from a dataframe.
 #
 #
 #  Return Type: styler
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          input_dataframe The parameter is the input dataframe.
 #  string  caption_string  The parameter is the table caption.
 #  integer precision_integer
 #                          This optional parameter is the decimal place 
 #                          precision of the displayed numbers.
 #  boolean hide_index_boolean
 #                          This optional parameter indicates whether the
 #                          index column is hidden or not.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_standard_format_styler \
        (input_dataframe,
         caption_string,
         precision_integer = 2,
         hide_index_boolean = True):
    
    try:
        
        temp_dataframe = input_dataframe.copy()
        
        
        if hide_index_boolean == True:
            
            return \
                temp_dataframe \
                    .style \
                    .set_caption(caption_string) \
                    .set_table_styles \
                        ([dict \
                             (selector = 'caption',
                              props = [('color', 'black'),
                                       ('font-size', '20px'),
                                       ('font-style', 'bold'),
                                       ('text-align', 'center')])]) \
                    .set_properties \
                         (**{'text-align':
                            'center',
                            'border':
                            '1.3px solid red',
                            'color':
                            'blue'}) \
                    .format \
                        (precision = precision_integer, 
                         thousands = ',', 
                         decimal = '.') \
                    .hide()
            
        else:
            
            return \
                temp_dataframe \
                    .style \
                    .set_caption(caption_string) \
                    .set_table_styles \
                        ([dict \
                             (selector = 'caption',
                              props = [('color', 'black'),
                                       ('font-size', '20px'),
                                       ('font-style', 'bold'),
                                       ('text-align', 'center')])]) \
                    .set_properties \
                         (**{'text-align':
                            'center',
                            'border':
                            '1.3px solid red',
                            'color':
                            'blue'}) \
                    .format \
                        (precision = precision_integer, 
                         thousands = ',', 
                         decimal = '.')
        
    except:
            
        logx.print_and_log_text \
            ('The function, return_standard_format_styler, '
             + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
             + 'was unable to format a dataframe as a styler object.')
        
        return None


# In[4]:


#*******************************************************************************************
 #
 #  Function Name:  save_image_and_return_styler
 #
 #  Function Description:
 #      This function saves the styler object as a png image then returns the object.
 #
 #
 #  Return Type: styler
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  styler  input_styler    The parameter is the input styler object.
 #  string  caption_string  The parameter is the table caption.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def save_image_and_return_styler \
        (input_styler,
         caption_string):
    
    try:
        
        if logx_constants.IMAGE_FLAG == True:

            image_file_path_string = logx.get_image_file_path(caption_string, 'png')
        
            dataframe_image.export(input_styler, image_file_path_string)
        
        return input_styler
        
    except:
        
        logx.print_and_log_text \
            ('The function, save_image_and_return_styler, '
             + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
             + 'cannot save an image of a styler object ' \
             + 'then return it to the caller.')
        
        return None


# In[5]:


#*******************************************************************************************
 #
 #  Function Name:  return_formatted_table
 #
 #  Function Description:
 #      This function returns a formatted table from a dataframe.
 #
 #
 #  Return Type: styler
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          input_dataframe The parameter is the input dataframe.
 #  string  caption_string  The parameter is the table caption.
 #  integer line_count_integer
 #                          The parameter is the number of displayed records.
 #  boolean hide_index_boolean
 #                          The parameter indicates whether the index is present.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_formatted_table \
        (input_dataframe,
         caption_string,
         line_count_integer = 10,
         hide_index_boolean = True):

    current_styler \
        = return_standard_format_styler \
            (input_dataframe.head(line_count_integer),
             caption_string, 
             hide_index_boolean = hide_index_boolean)

    return save_image_and_return_styler(current_styler, caption_string)


# In[6]:


#*******************************************************************************************
 #
 #  Function Name:  return_summary_statistics_as_dataframe
 #
 #  Function Description:
 #      This function converts a data series into summary statistics, assigns
 #      the statistics to a dataframe, and returns it.
 #
 #
 #  Return Type: dataframe
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  series  data_series     The parameter is the input series.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_summary_statistics_as_dataframe(data_series):

    try:
        
        # This line of code allocates the distribution for the quartiles.
        quartiles_series = data_series.quantile([0.25, 0.50, 0.75])
    
        # These lines of code establish the lower quartile and the upper quartile.
        lower_quartile_float = quartiles_series[0.25]

        upper_quartile_float = quartiles_series[0.75]
    
        # This line of code calculates the interquartile range (IQR).
        interquartile_range_float = upper_quartile_float - lower_quartile_float

        # These line of code calculate the lower bound and upper bound 
        # of the distribution.
        lower_bound_float = lower_quartile_float - (1.5*interquartile_range_float)
    
        upper_bound_float = upper_quartile_float + (1.5*interquartile_range_float)
    
        # This line of code establishes a list of outliers.
        outliers_series \
            = data_series.loc[(data_series < lower_bound_float) | (data_series > upper_bound_float)]
        
        # This line of code finds the number of outliers.
        number_of_outliers_integer = len(outliers_series)
  
        # These lines of code create a list of all the summary statistics and store
        # the data in a DataFrame.
        statistics_dictionary_list \
            = [{'Lower Quartile': lower_quartile_float,
                'Upper Quartile': upper_quartile_float,
                'Interquartile Range': interquartile_range_float,
                'Median': quartiles_series[0.5],
                'Lower Boundary': lower_bound_float,
                'Upper Boundary': upper_bound_float,
                'Number of Outliers': number_of_outliers_integer}]
  
        return pd.DataFrame(statistics_dictionary_list)
        
    except:
            
        logx.print_and_log_text \
            ('The function, return_summary_statistics_as_dataframe, '
             + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
             + 'cannot return summary statistics as a dataframe.')
            
        return None


# In[7]:


#*******************************************************************************************
 #
 #  Function Name:  return_formatted_rows
 #
 #  Function Description:
 #      This function formats the rows in a pandas styler and returns it.
 #
 #
 #  Return Type: styler
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  styler  input_styler    The parameter is the input styler.
 #  dictionary
 #          format_dictionary
 #                          The parameter is the dicitionary with the format specifications.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_formatted_rows \
        (input_styler, 
         format_dictionary):
    
    try:
    
        for key, value in format_dictionary.items():
        
            row_number = input_styler.index.get_loc(key)

            for column_number in range(len(input_styler.columns)):
            
                input_styler._display_funcs[(row_number, column_number)] = value
            
            
        return input_styler

    except:
        
        logx.print_and_log_text \
            ('The function, return_formatted_rows, '
             + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
             + 'was unable to format the rows in a pandas styler.')
        
        return None


# In[8]:


#*******************************************************************************************
 #
 #  Function Name:  return_dataframe_description
 #
 #  Function Description:
 #      This function takes a dataframe and returns the its formatted data statistics.
 #
 #
 #  Return Type: styler
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          input_dataframe The parameter is the input dataframe.
 #  string  caption_string  The parameter is the text for the caption.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_dataframe_description \
        (input_dataframe,
         caption_string):
    
    try:
        
        description_dataframe = input_dataframe.describe()
    
        format_dictionary \
            = {'count': lambda x: f'{x:,.0f}',
               'mean': lambda x: f'{x:,.2f}',
               'std': lambda x: f'{x:,.2f}',
               'min': lambda x: f'{x:,.0f}',
               '25%': lambda x: f'{x:,.2f}',
               '50%': lambda x: f'{x:,.2f}',
               '75%': lambda x: f'{x:,.2f}',
               'max': lambda x: f'{x:,.0f}'}

        description_styler \
            = return_formatted_rows(description_dataframe.style, format_dictionary)
        
        description_styler \
            .set_caption(caption_string) \
            .set_table_styles \
                ([{'selector': 'caption', 
                   'props': [('color', 'black'), 
                             ('font-size', '16px'),
                             ('font-style', 'bold'),
                             ('text-align', 'center')]}]) \
            .set_properties \
                (**{'text-align': 'center',
                    'border': '1.3px solid red',
                    'color': 'blue'})
            
        return description_styler
        
    except:
        
        logx.print_and_log_text \
            ('The function, return_dataframe_description, '
             + f'in source file, {CONSTANT_LOCAL_FILE_NAME}, '
             + "cannot display the DataFrame's description.")
        
        return None 


# In[9]:


#*******************************************************************************************
 #
 #  Function Name:  return_formatted_description
 #
 #  Function Description:
 #      This function returns a formatted dataframe description.
 #
 #
 #  Return Type: styler
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          input_dataframe The parameter is the input dataframe.
 #  string  caption_string  The parameter is the text for the caption.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_formatted_description \
        (input_dataframe,
         caption_string):

    current_styler = return_dataframe_description(input_dataframe, caption_string)

    return save_image_and_return_styler(current_styler, caption_string)


# In[ ]:




