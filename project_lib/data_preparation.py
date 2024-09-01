##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import pandas as pd
import numpy as np
from typing import Union
from project_lib.initial_config import initial_settings
from project_lib.input_validation import validate_input_types

# set the basic cofiguration for this module
initial_settings()

################################
####### MODULE FUNCTIONS #######
################################


def get_nested_unique_values(dataframe: pd.DataFrame, column: str) -> list:
    """
    Get values nested on rows of a given column. Check the note below to understand
    the expected data for the given colum.

    Args
        dataframe: a pandas dataframe
        column: a string with the column name to be used

    Return
        list_unique_processed: a list with unique nested values on the given column

    NOTE:
        If the the column has the following rows:
            '[]', '[A]', '[B]', '[A, B]', '[A, B, C, D]'
        Then the return list will be:
            [A, B, C, D ]
    """

    # import required libraries
    import re
    import pandas as pd

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"column": column}, (str,))

    # get the unique values for the given column
    list_unique_raw = dataframe[column].unique()

    # make sure all unique values are of string type
    series_raw_prep = [
        "[]" if not isinstance(item, str) else item for item in list_unique_raw
    ]

    # remove leading and trailing square brackets -> e.g. '[Inbound]' or '[Canais]'
    series_raw_items = [item[1:-1] for item in series_raw_prep]

    # split item on comma + single space
    series_split_items = [re.split(", ", item) for item in series_raw_items]

    # instanciate final list of unique values
    list_unique_processed = []

    # iterate over raw items
    for row in series_split_items:
        # iterate over item on row -> row are lists
        for item in row:
            # check if item is already in final list
            if item not in list_unique_processed:
                # append item to final list
                list_unique_processed.append(item)

    return list_unique_processed


def count_nested_values(
    dataframe: pd.DataFrame, column: str, list_unique_processed: list
) -> dict:
    """
    It get the output of the _get_nested_unique_values function and
    return a dict with the number of ocurrences for every unique value.
    Please check note below to understand how this function works.

    Args
        dataframe: a pandas dataframe
        column: a string with the column name to be used
        list_unique_processed: a list with unique nested values on the given column.
            Ideally, this params should be the output of
            the _get_nested_unique_values function of this module

    Return
        dict_values_count: a dict with the number of ocurrences for every unique value
            in list_unique_processed list

    NOTE:
        The dataframe column is expected to have rows similar to this:
            '[]', '[A]', '[B]', '[A, B]', '[A, B, C, D]'
    """

    # import required libraries
    import re
    import pandas as pd

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"column": column}, (str,))
    validate_input_types({"list_unique_processed": list_unique_processed}, (list,))

    # create a dictionary whose keys are unique items and
    # values are the number of occurences
    dict_values_count = {item: 0 for item in list_unique_processed}

    # get the unique values for the given column
    dataframe[column]

    # make sure all unique values are of string type
    series_raw_prep = dataframe[column].apply(
        lambda x: "[]" if not isinstance(x, str) else x
    )

    # remove leading and trailing square brackets -> e.g. '[Inbound]' or '[Canais]'
    series_raw_items = series_raw_prep.apply(lambda x: str(x)[1:-1])

    # split item on comma + single space
    series_split_items = series_raw_items.apply(lambda x: re.split(", ", x))

    # iterate over raw items
    for row in series_split_items:
        # iterate over item on row -> row are lists
        for item in row:
            # check if item is already in dict_values_count
            if item in dict_values_count.keys():
                # add item to final dict
                dict_values_count[item] += 1

    return dict_values_count


def count_nested_unique_values(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Get values nested on rows of a given column and return the number of ocurrences
    for every unique values. Check the note below to understand the expected data
    for the given colum.

    Args
        dataframe: a pandas dataframe
        column: a string with the column name to be used

    Return
        df_values_count: a pandas dataframe with the number of ocurrences for every unique value
            in list_unique_processed list

    NOTE:
        The dataframe column is expected to have rows similar to this:
            '[]', '[A]', '[B]', '[A, B]', '[A, B, C, D]'
    """

    # import required libraries
    import pandas as pd

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"column": column}, (str,))

    # use get_nested_unique_values function
    list_unique_processed = get_nested_unique_values(dataframe=dataframe, column=column)

    # use count_nested_values function
    dict_values_count = count_nested_values(
        dataframe=dataframe, column=column, list_unique_processed=list_unique_processed
    )

    # rename empty key to NA -> easier to understand
    dict_values_count["NA"] = dict_values_count[""]
    del dict_values_count[""]

    # create a dataframe from dictionary
    df_values_count = pd.DataFrame(data=dict_values_count, index=["count_values"]).T
    df_values_count.sort_values(by="count_values", ascending=False, inplace=True)

    return df_values_count


def split_nested_values_into_cols(
    dataframe: pd.DataFrame, col: str, copy: bool = True
) -> pd.DataFrame:
    """
    Get the nested values within a colum and create as many columns
    as there are unique values in that column. Then count the number of unique values
    for every row. Check the "NOTE" in the end of this doc for further details.

    Args
        dataframe: a pandas dataframe
        col: a string with the column name to be used
        copy: a boolean to indicate if user wants to return a copy of the input dataframe

    Return
        df_processing: a pandas dataframe with the processed data.
            Check the "NOTE" in the end of this doc for further details.

    NOTE:
        The expected input dataframe is as follows
                col
        0       [A]
        1       [A, B]
        2       [A, B, C]
        3       [E]

        Then the output will be as follows:
                col         A       B       C       E
        0       [A]         1       0       0       0
        1       [A, B]      1       1       0       0
        2       [A, B, C]   1       1       1       0
        3       [E]         0       0       0       1
        4       [A, A, C]   2       0       1       0
    """

    # import required libraries
    import pandas as pd

    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"column": col}, (str,))
    validate_input_types({"copy": copy}, (bool,))

    # check if user wants a copy
    if copy:
        # make a copy of original dataframe
        df_processing = dataframe.copy()
    # user don't want a copy
    else:
        # define df_processing variable
        df_processing = dataframe

    # use get_nested_unique_values to get unique nested values in the given column
    new_cols = get_nested_unique_values(df_processing, col)

    # create a new colum for every unique value
    for col_ in new_cols:
        # set the new column as a constant
        df_processing[col_] = 0

    # iterate over rows on dataframe
    for idx, row in df_processing.iterrows():

        # for the given row, get the nested values in he column=col
        # and then create a dataframe from it to reuse get_nested_unique_values function
        df_single_row = pd.DataFrame(row[col], index=[0], columns=[col])

        # get the nested unique values for that value
        unique_values = get_nested_unique_values(df_single_row, col)

        # iterate over found unique values
        for item in unique_values:
            # add the unique value found the its respective column
            df_processing.loc[idx, item] += 1

    # make sure there is no "" string -> easier to understand columns
    df_processing.columns = [
        "NA" if item == "" else item for item in df_processing.columns
    ]

    return df_processing


def get_diff_in_months(
    reference_date_series: Union[pd._libs.tslibs.timestamps.Timestamp, pd._libs.tslibs.nattype.NaTType],
    moving_date_series: Union[pd._libs.tslibs.timestamps.Timestamp, pd._libs.tslibs.nattype.NaTType]
) -> np.ndarray:
    """
    Get two datetime arrays and calculate the difference in months between them.
    
    Args
        reference_date_series: a datetime series with the reference dates.
        moving_date_series: a datetime series with the moving dates.
        
    Return
        diff_in_month: an array with the diffence between date series.
            Note that the calculating is made by "moving_date_series - reference_date_series",
            so a positive values means "moving_date_series" is posterior to "reference_date_series".

    NOTE:
        This function was created aiming specifically an dataframe.apply(lambda X: ... , axis=1) 
    """

    # input verification
    validate_input_types(
        {"reference_date_series": reference_date_series}, 
        (pd._libs.tslibs.timestamps.Timestamp, pd._libs.tslibs.nattype.NaTType)
        )
    validate_input_types(
        {"moving_date_series": moving_date_series}, 
        (pd._libs.tslibs.timestamps.Timestamp, pd._libs.tslibs.nattype.NaTType)
        )

    
    # get year and month info from both series
    reference_date_year = reference_date_series.year
    reference_date_month = reference_date_series.month
    moving_date_year = moving_date_series.year
    moving_date_month = moving_date_series.month    
    
    # subtract dates and get results in months
    diff_in_months = (moving_date_year*12 + moving_date_month) - (reference_date_year*12 + reference_date_month)
    
    return diff_in_months
