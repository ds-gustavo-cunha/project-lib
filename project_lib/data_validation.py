##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import numpy as np
import pandas as pd
from datetime import datetime
from os.path import splitext, exists
from project_lib.initial_config import initial_settings
from project_lib.data_description import check_duplicating_columns
from project_lib.input_validation import (
    validate_input_types,
    validate_input_values,
    validate_dataframe_cols,
)


# set the basic cofiguration for this module
initial_settings()


################################
####### MODULE FUNCTIONS #######
################################


class DataValidator:
    """
    Validate columns of the given dataframe based on the operations specified on a validation dictionaries.
    It can also validate duplicated rows based on the dataframe granalarity.
    Besides, it can save validations to a plain text file or csv file by means of save_validations method
    and it can also analyse validations historically by means of plot_historical_report method.

    Args
        dataframe: a pandas dataframe to validate columns and duplicated granularity
        col_dtypes: a dictionary with the required dtypes for each column (CHECK NOTE BELOW).
        col_funcions_checker: a dictionary with the values to validate (CHECK NOTE BELOW).
        col_aggregations_checker: a dictionary with the values to validate (CHECK NOTE BELOW).
        dataframe_granularity: a list with the column names to check duplicated granalarity for the given table.
        records_file: a string with the file path to write (or append) validations.
            It must have a .txt or .csv extension.

    Return
        None: a NoneType object


    NOTE:

        The dataset used on the following example was taken from
            https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales?resource=download

        >>> df.validation.sample(3)
             Quantity Branch  Gender  Unit price  Quantity       Date       City
        713         9      C  Female      13.850         9 2019-02-04  Naypyitaw
        651         6      B  Female      55.810         6 2019-01-22   Mandalay
        605         4      B    Male      31.750         4 2019-02-08   Mandalay

        >>> # define column dtypes to validate
        >>> col_dtypes = {
                "object": ["Branch", "Branch", "Gender", "City"],
                "integer": ["Quantity", "Quantity"],
                "float": ["Unit price"],
                "datetime64[ns]": ["Date"]
            }

        >>> # define functions to check columns
        >>> col_funcions_checker = {
                "Branch": [ # for Branch column
                    ("nunique", 3), # number of unique values for Branch column must be 3
                    ("NA_percent", [0, 5]) # percentage of NaN for Branch column must be within 0% and 5% [limits are included]
                ],
                "Gender": [ # for Gender column
                    ("nunique", 3), # number of unique values for Gender column must be 3
                    ("NA_total", 0) # total number of NaN values for Gender column must be 0
                ],
                "Unit price": [ # for 'Unit price' column
                    ("min", 0), # minimum value for 'Unit price' column must be 0 [equal or greater than 0]
                    ("median", [0, 10]) # median value for 'Unit price' column must be within 0 and 10 [limits are included]
                ],
                "Quantity": [ # for Quantity column
                    ("mean", [5, 25]) # mean value for Quantity column must be within 5 and 25 [limits are included]
                ],
                "Date": [ # for Date column
                    ("min", datetime.datetime(2020, 1, 1) ), # minimum value for 'Date' column must be 2020-01-01 [equal or greater than 2020-01-01]
                    ("max", datetime.datetime(2022, 12, 31) ) # maximum value for 'Date' column must be 2022-12-31 [equal or less than 2022-12-31]
                ]
            }

        Accepted values for the first item inside the tuple of col_funcions_checker:
        ("min", "max", "mean", "median", "nunique", "NA_total", "NA_percent")


        >>> # define aggregation checks
        >>> col_aggregations_checker = {
                    "Invoice ID": [ # group by "Invoice ID" column
                        # the maximum number of cities per "Invoice ID" must be equal or less than 1
                        ("City", "nunique", "max", 1), # df_validation.groupby( "Invoice ID" ).agg( {"City":"nunique"} ).max() <= 1
                        # the minimum number of date per "Invoice ID" must be equal of greater than 0
                        ("Date", "nunique", "min", 0) # df_validation.groupby( "Invoice ID" ).agg( {"Date" : "nunique"} ).min() >= 0
                    ],
                    "City":[ # group by "City" column
                        # the minimum of the mean of total per city must be equal or greater than 99
                        ("Total", "mean", "min", 99) # df_validation.groupby( "City" ).agg( {"Total" : "mean"} ).min() >= 99
                    ]
                }

        Accepted values for the second item inside the tuple of col_aggregations_checker:
            ("mean", "sum", "size", "count", "first", "last", "min", "max", "nunique")
        Accepted values for the third item inside the tuple of col_aggregations_checker:
            ("min", "max", "mean", "median", "nunique", "NA_total", "NA_percent")



        >>> # define dataframe granularity
        >>> df_grain = ["Invoice ID"] # granularity of the given dataframe is only column "Invoice ID". No problem if was more than one column!


        >>> # instanciate data validator object
        >>> dv = DataValidator(dataframe=df_validation,
                               col_funcions_checker=col_funcions_checker,
                               dataframe_granularity=df_grain,
                               col_aggregations_checker=col_aggregations_checker)
        >>> # validate dataframe
            dv.validate_data(display_validation=True)

            timestamp      grouby column        column                        condition                  result check                 result stats                                       notes                          duplicated rows checker column  percentage (%) of duplicated rows number of duplicated rows duplicating columns
        2022-11-15 16:56:58                            Branch                                 nunique = 3      ✅                                    nunique = 3.00 NA values were not considered on NUNIQUE calculation
        2022-11-15 16:56:58                            Branch                        0 <= NA_percent <= 5      ✅                                 NA_percent = 0.00
        2022-11-15 16:56:58                            Gender                                 nunique = 3      ❌                                    nunique = 2.00 NA values were not considered on NUNIQUE calculation
        2022-11-15 16:56:58                            Gender                                NA_total = 0      ✅                                      NA_total = 0
        2022-11-15 16:56:58                        Unit price                                     min = 0      ✅                                       min = 10.08     NA values were not considered on MIN calculation
        2022-11-15 16:56:58                        Unit price                           0 <= median <= 10      ❌                                    median = 55.23  NA values were not considered on MEDIAN calculation
        2022-11-15 16:56:58                          Quantity                             5 <= mean <= 25      ✅                                       mean = 5.51    NA values were not considered on MEAN calculation
        2022-11-15 16:56:58                              Date                   min = 2020-01-01 00:00:00      ❌                         min = 2019-01-01 00:00:00     NA values were not considered on MIN calculation
        2022-11-15 16:56:58                              Date                   max = 2022-12-31 00:00:00      ✅                         max = 2019-03-30 00:00:00     NA values were not considered on MAX calculation
        2022-11-15 16:56:58   Invoice ID   aggregated on City max of (nunique of City PER Invoice ID) = 1      ✅       max of (nunique of City PER Invoice ID) = 1     NA values were not considered on MAX calculation
        2022-11-15 16:56:58   Invoice ID   aggregated on Date min of (nunique of Date PER Invoice ID) = 0      ❌       min of (nunique of Date PER Invoice ID) = 1     NA values were not considered on MIN calculation
        2022-11-15 16:56:58         City  aggregated on Total        min of (mean of Total PER City) = 99      ❌          min of (mean of Total PER City) = 312.35     NA values were not considered on MIN calculation
        2022-11-15 16:56:58                                                                                    ✅                                                                     duplicated rows are DOUBLE COUNTED based on columns: ['Invoice ID']               0.00%                           0


    METHODS:

        DataValidator.validate_data(
            self,
            display_validation: bool = True
        ) -> None:
            Given the input, validate all of the required data

            Args
                display_validation: a boolean to indicate if 
                    validation will be displayed after executed

            Return
                self.df_records: a pandas dataframe with validation summary


        DataValidator.pass_validations(
            self
        ) -> None:
        Check if all validations were successful 

        Return
            passing_flag: a boolean with
                True if all validation was successful;
                False if at least one validation was failed


        DataValidator.save_validations(
            self,
            records_file: str = None
        ) -> None:
            Save validations to a csv or txt file.

            Args
                records_file: a string with the file path to write (or append) validations.
                    It must have a .txt or .csv extension.

            Return
                None: a NoneType object


        DataValidator.plot_historical_report(
            self,
            records_file: str = None,
            datetime_format: str = "%Y-%m-%d",
            save_report: bool = True,
        ) -> None:
            Plot the historical overview about validations.

            Args
                records_file: a string with the file path to the validation csv.
                datetime_format: a string with the datetime granularity for plotting purpose.
                    Ex.:
                        If you want to plot report day by day -> "%Y-%m-%d"
                        If you want to plot report month by month -> "%Y-%m"
                save_report: a boolean to indicate if reports are to be saved
                    on the same folder as the records file.

            Return
                None: a NoneType object
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        col_dtypes: dict = None,
        col_funcions_checker: dict = None,
        col_aggregations_checker: dict = None,
        pandas_queries: list = None,
        dataframe_granularity: list = None,
        records_file: str = None,
    ) -> None:

        # validate inputs
        self.input_validation_(
            dataframe=dataframe,
            col_dtypes=col_dtypes,
            col_funcions_checker=col_funcions_checker,
            col_aggregations_checker=col_aggregations_checker,
            pandas_queries=pandas_queries,
            dataframe_granularity=dataframe_granularity,
            records_file=records_file
        )

        # create attributes according to inputs
        self.dataframe = dataframe
        self.col_dtypes = col_dtypes
        self.col_funcions_checker = col_funcions_checker
        self.col_aggregations_checker = col_aggregations_checker
        self.pandas_queries = pandas_queries
        self.dataframe_granularity = dataframe_granularity
        self.records_file = records_file

        # define a dictionary to store results
        records = {
            "timestamp": [],
            "grouby column": [],
            "column": [],
            "pandas query": [],
            "condition": [],
            "result check": [],
            "result stats": [],
            "notes": [],
            "duplicated rows checker column": [],
            "percentage (%) of duplicated rows": [],
            "number of duplicated rows": [],
            "duplicating columns": [],
        }

        # set records dict as an attribute
        self.records = records

        # define dict with emojis
        self.emoji_dict = {True: "✅", False: "❌"}

        # get datetime validation
        self.val_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return None

    def validate_data(
        self,
        display_validation: bool = True
    ) -> None:
        """
        Given the input, validate all of the required data

        Args
            display_validation: a boolean to indicate if 
                validation will be displayed after executed

        Return
            self.df_records: a pandas dataframe with validation summary
        """
        # check if display_validation is boolean
        if display_validation is not None:
            # check display_validation type
            validate_input_types({"display_validation": display_validation}, (bool,))

        # check if a col_dtypes was input
        if self.col_dtypes is not None:
            # call checker
            self.validate_col_dtypes_()

        # check if a pandas_queries was input
        if self.pandas_queries is not None:
            # call checker
            self.validate_pandas_queries_()

        # check if a col_funcions_checker was input
        if self.col_funcions_checker is not None:
            # call checker
            self.validate_column_functions_()

        # check if a col_aggregations_checker was input
        if self.col_aggregations_checker is not None:
            # call checker
            self.validate_agg_functions_()

        # check if a dataframe_granularity was input
        if self.dataframe_granularity is not None:
            # call checker
            self.validate_dataframe_granularity_()

        # check if there was any input to validate
        if len(self.records["timestamp"]) > 0:

            # create a dictionary with records as a attribute
            self.df_records = pd.DataFrame(self.records)

            # check if validation has to be displayed
            if display_validation:
                # define variables to print message
                BOLD = "\033[1m"
                ENDC = "\033[0m"

                # print important message
                print(
                    f"\n{'*'*49}\n"
                    f"\t{BOLD}Remember that you must ensure dataframe data types (dtypes) are correct in regard to the validations you want to check{ENDC}.\n"
                    f"\t{BOLD}If the dtypes for some column is incorrect then there may be errors on validation.{ENDC}\n"
                    f"\t{BOLD}E.g.: trying to check the median of a string column may raise errors or return unexpected results.{ENDC}"
                    f"\n{'*'*49}\n"
                )

                # return results to user
                display(self.df_records)

            # check if a file path was input
            if self.records_file is not None:
                # save records
                self.save_validations(self.records_file)

        return self.df_records
                

    def input_validation_(
        self,
        dataframe: pd.DataFrame,
        col_dtypes: dict = None,
        col_funcions_checker: dict = None,
        col_aggregations_checker: dict = None,
        pandas_queries: list = None,
        dataframe_granularity: list = None,
        display_validation: bool = True,
        records_file: str = None,
    ) -> None:
        """
        Validate class inputs according to expected input formats, types and values
        """

        # dataframe param type
        validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))

        # define already prepared functions and aggregations
        prepared_functions = (
            "min",
            "max",
            "mean",
            "median",
            "nunique",
            "NA_total",
            "NA_percent",
        )
        prepared_agg = (
            "mean",
            "sum",
            "size",
            "count",
            "first",
            "last",
            "min",
            "max",
            "nunique",
        )

        # check if col_dtypes was input
        if col_dtypes is not None:
            # validate col_funcions_checker type
            validate_input_types(
                {"col_dtypes": col_dtypes}, (dict,)
            )
            # iterate over col_funcions_checker dict
            for dtype, cols in col_dtypes.items():
                # dtype must be a string
                validate_input_types(
                    {"dtype":dtype},
                    (str,),
                    f"Keys of col_dtypes dict must be strings with valid dtypes to check. {dtype} is not a string."
                )
                # dtype names must be within defined ones
                validate_input_values(
                    {"dtype": dtype},
                    ('object', 'integer', 'float', 'datetime64[ns]'),
                    f"dtype names must be one of ['object', 'integer', 'float', 'datetime64[ns]'] possible values so far. {dtype} is not a valid dtype.",                
                )
                # validate col_funcions_checker type
                validate_input_types(
                    {"cols": cols},
                    (list,),
                    f"Values of col_dtypes dict must be lists with column names. {cols} is not a list."
                )
                # iterate over available column names
                for col in cols:
                    # dtype must be a string
                    validate_input_types(
                        {"col":col},
                        (str,),
                        f"Values of col_dtypes dict must be lists of strings where strings are the column names. {col}, in '{dtype}': {cols}, is not a string."
                    )
                    # column must be in dataframe
                    validate_dataframe_cols(
                        dataframe,
                        (col,),
                        f"Values of col_dtypes dict must be lists of strings where strings are the column names. {col}, in '{dtype}': {cols}, is not a valid column name."
                    )

        # check if col_funcions_checker was input
        if col_funcions_checker is not None:
            # validate col_funcions_checker type
            validate_input_types(
                {"col_funcions_checker": col_funcions_checker}, (dict,)
            )
            # iterate over col_funcions_checker dict
            for col, func_list in col_funcions_checker.items():
                # column must be in dataframe
                validate_dataframe_cols(
                    dataframe,
                    (col,),
                    f"Keys of col_funcions_checker dict must be strings with column names of the input dataframe: '{col}' is not a dataframe column.",
                )
                # validate col_funcions_checker type
                validate_input_types(
                    {"func_list": func_list},
                    (list,),
                    f"Values of col_funcions_checker dict must be lists with function informations to be tested: {func_list} is not a list. Check the validation code for column '{col}'.",
                )
                # iterate over func_list
                for testing_tuple in func_list:
                    # validate testing_tuple type
                    validate_input_types(
                        {"testing_tuple": testing_tuple},
                        (tuple,),
                        f"Items to check each column must be tuples: {testing_tuple} is not an accepted type. Check the validation code for column '{col}'.",
                    )
                    # check item types
                    validate_input_types(
                        {"item[0]": testing_tuple[0]},
                        (str,),
                        f"Function names inside col_funcions_checker dict must be strings: {testing_tuple[0]} is not a string. Check the validation code for column '{col}'.",
                    )
                    validate_input_types(
                        {"item[1]": testing_tuple[1]},
                        (int, float, list, datetime),
                        f"Testing conditions inside col_funcions_checker dict must be int, float, datetime.datetime or list: {testing_tuple[1]} was not an accepted type. Check the validation code for column '{col}'.",
                    )
                    # function names must be within defined ones
                    validate_input_values(
                        {"item[0]": testing_tuple[0]},
                        prepared_functions,
                        f"Function names must be one of {list(prepared_functions)} possible values so far: {testing_tuple[0]} was not an accepted function. Check the validation code for column '{col}'.",
                    )

        # check if col_aggregations_checker was input
        if col_aggregations_checker is not None:
            # validate col_aggregations_checker type
            validate_input_types(
                {"col_funcions_checker": col_aggregations_checker},
                (dict,),
                f"Param col_aggregations_checker must be a dictionary.",
            )
            # iterate over col_funcions_checker dict
            for col, func_list in col_aggregations_checker.items():
                # column must be in dataframe
                validate_dataframe_cols(
                    dataframe,
                    (col,),
                    f"Keys of col_aggregations_checker dict must be strings with column names of the input dataframe. Column '{col}' is not on the dataframe. Check col_aggregations_checker dictionary.",
                )
                # validate col_funcions_checker type
                validate_input_types(
                    {"func_list": func_list},
                    (list,),
                    f"Values of col_aggregations_checker dict must be lists with function informations to be tested: {func_list} was not a accepted input. Check aggregations for column '{col}'",
                )
                # iterate over func_list
                for testing_tuple in func_list:
                    # validate testing_tuple type
                    validate_input_types(
                        {"testing_tuple": testing_tuple},
                        (tuple,),
                        f"Testing params inside col_aggregations_checker must be tuples: {testing_tuple} was not a accepted input. Check aggregations for column '{col}'.",
                    )
                    # check item types
                    validate_input_types(
                        {"item[0]": testing_tuple[0]},
                        (str,),
                        f"First testing param inside col_aggregations_checker must be strings: {testing_tuple[0]} was not a accepted input. Check aggregations for column '{col}'.",
                    )
                    validate_dataframe_cols(
                        dataframe,
                        (testing_tuple[0],),
                        f"First testing param inside col_aggregations_checker must strings with column names of the input dataframe: '{testing_tuple[0]}' is not a column of the dataframe. Check aggregations for column '{col}'.",
                    )
                    validate_input_types(
                        {"item[1]": testing_tuple[1]},
                        (str,),
                        f"Second testing param inside col_aggregations_checker must strings: {testing_tuple[1]} was not a accepted input. Check aggregations for column '{col}'.",
                    )
                    validate_input_values(
                        {"item[1]": testing_tuple[1]},
                        prepared_agg,
                        f"Second testing param inside col_aggregations_checker a must string with one of {list(prepared_agg)} possible values so far: {testing_tuple[1]} was not a accepted input. Check aggregations for column '{col}'.",
                    )
                    validate_input_types(
                        {"item[2]": testing_tuple[2]},
                        (str,),
                        f"Third testing param inside col_aggregations_checker must strings: {testing_tuple[2]} was not a accepted input. Check aggregations for column '{col}'.",
                    )
                    validate_input_values(
                        {"item[2]": testing_tuple[2]},
                        prepared_functions,
                        f"Third testing param inside col_aggregations_checker must strings with one of {list(prepared_functions)} possible values so far. {testing_tuple[2]} was not a accepted input. Check aggregations for column '{col}'.",
                    )
                    validate_input_types(
                        {"item[3]": testing_tuple[3]},
                        (int, float, list),
                        f"Third testing param inside col_aggregations_checker must be int, float or list: {testing_tuple[3]} was not a accepted input. Check aggregations for column '{col}'.",
                    )

        # check if col_aggregations_checker was input
        if pandas_queries is not None:
            # check dataframe_granularity type
            validate_input_types(
                {"pandas_queries": pandas_queries},
                (list,),
                f"Param pandas_queries must be of types: [<class 'list'>]. {pandas_queries} is not a list!",
            )

            # iterate over queries
            for pandas_query in pandas_queries:
                # check dataframe_granularity type
                validate_input_types(
                    {"pandas_query": pandas_query},
                    (list,),
                    f"Each pandas query to test must be a list. {pandas_query} in query {pandas_query} is not a list!",
                )

                # check list len
                if len(pandas_query) != 3:
                    # raise value error
                    raise ValueError(
                        f"Each pandas query to test must be a list three items. {pandas_query} is not a valid list!"
                    )

                # upack pandas query
                query, testing_type, testing_values = (*pandas_query,)

                # check query
                validate_input_types(
                    {"Pandas queries": query},
                    (str,),
                    f"First argument for every query to test must be a string with the chosen query. {query} in query {pandas_query} is not a string!",
                )

                # check testing type
                validate_input_types(
                    {"Testing type": testing_type},
                    (str,),
                    f"Second argument for every query to test must be a string with 'absolute' or 'percentual'. {testing_type} in query {pandas_query} is not a string!",
                )

                # check testing type values
                if testing_type not in ("absolute", "percentual"):
                    # raise error
                    raise ValueError(
                        f"Second argument for every query to test must be a string with 'absolute' or 'percentual'. {testing_type} in query {pandas_query} is not a valid input!"
                    )

                # check testing params
                validate_input_types(
                    {"Testing values": testing_values},
                    (
                        int,
                        float,
                        list,
                    ),
                    f"Third argument for every query to test must be of types: [<class 'int'>, <class 'float'>, <class 'list'>]. {testing_values} in query {pandas_query} is not a valid input!",
                )

                # check testing values as a list
                if isinstance(testing_values, list):
                    # check list len
                    if len(testing_values) != 2:
                        # raise value error
                        raise ValueError(
                            f"If there are more than one value to test a query, values must be inside of a list with only two items. {testing_values} in query {pandas_query} was not allowed!"
                        )

                    # iterate over items
                    for item in testing_values:
                        validate_input_types(
                            {"Item": item},
                            (
                                int,
                                float,
                            ),
                            f"If there are more than one value to test a query, values must be int or float. {item} in query {pandas_query} is not a int or a float!",
                        )

        # check if dataframe_granularity was input
        if dataframe_granularity is not None:
            # check dataframe_granularity type
            validate_input_types(
                {"dataframe_granularity": dataframe_granularity},
                (list,),
                "Param dataframe_granularity must be a list.",
            )
            # iterate over values in dataframe_granularity
            for col in dataframe_granularity:
                # column must be in dataframe
                validate_dataframe_cols(
                    dataframe,
                    (col,),
                    f"values of dataframe_granularity must be string with column names of the input dataframe: '{col}' is not a dataframe column.",
                )

        # check if records_file was input
        if records_file is not None:
            # check records_file type
            validate_input_types({"records_file": records_file}, (str,))

    def calculate_stats_(
        self, dataframe: pd.DataFrame, func_name: str, col_name: str
    ) -> float:
        """
        Calculate the func_name statistics for the col_name column of the given dataframe
        """

        # check if funcion is minimum
        if func_name == "min":
            # calculate minimum -> skip NA
            stats = dataframe[col_name].min(skipna=True)
        # check if funcion is maximum
        elif func_name == "max":
            # calculate maximum -> skip NA
            stats = dataframe[col_name].max(skipna=True)
        # check if funcion is mean
        elif func_name == "mean":
            # calculate mean -> skip NA
            stats = dataframe[col_name].mean(skipna=True)
        # check if funcion is mean
        elif func_name == "median":
            # calculate mean -> skip NA
            stats = dataframe[col_name].median(skipna=True)
        # check if funcion is number of unique values
        elif func_name == "nunique":
            # calculate nunique -> skip NA
            stats = dataframe[col_name].nunique(dropna=True)
        # check if funcion is NA_total
        elif func_name == "NA_total":
            # calculate total number of NAs
            stats = dataframe[col_name].isna().sum(axis=0)
        # check if funcion is NA_percent
        elif func_name == "NA_percent":
            # calculate proportion of NAs on the given column -> skip NA
            stats = np.ceil((dataframe[col_name].isna().mean(axis=0) * 100))

        return stats


    def validate_col_dtypes_(self) -> None:
        """
        Given the col_dtypes input, validate dataframe
        """

        # check if col_dtypes attribute is not None
        if self.col_dtypes is not None:

            # import required libraries
            from pandas.api.types import is_integer_dtype, is_object_dtype, is_float_dtype, is_datetime64_dtype

            # dtype checkers
            col_dtypes_checker = {
                "object": is_object_dtype,
                "integer": is_integer_dtype,
                "float": is_float_dtype,
                "datetime64[ns]": is_datetime64_dtype
            }

            # iterate over col_dtypes input
            for dtype, cols in self.col_dtypes.items():
                # iterate over cols for a given dtype
                for col in cols:
                    # check if col has the required dtype
                    dtype_val_flag = col_dtypes_checker[dtype](self.dataframe.dtypes[col])

                    # add results to records dict
                    self.records["timestamp"].append(self.val_timestamp)
                    self.records["grouby column"].append("")
                    self.records["column"].append(col)
                    self.records["pandas query"].append("")
                    self.records["condition"].append(f"dtype = {dtype}")
                    self.records["result check"].append(f"{self.emoji_dict[dtype_val_flag]}")
                    self.records["result stats"].append(f"dtype = {self.dataframe.dtypes[col]}")
                    self.records["notes"].append(f"")
                    self.records["duplicated rows checker column"].append("")
                    self.records["percentage (%) of duplicated rows"].append("")
                    self.records["number of duplicated rows"].append("")
                    self.records["duplicating columns"].append("")


    def validate_column_functions_(self) -> None:
        """
        Given the col_funcions_checker input, validate dataframe
        """

        # check if col_funcions_checker attribute is not None
        if self.col_funcions_checker is not None:

            # iterate over validation dict keys and values
            for col_name, funcs_list in self.col_funcions_checker.items():

                # iterate over funcs_dict -> function name and its values to test
                for func_tuple in funcs_list:
                    # get the name of the function
                    func_name = func_tuple[0]

                    # calculate statistics
                    stats = self.calculate_stats_(self.dataframe, func_name, col_name)

                    # add results to records dict
                    self.records["timestamp"].append(self.val_timestamp)
                    self.records["column"].append(col_name)
                    self.records["grouby column"].append("")
                    self.records["pandas query"].append("")
                    if isinstance(stats, (int, float)):
                        self.records["result stats"].append(
                            f"{func_name} = {stats:,.2f}"
                        )
                    else:
                        self.records["result stats"].append(f"{func_name} = {stats}")
                    self.records["duplicated rows checker column"].append("")
                    self.records["percentage (%) of duplicated rows"].append("")
                    self.records["number of duplicated rows"].append("")
                    self.records["duplicating columns"].append("")

                    # check if there are more than one testing value
                    if isinstance(func_tuple[1], list):
                        # check if the length of the tuple is 2
                        if len(func_tuple[1]) != 2:
                            raise ValueError(
                                "Function values to test must be a list with 2 values: [lower threshold, upper threshold]"
                            )

                        # check if the first value in the interval is less than the second one
                        if func_tuple[1][0] >= func_tuple[1][1]:
                            # raise exception
                            raise ValueError(
                                f"The first value to check of function {func_name.upper()} for the column {col_name.upper()} must be smaller than the second value to check!"
                            )

                        # check condition for the given function within the required interval
                        cond = func_tuple[1][0] <= stats <= func_tuple[1][1]

                        # add results to records dict
                        self.records["result check"].append(f"{self.emoji_dict[cond]}")
                        self.records["condition"].append(
                            f"{func_tuple[1][0]} <= {func_name} <= {func_tuple[1][1]}"
                        )

                    # testing value is not an interval
                    else:
                        # check if condition is min
                        if func_tuple[0] == "min":
                            # check condition for the given function on the required value
                            cond = stats >= func_tuple[1]

                        # check if condition is max
                        elif func_tuple[0] == "max":
                            # check condition for the given function on the required value
                            cond = stats <= func_tuple[1]

                        # condition is not min or max
                        else:
                            # check condition for the given function on the required value
                            cond = func_tuple[1] == stats

                        # add results to records dict
                        self.records["result check"].append(f"{self.emoji_dict[cond]}")
                        self.records["condition"].append(
                            f"{func_name} = {func_tuple[1]}"
                        )

                    # check if function name requires a note
                    if func_name in ("min", "max", "mean", "median", "nunique"):
                        # append empty note to record
                        self.records["notes"].append(
                            f"NA values were not considered on {func_name.upper()} calculation"
                        )

                    # function doesn't require a note
                    else:
                        # append empty note to record
                        self.records["notes"].append("")

    def validate_agg_functions_(self) -> None:
        """
        Given the col_aggregations_checker input, validate dataframe
        """

        # check if col_funcions_checker attribute is not None
        if self.col_aggregations_checker is not None:

            # iterate over validation dict keys and values
            for groupby_col, funcs_list in self.col_aggregations_checker.items():

                # iterate over funcs_dict -> function name and its values to test
                for func_tuple in funcs_list:
                    # get column name, agg function, checker function and condition
                    col_name = func_tuple[0]
                    agg_func = func_tuple[1]
                    checker_func = func_tuple[2]
                    condition = func_tuple[3]

                    # calculate the agg_func of column col_name per groupby_col
                    df_groupby = self.dataframe.groupby(groupby_col).agg(
                        {col_name: agg_func}
                    )

                    # calculate statistics
                    stats = self.calculate_stats_(df_groupby, checker_func, col_name)

                    # add results to records dict
                    self.records["timestamp"].append(self.val_timestamp)
                    self.records["grouby column"].append(groupby_col)
                    self.records["column"].append(f"aggregated on {col_name}")
                    self.records["pandas query"].append("")
                    if isinstance(stats, (int, float)):
                        self.records["result stats"].append(
                            f"{checker_func} of ({agg_func} of {col_name} PER {groupby_col}) = {stats:,.2f}"
                        )
                    else:
                        self.records["result stats"].append(
                            f"{checker_func} of ({agg_func} of {col_name} PER {groupby_col}) = {stats}"
                        )
                    self.records["duplicated rows checker column"].append("")
                    self.records["percentage (%) of duplicated rows"].append("")
                    self.records["number of duplicated rows"].append("")
                    self.records["duplicating columns"].append("")

                    # check if there are more than one testing value
                    if isinstance(condition, list):
                        # check if the length of the tuple is 2
                        if len(condition) != 2:
                            raise ValueError(
                                "Function values to test must be a list with 2 values: [lower threshold, upper threshold]"
                            )

                        # check if the first value in the interval is less than the second one
                        if condition[0] >= condition[1]:
                            # raise exception
                            raise ValueError(
                                f"The first value to checking condition must be smaller than the second value to check -> {str(condition)}"
                            )

                        # check condition for the given function within the required interval
                        cond = condition[0] <= stats <= condition[1]

                        # add results to records dict
                        self.records["result check"].append(f"{self.emoji_dict[cond]}")
                        self.records["condition"].append(
                            f"{condition[0]} <= {checker_func} of ({agg_func} of {col_name} PER {groupby_col}) <= {condition[1]}"
                        )

                    # testing value is not an interval
                    else:
                        # check condition for the given function on the required value
                        cond = condition == stats

                        # add results to records dict
                        self.records["result check"].append(f"{self.emoji_dict[cond]}")
                        self.records["condition"].append(
                            f"{checker_func} of ({agg_func} of {col_name} PER {groupby_col}) = {condition}"
                        )

                    # check if function name requires a note
                    if checker_func in ("min", "max", "mean", "median", "nunique"):
                        # append empty note to record
                        self.records["notes"].append(
                            f"NA values were not considered on {checker_func.upper()} calculation"
                        )

                    # function doesn't require a note
                    else:
                        # append empty note to record
                        self.records["notes"].append("")

    def validate_pandas_queries_(self) -> None:
        """
        Check if pandas SQL queries are valid
        """

        # check if col_queries attribute is not None
        if self.pandas_queries is not None:

            # iterate over queries
            for query_list in self.pandas_queries:

                # unpack query list
                query, checking_type, checking_values = (*query_list,)

                # get initital dataframe size
                initial_len = self.dataframe.shape[0]

                # run query and get how many rows
                query_result_rows = self.dataframe.query(expr=query).shape[0]

                # add results to records dict
                self.records["timestamp"].append(self.val_timestamp)
                self.records["grouby column"].append("")
                self.records["column"].append("")
                self.records["pandas query"].append(query)
                self.records["notes"].append(
                    "Rows with NaN where not considered. Based on pandas.DataFrame.query()"
                )
                self.records["duplicated rows checker column"].append("")
                self.records["percentage (%) of duplicated rows"].append("")
                self.records["number of duplicated rows"].append("")
                self.records["duplicating columns"].append("")

                # check testing condition
                if checking_type == "absolute":
                    # gerenate resulting statistics
                    self.records["result stats"].append(
                        f"The given query returned {query_result_rows} rows"
                    )

                # check testing condition
                elif checking_type == "percentual":
                    # gerenate resulting statistics
                    self.records["result stats"].append(
                        f"The given query returned {(query_result_rows / initial_len) * 100:.2f}% of rows"
                    )

                # check if there are more than one testing value
                if isinstance(checking_values, list):

                    # check if the first value in the interval is less than the second one
                    if checking_values[0] >= checking_values[1]:
                        # raise exception
                        raise ValueError(
                            f"The first testing value for the query '{query}' must be smaller than the second testing value!"
                        )

                    # check testing condition
                    if checking_type == "absolute":

                        # check condition for the given function within the required interval
                        cond = (
                            checking_values[0]
                            <= query_result_rows
                            <= checking_values[1]
                        )

                    # check testing condition
                    elif checking_type == "percentual":

                        # check condition for the given function within the required interval
                        cond = (
                            checking_values[0]
                            <= ((query_result_rows / initial_len) * 100)
                            <= checking_values[1]
                        )

                    self.records["condition"].append(
                        f"{checking_type.title()} number of resulting rows from query must be between {checking_values[0]} and {checking_values[1]} [limits included]"
                    )

                # testing value is not an interval
                else:
                    # check testing condition
                    if checking_type == "absolute":

                        # check condition for the given function on the required value
                        cond = query_result_rows == checking_values

                    # check testing condition
                    elif checking_type == "percentual":

                        # check condition for the given function on the required value
                        cond = (
                            (query_result_rows / initial_len) * 100
                        ) == checking_values

                    # save condition
                    self.records["condition"].append(
                        f"{checking_type.title()} number of resulting rows from query must be equal to {checking_values}"
                    )

                # add results to records dict
                self.records["result check"].append(f"{self.emoji_dict[cond]}")

    def validate_dataframe_granularity_(self) -> None:
        """
        Check if dataframe granularity has duplicated rows given the dataframe_granularity input
        """

        # check if dataframe_granularity attribute is not None
        if self.dataframe_granularity is not None:

            # check if user whats dataframe granularity validation
            if self.dataframe_granularity is not None:

                # get number of duplicated rows
                duplicated_grain = self.dataframe.duplicated(
                    subset=self.dataframe_granularity, keep=False
                )

                # add results to records dict
                self.records["timestamp"].append(self.val_timestamp)
                self.records["column"].append("")
                self.records["grouby column"].append("")
                self.records["condition"].append("")
                self.records["result stats"].append("")
                self.records["pandas query"].append("")
                self.records["notes"].append("duplicated rows are DOUBLE COUNTED")
                self.records["duplicated rows checker column"].append(
                    f"based on columns: {self.dataframe_granularity}"
                )

                # define variable to indicate if there are duplicated rows
                duplicates = duplicated_grain.sum() > 0

                # add result check emoji
                self.records["result check"].append(self.emoji_dict[~duplicates])

                # add results to records dict
                self.records["percentage (%) of duplicated rows"].append(
                    f"{duplicated_grain.mean()*100:,.2f}%"
                )
                self.records["number of duplicated rows"].append(
                    f"{duplicated_grain.sum():,.0f}"
                )

                # check if there are duplicated rows
                if duplicates:
                    # check columns that are duplicating granularity
                    self.records["duplicating columns"].append(
                        check_duplicating_columns(
                            dataframe=self.dataframe,
                            df_granularity=self.dataframe_granularity,
                            display_results=False,
                        )
                    )

                # there are no duplicated rows
                else:
                    # add results to records dict
                    self.records["duplicating columns"].append("")

            # create a dataframe from records dict
            df_records = pd.DataFrame(self.records)

            # upper case columns
            df_records.columns = [col.upper() for col in df_records.columns]

    def save_validations(self, records_file: str = None) -> None:
        """
        Save validations to a csv or txt file.

        Args
            records_file: a string with the file path to write (or append) validations.
                It must have a .txt or .csv extension.

        Return
            None: a NoneType object
        """
        ####### INPUT VALIDATION #######
        # check records_file type
        validate_input_types({"records_file": records_file}, (str,))

        # check if records_file input is available
        if records_file is not None:

            # extract extension -> filename, file_extension
            _, file_extension = splitext(records_file)

            # file extension is wrong
            validate_input_values(
                {"records_file": file_extension},
                (".txt", ".csv"),
                'The file to save validations must have extension ".txt" or ".csv"',
            )

            # check if user wants to save validation in txt format
            if file_extension == ".txt":
                # open log file with context manager
                with open(records_file, "a") as log_file:
                    # transform pandas dataframe to string -> with header
                    str_df = self.df_records.to_string(
                        index=False, header=True, justify="center"
                    )
                    # write to file
                    log_file.write(str_df)
                    log_file.write("\n")

            # user wants to save validation in csv format
            else:

                # check if file does't exists
                if not exists(records_file):
                    # save validation to a file with header
                    self.df_records.to_csv(
                        records_file, index=False, header=True, mode="w"
                    )

                # file exists
                else:

                    # open file with context manager
                    with open(records_file, "r") as file:
                        # get number of rows in file
                        has_headers = len(file.readlines()) > 0

                    # check if there are headers
                    if has_headers:
                        # check if headers in the csv is the same as in the validation records
                        if tuple(self.df_records.columns.tolist()) == tuple(
                            pd.read_csv(records_file).columns.tolist()
                        ):
                            # save validation to a file withOUT header
                            self.df_records.to_csv(
                                records_file,
                                index=False,
                                header=False,
                                mode="a",
                            )
                        # available headers are different
                        else:
                            # save validation to a file with header
                            self.df_records.to_csv(
                                records_file,
                                index=False,
                                header=True,
                                mode="a",
                            )

                    # there are no headers
                    else:
                        # save validation to a file with header
                        self.df_records.to_csv(
                            records_file, index=False, header=True, mode="a"
                        )

        # some path to save records must be input
        else:
            # raise error
            raise ValueError(
                "records_file param must be input (whether when calling save_validations method or when instanciating DataValidator object)"
            )

    def plot_historical_report(
        self,
        records_file: str = None,
        datetime_format: str = "%Y-%m-%d",
        save_report: bool = False,
    ) -> None:
        """
        Plot the historical overview about validations.

        Args
            records_file: a string with the file path to the validation csv.
            datetime_format: a string with the datetime granularity for plotting purpose.
                Ex.:
                    If you want to plot report day by day -> "%Y-%m-%d"
                    If you want to plot report month by month -> "%Y-%m"
            save_report: a boolean to indicate if reports are to be saved
                on the same folder as the records file.

        Return
            None: a NoneType object
        """

        ################################
        ####### INPUT VALIDATION #######

        # save_report boolean
        validate_input_types({"datetime_format": datetime_format}, (str,))

        # save_report boolean
        validate_input_types({"save_report": save_report}, (bool,))

        # validate input
        if records_file is not None:
            # check records_file type
            validate_input_types({"records_file": records_file}, (str,))

        # check if file does't exists
        if not exists(records_file):
            # raise exception
            raise ValueError("Invalid records_file params: there is no such file!")

        # make sure the csv contains the validations previously saved by exactly this class
        if pd.read_csv(records_file, nrows=0).columns.tolist() != [
            "timestamp",
            "grouby column",
            "column",
            "pandas query",
            "condition",
            "result check",
            "result stats",
            "notes",
            "duplicated rows checker column",
            "percentage (%) of duplicated rows",
            "number of duplicated rows",
            "duplicating columns",
        ]:
            # raise exception
            raise Exception(
                "The records_file doesn't contain a record file as defined by the DataValidator class!"
            )

        ##################################
        ####### REQUIRED LIBRARIES #######

        # import required libraries
        import matplotlib.ticker as mtick
        import matplotlib.pyplot as plt
        import seaborn as sns

        #########################
        ####### CONSTANTS #######

        # define constants
        blue_strong = "#061D74"
        pink_strong = "#F5009C"

        #########################
        ####### LOAD DATA #######

        # load data frame csv
        df_records = pd.read_csv(
            records_file,
            infer_datetime_format=True,
            parse_dates=["timestamp"],
            usecols=[
                "timestamp",
                "grouby column",
                "column",
                "pandas query",
                "result check",
                "duplicated rows checker column",
            ],
            low_memory=False,
        )

        ###########################################################################
        ####### PREPARE DATA FOR SUCCESS PERCENTAGE AND NUMBER OF TESTS PLOT #######

        # create a mapping column to check status
        df_records["result_check_bool"] = df_records["result check"].map(
            {"❌": False, "✅": True}
        )

        # get a column to indicate date only (not date and time)
        df_records["timestamp_date"] = df_records["timestamp"].dt.strftime(
            datetime_format
        )

        # iterate over columns to input on report
        for col in [
            "grouby column",
            "column",
            "pandas query",
            "duplicated rows checker column",
        ]:
            # map columns
            # where column value is not null, replicate result_check_bool
            # else keep as None
            df_records[f'{col.replace(" ", "_")}_bool'] = np.where(
                ~df_records[col].isna(), df_records["result_check_bool"], None
            )

        # group information by timestamp and get number of numbers, number of sucesses and success rate
        df_plot = df_records.groupby("timestamp_date", as_index=False).agg(
            tests_count=("result_check_bool", "count"),
            success_test=("result_check_bool", "sum"),
            success_rate=("result_check_bool", "mean"),
        )

        # set success rate as percentage
        df_plot["success_rate"] = df_plot["success_rate"] * 100

        ################################################################
        ####### PLOT SUCCESS PERCENTAGE AND NUMBER OF TESTS PLOT #######

        # define figure and axis
        fig, ax1 = plt.subplots()
        # plot tests count
        ax1.bar(
            df_plot["timestamp_date"],
            df_plot["tests_count"],
            color=pink_strong,
            edgecolor="black",
        )
        # define title, labels, ticks and grid
        ax1.set_title(
            "Success rate and number of test validations over time\n",
            loc="center",
            fontweight="normal",
        )
        ax1.set_xlabel("Timestamp", fontweight="bold")
        ax1.set_ylabel("Number of tests", color=pink_strong, fontweight="bold")
        ax1.set_ylim(bottom=0, top=df_plot["tests_count"].max() + 5)
        ax1.tick_params(axis="x", labelrotation=90)
        ax1.set_ylim(bottom=0)
        ax1.grid(False)
        # set graphs to share x axis
        ax2 = ax1.twinx()
        # plot success rate
        ax2.plot(
            df_plot["timestamp_date"],
            df_plot["success_rate"],
            color=blue_strong,
            linestyle="-",
            marker="X",
            markeredgecolor="black",
        )
        # define y axis as percentage
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        # define label and grid
        ax2.set_ylabel("Success rate", color=blue_strong, fontweight="bold")
        ax2.set_ylim(bottom=0, top=df_plot["success_rate"].max() + 10)
        ax2.grid(False)

        # check if user want to save report
        if save_report:
            # get datetime validation
            timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            # extract extension -> filename, file_extension
            file_path, _ = splitext(records_file)
            # save figure
            plt.savefig(
                f"{file_path}_report_at_{timestamp}_success_rate_and_test_counts.png",
                facecolor="white",
                bbox_inches="tight",
                dpi=900,
            )

        # display
        plt.show()

        ###########################################################
        ####### PREPARE DATA TOP MOST FAILURE-PRONE COLUMNS #######

        # get only fail tests
        df_fail = df_records[df_records["result check"] == "❌"]

        # get rows where groupby is not null
        df_fail_groupby = df_fail.loc[
            df_fail["grouby_column_bool"].astype(str) == "False", "grouby column"
        ]
        # get rows where columns is not null
        df_fail_col = df_fail.loc[
            df_fail["column_bool"].astype(str) == "False", "column"
        ]
        # get rows where columns is not null
        df_fail_query = df_fail.loc[
            df_fail["pandas_query_bool"].astype(str) == "False", "pandas query"
        ]

        # get rows where granularity is not null
        df_fail_grain = df_fail.loc[
            df_fail["duplicated_rows_checker_column_bool"].astype(str) == "False",
            "duplicated rows checker column",
        ]

        # create a value count dataframe
        df_fail_col_value_counts = pd.DataFrame(df_fail_col.value_counts()).reset_index(
            names="label"
        )
        # refactor columns to make it more intuitive
        df_fail_col_value_counts["label"] = df_fail_col_value_counts["label"].apply(
            lambda x: f"COLUMN {x}"
        )
        df_fail_col_value_counts = df_fail_col_value_counts.rename(
            columns={"column": "count"}
        )

        # create a value count dataframe
        df_fail_query_value_counts = pd.DataFrame(
            df_fail_query.value_counts()
        ).reset_index(names="label")
        # refactor columns to make it more intuitive
        df_fail_query_value_counts["label"] = df_fail_query_value_counts["label"].apply(
            lambda x: f"QUERY {x}"
        )
        df_fail_query_value_counts = df_fail_query_value_counts.rename(
            columns={"pandas query": "count"}
        )

        # create a value count dataframe
        df_fail_groupby_value_counts = pd.DataFrame(
            df_fail_groupby.value_counts()
        ).reset_index(names="label")
        # refactor columns to make it more intuitive
        df_fail_groupby_value_counts["label"] = df_fail_groupby_value_counts[
            "label"
        ].apply(lambda x: f"GROUP BY {x}")
        df_fail_groupby_value_counts = df_fail_groupby_value_counts.rename(
            columns={"grouby column": "count"}
        )

        # create a value count dataframe
        df_fail_grain_value_counts = pd.DataFrame(
            df_fail_grain.value_counts()
        ).reset_index(names="label")
        # refactor columns to make it more intuitive
        df_fail_grain_value_counts["label"] = df_fail_grain_value_counts["label"].apply(
            lambda x: f"GRANULARITY {x}"
        )
        df_fail_grain_value_counts = df_fail_grain_value_counts.rename(
            columns={"duplicated rows checker column": "count"}
        )

        # concatenate dataframes
        df_concat = pd.concat(
            objs=[
                df_fail_col_value_counts,
                df_fail_query_value_counts,
                df_fail_groupby_value_counts,
                df_fail_grain_value_counts,
            ],
            ignore_index=True,
        )
        # sort values by count
        df_concat = df_concat.sort_values("count", ascending=False, ignore_index=True)

        ###################################################
        ####### PLOT TOP MOST FAILURE-PRONE COLUMNS #######

        # plot a bar chart
        ax = sns.barplot(
            data=df_concat.head(10),
            y="label",
            x="count",
            palette="PuBu_r",
            edgecolor="black",
        )
        # define title, labels
        plt.title("Top 10 validations with more failing tests")
        plt.xlabel("Number of failed tests")
        plt.xticks([*range(0, df_concat["count"].max() + 1)])
        plt.xlim(0, df_concat["count"].max() + 0.25)
        # remove y label
        ax.set(ylabel=None)

        # check if user want to save report
        if save_report:
            # save figure
            plt.savefig(
                f"{file_path}_report_at_{timestamp}_top_failing_features.png",
                facecolor="white",
                bbox_inches="tight",
                dpi=900,
            )

        # display figure
        plt.show()


    def pass_validations(self) -> None:
        """
        Check if all validations were successful 

        Return
            passing_flag: a boolean with
                True if all validation was successful;
                False if at least one validation was failed

        """
        # check if any validation was failed
        passing_flag = (self.df_records["result check"] == "❌").sum() == 0

        return passing_flag
