# ##############################
# ####### INITIAL CONFIG #######
# ##############################

# # import required project modules
# import pandas as pd
# from project_lib.input_validation import validate_input_types

# ################################
# ####### MODULE FUNCTIONS #######
# ################################


# # def downcast_dataframe(
# #     dataframe: pd.DataFrame, copy: bool = True, verbose: bool = True
# # ) -> pd.DataFrame:
# #     """
# #     Try to downcast numeric columns of the dataframe so as to use less memory.

# #     Args
# #         dataframe: a pd.DataFrame object.
# #         copy: a boolean to indicate if user wants to change the original dataframe
# #             or return a copy of it.
# #         verbose: a boolean to check if user wants to see the downcast report.

# #     Return
# #         dataframe: a pd.DataFrame object with downcasted columns if it was possible;
# #             otherwise, it will just return the original dataframe.
# #             Depending on the copy param, it will return the original dataframe
# #             or a copy of it.

# #     NOTE:
# #         The smallest int dtype after downcast is np.int8
# #         The smallest float after downcast is np.float32
# #     """

# #     # import required libraries
# #     import pandas as pd

# #     # input validation
# #     validate_inputs({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
# #     validate_inputs({"copy": copy}, (bool,))
# #     validate_inputs({"verbose": verbose}, (bool,))

# #     # input types verification
# #     assert isinstance(
# #         dataframe, pd.core.frame.DataFrame
# #     ), "df must be a pd.core.frame.DataFrame object!"
# #     assert isinstance(verbose, bool), "verbose must be a boolean object!"

# #     # check if user wants a copy
# #     if copy:
# #         # make a copy of the input dataframe
# #         df = dataframe.copy()

# #     # user doesn't want a copy
# #     else:
# #         # reassign variable
# #         df = dataframe

# #     # get total dataframe input size in bytes
# #     # the size will include the index size
# #     input_size = df.memory_usage(index=True, deep=True).sum()

# #     # iterate over numeric columns types
# #     for type in ["float", "integer"]:

# #         # get the column names whose type is the given type iteration
# #         list_cols = list(df.select_dtypes(include=type))

# #         # iterate over the columns with the selected types
# #         for col in list_cols:

# #             # downcast the given column to the smallest numerical dtype possible
# #             df[col] = pd.to_numeric(df[col], downcast=type)

# #     # check if user wants a quick report of the results
# #     if verbose:

# #         # get total dataframe output size in bytes
# #         # the size will include the index size
# #         output_size = df.memory_usage(index=True, deep=True).sum()
# #         # get the percentage size that was reduced
# #         ratio = (1 - (output_size / input_size)) * 100

# #         # print report
# #         print(
# #             f"Dataframe size was reduced {ratio:.2f}% of its original size.",
# #             f"\nApproximate initial dataframe size: {input_size / 1000000:,.2f} MB",
# #             f"\nApproximate final dataframe size: {output_size / 1000000:,.2f} MB",
# #         )

# #     return df


# def compress(df, **kwargs):
#     """
#     Reduces size of dataframe by downcasting numerical columns
#     """
#     input_size = df.memory_usage(index=True).sum()/ 1024
#     print("new dataframe size: ", round(input_size,2), 'kB')

#     in_size = df.memory_usage(index=True).sum()
#     for type in ["float", "integer"]:
#         l_cols = list(df.select_dtypes(include=type))
#         for col in l_cols:
#             df[col] = pd.to_numeric(df[col], downcast=type)
#     out_size = df.memory_usage(index=True).sum()
#     ratio = (1 - round(out_size / in_size, 2)) * 100

#     print("optimized size by {} %".format(round(ratio,2)))
#     print("new dataframe size: ", round(out_size / 1024,2), " kB")

#     return df
