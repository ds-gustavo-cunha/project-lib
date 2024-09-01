##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import pandas as pd
from project_lib.initial_config import initial_settings
from project_lib.input_validation import validate_input_types, validate_dataframe_cols

# set the basic cofiguration for this module
initial_settings()

################################
####### MODULE FUNCTIONS #######
################################


def check_for_bias(
    dataframe: pd.DataFrame, 
    treatment: str,
    showfliers: bool = False,
    figsize: tuple = None,
    num_cols: int = 3,
    saving_path: str = None
) -> None:
    """
    Iterate over combinations of outcome variable and dataframe columns
    and plot over these combinations to check for bias before AB testing

    Args
        dataframe: a pandas DataFrame with the data to check for bias
        treatment: a str with the treatment variable column name
        showfliers: a boolean to indicate whether to show (or not) outliers on boxplot
        figsize: a tuple with the figsize to plot
        num_cols: an int with the number of columns to plot variables
        saving_path: a string with the path to save figure
    """
    # input verification
    validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
    validate_input_types({"outcome_variable": treatment}, (str,))
    validate_input_types({"showfliers": showfliers}, (bool,))
    validate_dataframe_cols(dataframe, (treatment,))
    if figsize is not None:
        validate_input_types({"figsize": figsize}, (tuple,))
    validate_input_types({"num_cols": num_cols}, (int,))
    if saving_path is not None:
        validate_input_types({"saving_path": saving_path}, (str,))

    # import required libraries
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns
    from scipy.stats import linregress

    # define plot style
    plt.style.use("fivethirtyeight")

    # define columns according to dtypes
    numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
    categ_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
    date_cols = dataframe.select_dtypes(include=['datetime']).columns.tolist()

    # get columns that are not outcome variable
    non_treatment = numeric_cols + categ_cols
    non_treatment.remove(treatment)
    non_treatment

    # define number of rows
    n_rows = dataframe.shape[1] // num_cols + 1

    # check if user input figsize
    if figsize is None:
        # assign th default figsize
        figsize = (num_cols * 6, n_rows * 4.5)

    # create a figure object
    fig = plt.figure(figsize=figsize, tight_layout=True)

    # create grid for plotting
    specs = gridspec.GridSpec(ncols=num_cols, nrows=n_rows, figure=fig)

    # get number of plots
    num_plots = len(non_treatment)
        
    # define orientation of boxplot
    orient = {True:"v", False:"h"}
        
    #  iterate over combination of outcome and each other variable
    for index, outcome, non_outcome in (
        zip(range(0, num_plots), [treatment]*num_plots, non_treatment)
    ):
        # create a subplot to plot the given feature
        ax = fig.add_subplot(specs[index // num_cols, index % num_cols])

        # check if both plotting variables are numeric ---> regression plot
        if (outcome in numeric_cols) and (non_outcome in numeric_cols): 
            # plot regression over scatter plot
            rp = sns.regplot(
                data=dataframe, x=non_outcome, y=outcome, 
                scatter=True, fit_reg=True, ci=95, n_boot=1000,
                line_kws={"color": "red", "linewidth":1, "linestyle":"--"},
                ax=ax
            )
            #calculate slope and intercept of regression equation
            slope, intercept, rvalue, pvalue, sterr = linregress(
                x=rp.get_lines()[0].get_xdata(),
                y=rp.get_lines()[0].get_ydata(),
                alternative="two-sided"
                )

            # define title    
            ax.set_title(f"Treatment: {outcome}\nVariable: {non_outcome}\nReg slope [p-value]: {slope:.2f} [{pvalue:.3f}]")

        # check if both plotting variables are categorical ---> bar plot
        elif (outcome in categ_cols) and (non_outcome in categ_cols):    
            # groupby dataframe by variables to plot and get size
            df_plot = dataframe.groupby(by=[non_outcome, outcome], as_index=False).size() # not count NaNs
            # plot a bar chart
            sns.barplot(
                data=df_plot, x=non_outcome, y="size", 
                hue=outcome, edgecolor=".0",
                ax=ax
            )
            # define title    
            ax.set_title(f"Treatment: {outcome}\nVariable: {non_outcome}")

        # one variabel is numeric while the other is categoric ---> box plot
        else:
            # plot a box plot
            sns.boxplot(
                data=dataframe, x=non_outcome, y=outcome, 
                orient=orient[outcome in numeric_cols],
                meanline=True, showmeans=True, meanprops={"color": "black", "marker": "*"},
                showfliers=showfliers,
                ax=ax
            )
            # define title    
            ax.set_title(f"Treatment: {outcome}\nVariable: {non_outcome}\nShow outliers: {showfliers}")

        # check if x axis has many categories
        if non_outcome in categ_cols:
            # rotate x labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # check if user want to save heatmap
    if saving_path is not None:
        # save figure to inspect outside notebook
        plt.savefig(
            saving_path,
            dpi=200,
            transparent=False,
            bbox_inches="tight",
            facecolor="white",
        )

    # display plot
    plt.show(); 