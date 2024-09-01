##############################
####### INITIAL CONFIG #######
##############################

# import required project modules
from project_lib.input_validation import validate_input_types


################################
####### MODULE FUNCTIONS #######
################################


def initial_settings(storytelling: bool = True) -> None:
    """
    Set initial settings for dataframes and plotting diplays

    Args
        storytelling: a boolean to indicate if plot are being prepared for storytelling (or not).
            This change the theme to colorblind and remove grid
    """
    ####################
    # input validation #
    ####################
    validate_input_types({"storytelling": storytelling}, (bool,))

    ####################
    # import libraries #
    ####################
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from IPython.display import display, HTML

    #####################
    # pandas dataframes #
    #####################

    # set cientific notation for pandas
    pd.set_option(
        "display.float_format", "{:,.3f}".format
    )  # used in some places like SeriesFormatter
    pd.set_option(
        "display.precision", 3
    )  # for regular formatting as well as scientific notation
    pd.set_option(
        "styler.format.precision", 3
    )  # The precision for floats and complex numbers
    # don't truncate columns
    pd.set_option("display.max_colwidth", 100)  # None for unlimited
    # display all columns
    pd.set_option("display.max_columns", None)
    # display up to 100 rows
    pd.set_option("display.max_rows", 100)
    # display dimensions
    pd.set_option("display.show_dimensions", True)
    # define decimals and thousand separation
    pd.set_option("styler.format.decimal", ",")
    pd.set_option("styler.format.thousands", ".")

    ####################
    # matplotlib plots #
    ####################

    # set default plt figure size
    plt.rcParams["figure.figsize"] = [10, 5]
    # figure suptitle
    plt.rcParams["figure.titlesize"] = "large"
    plt.rcParams["figure.titleweight"] = "bold"
    # set default plt font size
    plt.rcParams["font.size"] = 24
    # font weight
    # plt.rcParams["font.weight"] = "bold"
    # title location
    plt.rcParams["axes.titlelocation"] = "left"
    # title size
    plt.rcParams["axes.titlesize"] = "large"
    # title wight
    plt.rcParams["axes.titleweight"] = "bold"
    # plt.rcParams["axes.labelweight"] = "bold"
    # spines
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    # axis labels
    plt.rcParams["xaxis.labellocation"] = "left"
    plt.rcParams["yaxis.labellocation"] = "top"
    # figure layout
    plt.rcParams["figure.autolayout"] = False
    # save figures
    plt.rcParams["savefig.dpi"] = 900
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.format"] = "png"

    # set figures to seaborn style
    sns.set()

    #####################
    # jupyter notebooks #
    #####################

    # set cell size to be expanded
    display(HTML("<style>.container { width:100% !important; }</style>"))

    #######################
    # storytelling styles #
    #######################

    # check if plots are to be prepared for storytelling
    if storytelling:
        # set style
        plt.style.use("tableau-colorblind10")
        # plt.style.use("fivethirtyeight")

        # # set the face color globally for all axes objects
        plt.rcParams["axes.facecolor"] = "white"

    # plots are NOT to be prepared for storytelling
    else:
        # set ggplot pallete
        # # plt.style.use("ggplot")
        plt.style.use("fivethirtyeight")

        # # set the face color globally for all axes objects
        # plt.rcParams["axes.facecolor"] = "lightgrey"

    return None
