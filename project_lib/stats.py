##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import numpy as np
from typing import Union
from project_lib.initial_config import initial_settings
from project_lib.input_validation import validate_input_types

# set the basic cofiguration for this module
initial_settings()

################################
####### MODULE FUNCTIONS #######
################################


def apply_stats(array: np.ndarray, stats_name: str, ddof: int = 1) -> float:
    """
    Apply some numpy statistics (via the stats_name tag) to the input array,
    taking into account any input param to numpy statistics.
    It doesn't consider any NaN values of calculations

    Args
        array: a numpy array with the statistic to be calculated.
        stats_name: a string with the tag name of numpy statistic to be calculated.
            Must be "mean", "median", "variance", "std" or "proportion".
        ddof: the ddof param of np.nanstd and np.nanvar functions.
            Basically, an integer with “Delta Degrees of Freedom”:
            the divisor used in the calculation is N - ddof,
            where N represents the number of elements.

    Return
        a float with the applied statistic on the given array, taking into account any input param
    """

    # validate inputs
    validate_input_types({"array": array}, (np.ndarray,))
    validate_input_types({"stats_name": stats_name}, (str,))
    validate_input_types({"ddof": ddof}, (int,))
    assert stats_name in (
        "mean",
        "median",
        "variance",
        "std",
        "proportion",
    ), 'stats_name must be "mean", "median", "variance", "std"'

    # define avaible stats for permutation test
    stats = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "variance": np.nanvar,
        "std": np.nanstd,
        "proportion": np.nanmean,
    }

    # check is stats is std of variance
    if stats_name in ("std", "variance"):
        # apply numpy function with required params
        return stats[stats_name](array, ddof=ddof)
    # stats is not std nor variance
    else:
        # apply numpy function
        return stats[stats_name](array)


def two_sample_permutation_test(
    array_one: np.ndarray,
    array_two: np.ndarray,
    stats_name: str,
    iterations: int = 10_000,
    confidence_interval: int = 95,
    display_report: bool = True,
    display_plot: bool = False,
    ddof: int = 1,
) -> tuple:
    """
    Calculate the permutation test for the given arrays.
    The permutation test take into account some numpy statistics (via the stats_name tag)
    to the input arrays, considering ddof param for std and variance.
    It doesn't consider any NaN values of calculations.

    Returned array has length = 'iterations'.
    Null hypothesis: both samples has same statistical measure.

    Args
        array_one: a numpy array with first sequence of values
        array_two: a numpy array with second sequence of values
        stats_name: a string with the tag name of numpy statistic to be calculated.
            Must be "mean", "median", "variance", "std" or "proportion".
        iterations: an integer with the final size of samples to generate
        confidence_interval: a integer (between 0 and 100) with the value of
            the confidence interval of the p-value.
            confidence interval = 1 - level of significance
        display_report: a boolean to indicate if user wants to see results
        display_plot: a boolean to indicate if user wants to see plot
        ddof: the ddof param of np.nanstd and np.nanvar functions.
            Basically, an integer with “Delta Degrees of Freedom”:
            the divisor used in the calculation is N - ddof,
            where N represents the number of elements.

    Return
        dict: a dict with the following key:
            permutation difference array: a numpy array with the values of the difference between
                the mean of samples after shuffling (for every shuffing)
            original difference: a float with the original difference between samples
            p-value: p-value of the statistical test
            confidence interval limits: lower and upper limits of confidence interval
                of the permutation difference array
    """
    ###############################
    ###### IMPORT LIBRARIES #######
    ###############################

    # import required libraries
    import seaborn as sns
    import matplotlib.pyplot as plt

    ################################
    ####### INPUT VALIDATION #######
    ################################

    # input types verification
    validate_input_types({"array_one": array_one}, (np.ndarray,))
    validate_input_types({"array_two": array_two}, (np.ndarray,))
    validate_input_types({"stats_name": stats_name}, (str,))
    validate_input_types({"iterations": iterations}, (int,))
    validate_input_types({"confidence_interval": confidence_interval}, (int,))
    validate_input_types({"ddof": ddof}, (int,))

    # validate inputs for proportion
    if stats_name == "proportion":
        # assert input arrays are of type int
        # and its values are 0 or 1
        assert (
            (array_one.max() == 1)
            and (array_one.min() == 0)
            and (str(array_one.dtype).startswith("int"))
        ), "array_one must be of type int and with only values 0s and 1s"
        assert (
            (array_two.max() == 1)
            and (array_two.min() == 0)
            and (str(array_two.dtype).startswith("int"))
        ), "array_one must be of type int and with only values 0s and 1s"

    ####################################
    ####### CALCULATE STATISTICS #######
    ####################################

    # calculate original difference between sample statistics
    original_diff = apply_stats(array_one, stats_name, ddof=ddof) - apply_stats(
        array_two, stats_name, ddof=ddof
    )

    # concatenate both arrays
    concat_array = np.concatenate((array_one, array_two))

    # create an empty list
    permutation_diff = np.empty(iterations)

    # iterate "iterations" times
    for i in range(iterations):
        # random reordering of entries in an array
        shuffled_array = np.random.permutation(concat_array)

        # slice the shuffled array into two samples
        # with the same size as the original samples
        shuffled_array_one = shuffled_array[: len(array_one)]
        shuffled_array_two = shuffled_array[len(array_one) :]

        # subtract the statistics of each shuffled array
        stats = apply_stats(shuffled_array_one, stats_name, ddof=ddof) - apply_stats(
            shuffled_array_two, stats_name, ddof=ddof
        )

        # append stats to results
        permutation_diff[i] = stats

    ###########################################
    ###### CONFIDENCE INTERVAL & P-VALUE ######
    ###########################################

    # calculate bilateral p-value for the original difference
    p_value = np.mean(abs(permutation_diff) >= abs(original_diff))

    # calculate upper and lower limit of confidence interval
    lower_limit = (100 - confidence_interval) / 2
    upper_limit = 100 - lower_limit

    # calculate the confidence required confidence interval
    c_i = np.percentile(permutation_diff, [lower_limit, upper_limit])

    # check if user want to see report
    if display_report:
        # print p-value information
        print(
            "*" * 49
            + "\n"
            + "\033[1m"
            + f"\n\tBIlateral (TWO-tailed) test:\n"
            + f"\nstatistics = {stats_name}"
            + f"\n\np-value = {p_value:.3f}"
            + f"\nsignificance level = {(100-confidence_interval)/100:.2f}"
            + f"\n\narray one {stats_name} = {apply_stats(array_one, stats_name, ddof=ddof):.2f}"
            + f"\narray two {stats_name} = {apply_stats(array_two, stats_name, ddof=ddof):.2f}"
            + f"\nobserved difference = {-original_diff:.2f}"
            + f"\nconfidence interval limits = [{c_i[0]:.2f}, {c_i[1]:.2f}]"
            + "\033[0m"
            + "\n\n"
            + "*" * 49
        )

    # check if user want to check plot
    if display_plot:
        ########
        # PLOT #
        ########

        # plot histogram of permutation_differences
        sns.histplot(
            permutation_diff,
            stat="probability",
            kde=True,
            color="steelblue",
            label="PDF",
        )
        # set plot details
        plt.title(
            f"""OBSERVED DIFFERENCE vs RESAMPLING DIFFERENCE
statistics = {stats_name}
p-value = {p_value:.3f}
significance level = {(100-confidence_interval)/100:.2f}
observed difference = {original_diff:.2f}
confidence interval limits = [{c_i[0]:.2f}, {c_i[1]:.2f}]
""",
            loc="left",
        )
        plt.ylabel("PDF", loc="top")
        plt.xlabel(
            "DIFFERENCE OF THE CHOSEN STATISTIC FOR THE GIVEN ARRAYS", loc="left"
        )
        # plot limits of confidence interval
        plt.axvline(
            x=c_i[0],
            color="red",
            label=f"Resampling {confidence_interval}% Confidence Interval",
        )
        plt.axvline(x=c_i[1], color="red")
        # plot original difference between samples
        plt.axvline(x=original_diff, color="green", label="Observed difference")
        # show legend
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # display plot
        plt.show()

    return {
        "permutation difference array": permutation_diff,
        f"original {stats_name} difference": float(original_diff),
        "p-value": p_value,
        "confidence interval limits": c_i,
    }


def one_sample_bootstrap_test(
    sample_array: np.ndarray,
    test_value: Union[float, int],
    stats_name: str,
    iterations: int = 10_000,
    ddof: int = 1,
) -> tuple:
    """
    Calculate the bootstrap statistic difference for the given array and
    the given test_value. Returned array has length = 'iterations'.
    The given test_value is the statistic of the some unkown sample and
    the null hypothesis is that the sample_array and the unkown sample
    have the same statistic.

    Args
        sample_array: a numpy array with the values of the sample
        test_value: a float/integer with the value to be tested
        stats_name: a string with the tag name of numpy statistic to be calculated.
            Must be "mean", "median", "variance", "std" or "proportion".
        iterations: an integer with the final size of samples to generate
        ddof: the ddof param of np.nanstd and np.nanvar functions.
            Basically, an integer with “Delta Degrees of Freedom”:
            the divisor used in the calculation is N - ddof,
            where N represents the number of elements.

    Return
        bootstrap_diff: a numpy array with the values of the difference between
            the mean of the bootstraped array and the test value
        original_diff: a float with the original difference between
            the given sample_array and the given test_value
    """

    # input types verification
    validate_input_types({"sample_array": sample_array}, (np.ndarray,))
    validate_input_types(
        {"test_value": test_value},
        (
            int,
            float,
        ),
    )
    validate_input_types({"stats_name": stats_name}, (str,))
    validate_input_types({"iterations": iterations}, (int,))
    validate_input_types({"ddof": ddof}, (int,))

    # calculate original difference
    original_diff = apply_stats(sample_array, stats_name, ddof=ddof) - test_value

    # take the array and shift its statistic to the test_value
    # now the statistic of sample_array is equal to the statistic of test sample
    shifted_array = (
        sample_array - apply_stats(sample_array, stats_name, ddof=ddof) + test_value
    )

    # instanciate an empty list
    bootstrap_diff = np.empty(iterations)

    # iterate over bootstraping samples
    for i in range(iterations):
        # create an array with bootstraped value of the given array
        bootstraped_array = np.random.choice(
            shifted_array, len(shifted_array), replace=True
        )

        # subtract the statistic of boostraped array from the test_value
        bootstrap_diff[i] = (
            apply_stats(bootstraped_array, stats_name, ddof=ddof) - test_value
        )

    return (bootstrap_diff, float(original_diff))


def num_corr_tests(
    array_one: np.ndarray, array_two: np.ndarray, display_report: bool = True
) -> tuple:
    """
    Calculate pearson correlation test and spearman correlation test to check
    linear and non-linear independence between two arrays of numeric values.

    Args
        array_one: a numpy ndarray with first sequence of values
        array_two: a numpy ndarray with second sequence of values
        display_report: a boolean to indicate whether to display statistical tests results
            or only return them

    Return
        pearson_correlation: a float with the pearson correlation coefficient
        pearson_pvalue: a float with the p-value for the pearson correlation test
        spearman_test: a float with the spearman correlation coefficient
        spearman_test.pvalue: a float with the p-value for the spearman correlation test
        kendall_test: a float with the kendall-tau correlation coefficient
        kendall-test.pvalue: a float with the p-value for the kendall-tau correlation test
    """

    # import required libraries
    from scipy.stats import pearsonr, spearmanr, kendalltau

    # input types verification
    validate_input_types({"array_one": array_one}, (np.ndarray,))
    validate_input_types({"array_two": array_two}, (np.ndarray,))

    ################################
    ####### CORRELATION TEST #######
    ################################

    ####### LINEARITY #######
    # do spearman correlation test
    pearson_test = pearsonr(array_one, array_two)

    ####### NON-LINEARITY #######
    # do spearman correlation test
    spearman_test = spearmanr(array_one, array_two)

    # do kendal correlation test
    kendall_test = kendalltau(array_one, array_two)

    # check if user want the results to be displayed
    if display_report:
        # print reports
        print(
            "\t",
            "\033[91m",
            "PEARSON CORRELATION TEST",
            "\033[0m",
            "\n\n",
            "\033[1m",
            "NULL HYPOTHESIS: two sets of data are uncorrelated.",
            "\033[0m",
            "\n\n",
            "\033[1m",
            "Calculated p-value assumes that each dataset is normally distributed.",
            "\033[0m",
            "\n\n",
            "\033[1m",
            f"Pearson correlation coef: {pearson_test.statistic:.3f}",
            "\033[0m",
            "\n\n",
            "\033[1m",
            f"p-value: {pearson_test.pvalue:.3f}",
            "\033[0m",
            "\n",
        )

        print(
            "\n\t",
            "\033[91m",
            "SPEARMAN CORRELATION TEST",
            "\033[0m",
            "\n\n",
            "\033[1m",
            "NULL HYPOTHESIS: two sets of data are uncorrelated.",
            "\033[0m",
            "\n\n",
            "\033[1m",
            "Nonparametric test. It does NOT assume that both datasets are normally distributed.",
            "\033[0m",
            "\n\n",
            "\033[1m",
            f"Spearman correlation coef: {spearman_test.correlation:.3f}",
            "\033[0m",
            "\n\n",
            "\033[1m",
            f"p-value: {spearman_test.pvalue:.3f}",
            "\033[0m",
            "\n",
        )

        print(
            "\n\t",
            "\033[91m",
            "KENDALL TAU CORRELATION TEST",
            "\033[0m",
            "\n\n",
            "\033[1m",
            "NULL HYPOTHESIS: two sets of data are uncorrelated.",
            "\033[0m",
            "\n\n",
            "\033[1m",
            "Nonparametric test. It does NOT assume that both datasets are normally distributed.",
            "\033[0m",
            "\n\n",
            "\033[1m",
            f"Kendall-tau correlation coef: {kendall_test.correlation:.3f}",
            "\033[0m",
            "\n\n",
            "\033[1m",
            f"p-value: {kendall_test.pvalue:.3f}",
            "\033[0m",
            "\n",
        )

    return (
        pearson_test.statistic,
        pearson_test.pvalue,
        spearman_test.correlation,
        spearman_test.pvalue,
        kendall_test.correlation,
        kendall_test.pvalue,
    )


def plot_bootstrapping(
    sample_array: np.ndarray,
    stats_name: str,
    iterations: int = 10_000,
    ddof: int = 1,
    confidence_interval: int = 95,
    round_stats: int = 2,
) -> float:
    """
    Calculate the bootstrap statistic for the given array and plot.

    Args
        sample_array: a numpy array with the values of the sample
        stats_name: a string with the tag name of numpy statistic to be calculated.
            Must be "mean", "median", "variance", "std" or "proportion".
        iterations: an integer with the final size of samples to generate
        ddof: the ddof param of np.nanstd and np.nanvar functions.
            Basically, an integer with “Delta Degrees of Freedom”:
            the divisor used in the calculation is N - ddof,
            where N represents the number of elements.
        confidence_interval: a integer (between 0 and 100) with the value of
            the confidence interval of the array_diff measures to plot on chart
        round_stats: an integer with the number of decimel to round the confidence interval limits

    Return
        boots_stats: a float with the statistic for the bootstrapping distribution
    """
    ####################
    # IMPORT LIBRARIES #
    ####################

    # import required libraries
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    ###################
    # VALIDATE INPUTS #
    ###################

    # input types verification
    validate_input_types({"sample_array": sample_array}, (np.ndarray,))
    validate_input_types({"stats_name": stats_name}, (str,))
    validate_input_types({"iterations": iterations}, (int,))
    validate_input_types({"ddof": ddof}, (int,))
    validate_input_types({"confidence_interval": confidence_interval}, (int,))
    validate_input_types({"round_stats": round_stats}, (int,))

    #################
    # BOOTSTRAPPING #
    #################

    # instanciate an empty list
    boots_array = np.empty(iterations)

    # iterate over bootstraping samples
    for i in range(iterations):
        # create an array with bootstraped value of the given array
        bootstraped_array = np.random.choice(
            sample_array, len(sample_array), replace=True
        )

        # subtract the statistic of boostraped array from the test_value
        boots_array[i] = apply_stats(bootstraped_array, stats_name, ddof=ddof)

    #######################
    # CONFIDENCE INTERVAL #
    #######################

    # calculate upper and lower limit of confidence interval
    lower_limit = (100 - confidence_interval) / 2
    upper_limit = 100 - lower_limit

    # calculate the confidence required confidence interval
    c_i = np.percentile(boots_array, [lower_limit, upper_limit])

    # calculate the statistic for the bootstrap distribution
    # boots_stats = apply_stats(boots_array, stats_name, ddof=ddof)
    boots_stats = np.percentile(boots_array, 50)

    ########
    # PLOT #
    ########

    # plot histogram of permutation_differences
    sns.histplot(
        boots_array, stat="probability", kde=True, color="steelblue", label="PDF"
    )
    # set plot details
    plt.title(
        f"BOOTSTRAPPING OF THE {stats_name.upper()}\
        \nconfidence interval percentage: {confidence_interval}%\
        \nconfidence interval limits: {list(np.round_(c_i, round_stats))}\
        \n{stats_name.lower()} of the bootstrap distribution: {np.round_(boots_stats, round_stats)}",
        loc="left",
    )
    plt.ylabel("PDF", loc="top")
    plt.xlabel(
        f"VALUES OF {stats_name.upper()} FOR THE BOOTSTRAP DISTRIBUION", loc="left"
    )
    # plot limits of confidence interval
    plt.axvline(
        x=c_i[0],
        color="red",
        label=f"{confidence_interval}% confidence interval limits",
    )
    plt.axvline(x=c_i[1], color="red")
    # plot original difference between samples
    plt.axvline(
        x=boots_stats, color="green", label="Median of the bootstrap distribution"
    )
    # show legend
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # display
    plt.show()

    return boots_stats


def bootstrapping_confidence_interval(
    sample_array: np.ndarray,
    stats_name: str,
    percentiles: list,
    iterations: int = 10_000,
    ddof: int = 1,
) -> list:
    """
    Calculate the bootstrap percentiles of the bootstrapping sampling
    accoridng to the chosen statistic for the given array.

    Args
        sample_array: a numpy array with the values of the sample
        stats_name: a string with the tag name of numpy statistic to be calculated.
            Must be "mean", "median", "variance", "std" or "proportion".
        percentiles: a list of intergers (between 0 and 100, inclusive) with the percentiles
            the get from bootstrapping
        iterations: an integer with the final size of samples to generate
        ddof: the ddof param of np.nanstd and np.nanvar functions.
            Basically, an integer with “Delta Degrees of Freedom”:
            the divisor used in the calculation is N - ddof,
            where N represents the number of elements.

    Return
        boots_percentiles: a list with the percentiles of the bootstrapping
            in the input order
    """

    ###################
    # VALIDATE INPUTS #
    ###################

    # input types verification
    validate_input_types({"sample_array": sample_array}, (np.ndarray,))
    validate_input_types({"stats_name": stats_name}, (str,))
    validate_input_types({"iterations": iterations}, (int,))
    validate_input_types({"ddof": ddof}, (int,))
    validate_input_types({"percentiles": percentiles}, (list,))
    for percentile in percentiles:
        assert (
            (isinstance(percentile, int)) & (percentile >= 0) & (percentile <= 100)
        ), "Percentiles must be a list of intergers (between 0 and 100, inclusive)"

    #################
    # BOOTSTRAPPING #
    #################

    # instanciate an empty list
    boots_array = np.empty(iterations)

    # iterate over bootstraping samples
    for i in range(iterations):
        # create an array with bootstraped value of the given array
        bootstraped_array = np.random.choice(
            sample_array, len(sample_array), replace=True
        )

        # subtract the statistic of boostraped array from the test_value
        boots_array[i] = apply_stats(bootstraped_array, stats_name, ddof=ddof)

    #######################
    # CONFIDENCE INTERVAL #
    #######################

    # calculate the confidence required confidence interval
    boots_percentiles = np.percentile(boots_array, q=percentiles).tolist()

    return boots_percentiles
