##############################
####### INITIAL CONFIG #######
##############################

# import required library to configure module
import pandas as pd
from typing import Union
from project_lib.initial_config import initial_settings
from project_lib.input_validation import (
    validate_input_types,
    validate_dataframe_cols,
    validate_input_values,
)

# set the basic cofiguration for this module
initial_settings()


################################
####### MODULE FUNCTIONS #######
################################


class SurvAnalysisFeaturesOverview:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        duration_col: str,
        event_col: str,
        feature_col: list,
        survival_in_time_limit: int,
        signif_level: float = 0.05,
        threshold_type: str = "median",
    ) -> None:
        """
        Perform a survival analysis based on threshold split and on COX modelling.

        Args
            dataframe: a pandas DataFrame with data to be analysed.
            duration_col: a str with the column name for duration information for survival analysis.
            event_col: a str with the column name for event information for survival analysis.
            feature_col: a string with the column name to be analysed.
            survival_in_time_limit: an int with the timestamp to be analysed on
                difference between survival curves at time = survival_in_time_limit.
            signif_level: a float with the significant level for p-values comparison. Default = 0.05.
            threshold_type: a string, "median" or "mean", with the statistics to split data
                and compare survival curves.

        Return
            None: a None type object


        *************************************************

        Methods

            SurvAnalysisFeaturesOverview().plot():
                Plot threshold survival analysis and/or cox survival analysis.



        Atributes

            SurvAnalysisFeaturesOverview().df_survival_cox_summary:
                a pandas Dataframe with statistics of survival analysis based on threshold split.

            SurvAnalysisFeaturesOverview().df_survival_threshold_summary:
                a pandas Dataframe with statistics of survival analysis based on COX modelling.

        """

        # validate input params
        self.validate_instanciation_params_(
            dataframe=dataframe,
            duration_col=duration_col,
            event_col=event_col,
            feature_col=feature_col,
            survival_in_time_limit=survival_in_time_limit,
            signif_level=signif_level,
            threshold_type=threshold_type,
        )

        # create required object attributes that will be used throughout the script
        self.create_attributes_(
            dataframe=dataframe,
            duration_col=duration_col,
            event_col=event_col,
            feature_col=feature_col,
            survival_in_time_limit=survival_in_time_limit,
            signif_level=signif_level,
            threshold_type=threshold_type,
        )

        # get survival threshold analysis for the given column
        self.surv_analysis_column_threshold_overview_()

        # get survival cox analysis for the given column
        self.survival_analysis_cox_variable_overview_()

        # sort dataframes
        self.df_survival_threshold_summary = (
            self.df_survival_threshold_summary.sort_values(
                by=[
                    "log_rank_signif_p_value",
                    "rmst_above_over_below_%",
                    "passing_cox_assumptions_signif_p_value",
                    "cox_coef_signif_p_value",
                    "cox_exp(coef)",
                    "survival_in_time_signif_p_value",
                ],
                ascending=[False, False, False, False, True, False],
            )
        )
        self.df_survival_cox_summary = self.df_survival_cox_summary.sort_values(
            by=[
                "passing_cox_signif_p_value",
                "cox_coef_signif_p_value",
                "restricted_mean_surv_time",
                "cox_concordance_index",
                "cox_AIC_partial",
                "cox_log_likelihood",
            ],
            ascending=[False, False, False, False, False, False],
        )

    def plot(
        self,
        plot_survival_curves: dict = {"threshold_analysis": True, "cox_analysis": True},
        figsize: tuple = None,
        figure_suptitle: str = None,
        plot_stats: bool = True,
        y_axis_label: str = "Survival curve",
        x_axis_label: str = "Timeline",
        above_label: str = "Above threshold",
        below_label: str = "Below threshold",
        baseline_labels: str = "baseline survival",
        threshold_title: str = None,
        cox_title: str = None,
        plot_threshold_baseline: bool = True,
        plot_cox_baseline: bool = True,
        threshold_legend: str = None,
        cox_partial_values: list = None,
        cox_col_name_legend: str = None,
        round_cox_quantiles: int = 2,
        saving_path: str = None,
    ) -> None:
        """
        Plot threshold analysis and/or cox analysis.


        Args
            plot_survival_curves: a dictionary with mapping to what plots to display.
                Ex.: {"threshold_analysis": True, "cox_analysis": False}
            figsize: a tuple with the whole figure size (width, height).
            figure_suptitle: a string with the figure superior title.
            plot_stats: a boolean to indicate if survival analysis statistics must be shown on plots.
            y_axis_label: a string with the y axis label.
            x_axis_label: a string with the x axis label.
            threshold_legend: a string with the title of the legend of threshold analysis.
            above_label: a string with the label for the "Above threshold" survival line on plot legend.
            below_label: a string with the label for the "Below threshold" survival line on plot legend.
            baseline_labels: a string with the label for "baseline survival" survival line on plot legend.
            plot_threshold_baseline: a boolean to indicate if baseline must be shown for threshold analysis.
            plot_cox_baseline: a boolean to indicate if baseline must be shown for cox analysis.
            cox_partial_values: a list with values to display on cox analysis.
            cox_col_name_legend: a string with names to be display on legend of cox analysis.
            round_cox_quantiles: an integer with number of decimals for cox analysis values.
            saving_path: a string with the path to save figure.


        Return
            None: a None type object

        """

        # validate input params
        self.validate_plot_params_(
            plot_survival_curves=plot_survival_curves,
            figsize=figsize,
            figure_suptitle=figure_suptitle,
            plot_stats=plot_stats,
            y_axis_label=y_axis_label,
            x_axis_label=x_axis_label,
            above_label=above_label,
            below_label=below_label,
            baseline_labels=baseline_labels,
            threshold_title=threshold_title,
            cox_title=cox_title,
            plot_threshold_baseline=plot_threshold_baseline,
            plot_cox_baseline=plot_cox_baseline,
            threshold_legend=threshold_legend,
            cox_partial_values=cox_partial_values,
            cox_col_name_legend=cox_col_name_legend,
            round_cox_quantiles=round_cox_quantiles,
            saving_path=saving_path,
        )

        # create required object attributes that will be used for plotting purpose
        self.create_plot_attributes_(
            plot_survival_curves=plot_survival_curves,
            figsize=figsize,
            figure_suptitle=figure_suptitle,
            plot_stats=plot_stats,
            y_axis_label=y_axis_label,
            x_axis_label=x_axis_label,
            above_label=above_label,
            below_label=below_label,
            baseline_labels=baseline_labels,
            threshold_title=threshold_title,
            cox_title=cox_title,
            plot_threshold_baseline=plot_threshold_baseline,
            plot_cox_baseline=plot_cox_baseline,
            threshold_legend=threshold_legend,
            cox_partial_values=cox_partial_values,
            cox_col_name_legend=cox_col_name_legend,
            round_cox_quantiles=round_cox_quantiles,
            saving_path=saving_path,
        )

        # check if user want to plot survival curves
        if self.num_plot > 0:

            # import required libraries
            import matplotlib.pyplot as plt

            # define figure layout
            fig, axs = plt.subplots(
                nrows=1, ncols=self.num_plot, figsize=self.figsize, tight_layout=True
            )

            # check if user defined a specific column name
            if self.figure_suptitle is None:
                # define superior title with column info
                fig.suptitle(self.feature_col)

            # no column name specified
            else:
                # define superior title with column info
                fig.suptitle(self.figure_suptitle)

            # check want plot the user wants
            if (
                self.plot_survival_curves["threshold_analysis"]
                and self.plot_survival_curves["cox_analysis"]
            ):
                # plot threshold analysis
                self.plot_threshold_analysis_(
                    df_column_threshold_survival=self.df_feature_threshold_survival,
                    cph_threshold=self.cph_threshold,
                    km_above=self.km_above_threshold,
                    km_below=self.km_below_threshold,
                    ax=axs[0],
                )
                # plot cox analysis
                self.plot_cox_analysis_(
                    df_column_cox_survival=self.df_cph_feature_cox,
                    cph_cox=self.cph_feature_cox,
                    ax=axs[1],
                )

            # check if user want to plot threshold analysis
            elif self.plot_survival_curves["threshold_analysis"]:
                # plot threshold analysis
                self.plot_threshold_analysis_(
                    df_column_threshold_survival=self.df_feature_threshold_survival,
                    cph_threshold=self.cph_threshold,
                    km_above=self.km_above_threshold,
                    km_below=self.km_below_threshold,
                    ax=axs,
                )

            # check if user wanto to plot cox analysis
            elif self.plot_survival_curves["cox_analysis"]:
                # plot cox analysis
                self.plot_cox_analysis_(
                    df_column_cox_survival=self.df_cph_feature_cox,
                    cph_cox=self.cph_feature_cox,
                    ax=axs,
                )

            # check if user want to save figure
            if self.saving_path is not None:

                # save figure
                plt.savefig(
                    self.saving_path, facecolor="white", bbox_inches="tight", dpi=900
                )

            # display figure
            plt.show()

    def validate_instanciation_params_(
        self,
        dataframe,
        duration_col,
        event_col,
        feature_col,
        survival_in_time_limit,
        signif_level,
        threshold_type,
    ) -> None:
        """
        Validate class input params according to the expected dtypes and values
        """

        # input validation
        validate_input_types({"dataframe": dataframe}, (pd.core.frame.DataFrame,))
        validate_input_types({"duration_col": duration_col}, (str,))
        validate_input_types({"event_col": event_col}, (str,))
        validate_input_types({"feature_col": feature_col}, (str,))
        validate_input_types({"survival_in_time_limit": survival_in_time_limit}, (int,))
        validate_input_types({"signif_level": signif_level}, (float,))
        validate_input_types({"threshold_type": threshold_type}, (str,))
        validate_dataframe_cols(
            dataframe=dataframe, columns=(duration_col, event_col, feature_col)
        )

    def validate_plot_params_(
        self,
        plot_survival_curves,
        figsize,
        figure_suptitle,
        plot_stats,
        y_axis_label,
        x_axis_label,
        above_label,
        below_label,
        baseline_labels,
        threshold_title,
        cox_title,
        plot_threshold_baseline,
        plot_cox_baseline,
        threshold_legend,
        cox_partial_values,
        cox_col_name_legend,
        round_cox_quantiles,
        saving_path,
    ) -> None:
        """
        Validate plot input params according to the expected dtypes and values
        """

        # input validation
        validate_input_types({"y_axis_label": y_axis_label}, (str,))
        validate_input_types({"x_axis_label": x_axis_label}, (str,))
        validate_input_types({"plot_stats": plot_stats}, (bool,))
        validate_input_types({"above_label": above_label}, (str,))
        validate_input_types({"below_label": below_label}, (str,))
        validate_input_types({"baseline_labels": baseline_labels}, (str,))
        validate_input_types(
            {"plot_threshold_baseline": plot_threshold_baseline}, (bool,)
        )
        validate_input_types({"plot_cox_baseline": plot_cox_baseline}, (bool,))
        validate_input_types({"plot_survival_curves": plot_survival_curves}, (dict,))
        validate_input_types({"round_cox_quantiles": round_cox_quantiles}, (int,))
        for k, v in plot_survival_curves.items():
            validate_input_types(
                {"plot_survival_curves_keys": k},
                (str,),
                "plot_survival_curves keys must be strings",
            )
            validate_input_values(
                {"plot_survival_curves_keys": k},
                ("threshold_analysis", "cox_analysis"),
                "plot_survival_curves keys must be: threshold_analysis or cox_analysis",
            )
            validate_input_types(
                {"plot_survival_curves_values": v},
                (bool,),
                "plot_survival_curves values keys must be boolean",
            )
            validate_input_values(
                {"plot_survival_curves_values": v},
                (True, False),
                "plot_survival_curves values must be: True or False",
            )
        if threshold_title is not None:
            validate_input_types({"threshold_title": threshold_title}, (str,))
        if cox_title is not None:
            validate_input_types({"cox_title": cox_title}, (str,))
        if figsize is not None:
            validate_input_types({"figsize": figsize}, (tuple,))
        if figure_suptitle is not None:
            validate_input_types({"figure_suptitle": figure_suptitle}, (str,))
        if saving_path is not None:
            validate_input_types({"saving_path": saving_path}, (str,))
        if threshold_legend is not None:
            validate_input_types({"threshold_legend": threshold_legend}, (str,))
        if cox_partial_values is not None:
            validate_input_types({"cox_partial_values": cox_partial_values}, (list,))
            if len(cox_partial_values) > 6:
                raise ValueError(
                    "cox_partial_values must be a list with not more than 6 items!"
                )
        if cox_col_name_legend is not None:
            validate_input_types({"cox_col_name_legend": cox_col_name_legend}, (str,))

    def create_attributes_(
        self,
        dataframe,
        duration_col,
        event_col,
        feature_col,
        survival_in_time_limit,
        signif_level,
        threshold_type,
    ) -> None:
        """
        Create the object attributes that will be used throughout the object methods.
        """

        # define required atributes based on class input
        self.dataframe = dataframe
        self.duration_col = duration_col
        self.event_col = event_col
        self.feature_col = feature_col
        self.survival_in_time_limit = survival_in_time_limit
        self.signif_level = signif_level
        self.threshold_type = threshold_type

        # define dataframes to store survival statistics
        self.df_survival_threshold_summary = pd.DataFrame(
            columns=[
                "column",
                "above_label",
                "below_label",
                "threshold_label",
                "signif_level",
                "log_rank_p_value",
                "log_rank_signif_p_value",
                "log_rank_signif_p_value_flag",
                "rmst_above_over_below_%",

                "survival_in_time_p_value",
                "survival_in_time_signif_p_value",
                "survival_in_time_signif_p_value_flag",

                "passing_cox_assumptions_p_value",
                "passing_cox_assumptions_signif_p_value",
                "passing_cox_assumptions_signif_p_value_flag",

                "cox_coef_p_value",
                "cox_coef_signif_p_value",
                "cox_coef_signif_p_value_flag",
                "cox_exp(coef)",
            ]
        )
        self.df_survival_threshold_summary = self.df_survival_threshold_summary.astype(
            {
                "passing_cox_assumptions_signif_p_value": bool,
                "cox_coef_signif_p_value": bool,
                "log_rank_signif_p_value": bool,
                "survival_in_time_signif_p_value": bool,
            }
        )
        self.df_survival_cox_summary = pd.DataFrame(
            columns=[
                "column",
                "bool_col",
                "cox_coef_p_value",
                "cox_coef_signif_p_value",
                "cox_coef_signif_p_value_flag",
                "cox_exp(coef)",
                "passing_cox_p_value",
                "passing_cox_signif_p_value",
                "passing_cox_signig_p_value_flag",
                "restricted_mean_surv_time",
                "cox_concordance_index",
                "cox_AIC_partial",
                "cox_log_likelihood",
            ]
        )
        self.df_survival_cox_summary = self.df_survival_cox_summary.astype(
            {
                "bool_col": bool,
                "cox_coef_signif_p_value": bool,
                "passing_cox_signif_p_value": bool,
            }
        )

    def create_plot_attributes_(
        self,
        plot_survival_curves,
        figsize,
        figure_suptitle,
        plot_stats,
        y_axis_label,
        x_axis_label,
        above_label,
        below_label,
        baseline_labels,
        threshold_title,
        cox_title,
        plot_threshold_baseline,
        plot_cox_baseline,
        threshold_legend,
        cox_partial_values,
        cox_col_name_legend,
        round_cox_quantiles,
        saving_path,
    ) -> None:
        """
        Create plot attributes that will be used for plotting purpose
        """

        # define required atributes based on class input
        self.plot_survival_curves = plot_survival_curves
        self.figure_suptitle = figure_suptitle
        self.plot_stats = plot_stats
        self.y_axis_label = y_axis_label
        self.x_axis_label = x_axis_label
        self.above_label = above_label
        self.below_label = below_label
        self.baseline_labels = baseline_labels
        self.threshold_title = threshold_title
        self.cox_title = cox_title
        self.plot_threshold_baseline = plot_threshold_baseline
        self.plot_cox_baseline = plot_cox_baseline
        self.threshold_legend = threshold_legend
        self.cox_partial_values = cox_partial_values
        self.cox_col_name_legend = cox_col_name_legend
        self.round_cox_quantiles = round_cox_quantiles
        self.saving_path = saving_path

        # check number of plots
        self.num_plot = sum([v for v in self.plot_survival_curves.values()])
        # define default figure sizes accorinding to number of plot
        default_figsize = {1: (10, 5), 2: (12, 6)}
        # check if figsize input is None
        if figsize is None:
            # define figsize attribute
            self.figsize = default_figsize[self.num_plot]
        # figsize input is not none
        else:
            # define figsize attribute
            self.figsize = figsize

        # define colors to plot
        self.blue_strong, self.pink_strong, self.yellow_strong = (
            "#061D74",
            "#F5009C",
            "#ffdd19",
        )

    def surv_analysis_column_threshold_overview_(
        self,
    ) -> None:
        """
        Perform an general survival analysis based on thresholds (mean, median or 0.5).

        Args:
            col: a string with a valid column
        """

        # import required libraries
        from lifelines.statistics import logrank_test
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.statistics import (
            survival_difference_at_fixed_point_in_time_test,
            proportional_hazard_test,
        )
        from lifelines.utils import restricted_mean_survival_time

        # define dict with emojis
        emoji_dict = {True: "✅", False: "❌"}

        # define a variable to indicate if columns is boolean
        bool_col = self.dataframe[self.feature_col].dtype == "bool"

        # check if column dtype is boolean
        if bool_col:
            # set threshold to 0.5
            threshold = 0.5
            # define labels to plot
            above_label = "True"
            below_label = "False"
            threshold_label = "boolean"

        # column is not boolean
        else:
            # check if threshold is median
            if self.threshold_type == "median":
                # get the median as threshold
                threshold = self.dataframe[self.feature_col].median()

            # threshold is mean
            elif self.threshold_type == "mean":
                # get the mean as threshold
                threshold = self.dataframe[self.feature_col].mean()

            # define labels to plot
            above_label = "Above threshold"
            below_label = "Below threshold"
            threshold_label = self.threshold_type

        # create two dataframes according to above or below threshold
        df_above_threshold = self.dataframe[
            self.dataframe[self.feature_col] > threshold
        ]
        df_below_threshold = self.dataframe[
            self.dataframe[self.feature_col] <= threshold
        ]

        # define durations and observed arrays for above threhold
        durations_above = df_above_threshold[self.duration_col].to_numpy()
        observed_above = df_above_threshold[self.event_col].to_numpy()

        # define durations and observed arrays for below threshold
        durations_below = df_below_threshold[self.duration_col].to_numpy()
        observed_below = df_below_threshold[self.event_col].to_numpy()

        # calculate log rank test between survival curves
        log_rank = logrank_test(
            durations_A=durations_above,
            event_observed_A=observed_above,
            durations_B=durations_below,
            event_observed_B=observed_below,
        )

        # get log rank p-value
        log_rank_p_value = log_rank.p_value

        # fit Cox model to data
        cph = CoxPHFitter().fit(
            df=self.dataframe[[self.duration_col, self.event_col, self.feature_col]],
            duration_col=self.duration_col,
            event_col=self.event_col,
            show_progress=False,
        )

        # get cox p-value
        cox_coeff_p_value = cph.summary["p"][0]
        # get increase in hazard
        cox_hazard_increase = cph.summary["exp(coef)"][0]

        # get proportion of passed tests regarding cox assumptions
        cox_assumption_p_value = proportional_hazard_test(
            cph,
            self.dataframe[[self.duration_col, self.event_col, self.feature_col]],
            time_transform=["km"],
        ).p_value

        # instanciate Kaplan-Meier model
        km_above = KaplanMeierFitter()
        # fit model to above data
        km_above.fit(durations=durations_above, event_observed=observed_above)
        # calculate restricted mean survival time for above data
        restricted_mean_surv_time_above = restricted_mean_survival_time(
            km_above, t=self.survival_in_time_limit
        )

        # instanciate Kaplan-Meier model
        km_below = KaplanMeierFitter()
        # fit model to below data
        km_below.fit(durations=durations_below, event_observed=observed_below)
        # calculate restricted mean survival time for below data
        restricted_mean_surv_time_below = restricted_mean_survival_time(
            km_below, t=self.survival_in_time_limit
        )

        # get the percent of increase (or decrease) between above and below survival curves
        restrict_above_over_below = (
            restricted_mean_surv_time_above - restricted_mean_surv_time_below
        ) / restricted_mean_surv_time_below

        # calculate the difference in time test between survival curves
        surv_diff_in_time = survival_difference_at_fixed_point_in_time_test(
            self.survival_in_time_limit, km_above, km_below
        )

        # create a pandas dataframe with prepared info
        df_column_survival = pd.DataFrame(
            data={
                "column": self.feature_col,
                "above_label": above_label,
                "below_label": below_label,
                "threshold_label": threshold_label.title(),
                "signif_level": self.signif_level,
                "passing_cox_assumptions_p_value": cox_assumption_p_value,
                "passing_cox_assumptions_signif_p_value": cox_assumption_p_value
                < self.signif_level,
                "passing_cox_assumptions_signif_p_value_flag": emoji_dict[
                    bool(cox_assumption_p_value < self.signif_level)
                ],
                "cox_coef_p_value": cox_coeff_p_value,
                "cox_coef_signif_p_value": cox_coeff_p_value < self.signif_level,
                "cox_coef_signif_p_value_flag": emoji_dict[
                    cox_coeff_p_value < self.signif_level
                ],
                "cox_exp(coef)": cox_hazard_increase,
                "log_rank_p_value": log_rank_p_value,
                "log_rank_signif_p_value": log_rank_p_value < self.signif_level,
                "log_rank_signif_p_value_flag": emoji_dict[
                    log_rank_p_value < self.signif_level
                ],
                "survival_in_time_p_value": surv_diff_in_time.p_value,
                "survival_in_time_signif_p_value": surv_diff_in_time.p_value
                < self.signif_level,
                "survival_in_time_signif_p_value_flag": emoji_dict[
                    surv_diff_in_time.p_value < self.signif_level
                ],
                "rmst_above_over_below_%": restrict_above_over_below * 100,
            },
            index=[0],
        )

        # concatenate dataframe to survival threshold summary
        self.df_survival_threshold_summary = pd.concat(
            objs=[self.df_survival_threshold_summary, df_column_survival],
            ignore_index=True,
        )

        # create respective attributes
        self.df_feature_threshold_survival = df_column_survival
        self.cph_threshold = cph
        self.km_above_threshold = km_above
        self.km_below_threshold = km_below

    def survival_analysis_cox_variable_overview_(
        self,
    ) -> None:
        """
        Perform an general survival analysis based on cox proportional-hazard model.
        """

        # import required libraries
        from lifelines import KaplanMeierFitter, CoxPHFitter
        from lifelines.utils import restricted_mean_survival_time
        from lifelines.statistics import proportional_hazard_test

        # define dict with emojis
        emoji_dict = {True: "✅", False: "❌"}

        # define duration limits
        min_duration = self.dataframe[self.duration_col].min()
        max_duration = self.dataframe[self.duration_col].max()

        # define a variable to indicate if columns is boolean
        bool_col = self.dataframe[self.feature_col].dtype == "bool"

        # check if variable is boolean
        if bool_col:
            # define formula in case of boolean variable
            formula = f"C({self.feature_col})"

        # variable is not boolean
        else:
            # define formula
            formula = f"{self.feature_col}"

        # fit Cox model to data
        cph = CoxPHFitter().fit(
            df=self.dataframe[[self.duration_col, self.event_col, self.feature_col]],
            duration_col=self.duration_col,
            event_col=self.event_col,
            show_progress=False,
            formula=formula,
        )

        # get statistics from model training
        cox_coef_p_value = cph.summary["p"][0]
        cox_hazard_increase = cph.summary["exp(coef)"][0]
        cox_log_likelihood = cph.log_likelihood_
        cox_AIC_partial = cph.AIC_partial_
        cox_concordance_index = cph.concordance_index_

        # check cph assumtions
        passing_cox_assumption = proportional_hazard_test(
            cph,
            self.dataframe[[self.duration_col, self.event_col, self.feature_col]],
            time_transform=["km"],
        ).p_value

        # instanciate Kaplan-Meier model
        km = KaplanMeierFitter()
        # fit model to above data
        km.fit(
            durations=self.dataframe[self.duration_col],
            event_observed=self.dataframe[self.event_col],
        )

        # calculate restricted meansurvival time
        rmst = restricted_mean_survival_time(km, t=max_duration) / (
            max_duration - min_duration
        )

        # create a pandas dataframe with prepared info
        df_column_survival = pd.DataFrame(
            data={
                "column": self.feature_col,
                "bool_col": bool_col,
                "cox_coef_p_value": cox_coef_p_value,
                "cox_coef_signif_p_value": cox_coef_p_value < self.signif_level,
                "cox_coef_signif_p_value_flag": emoji_dict[
                    cox_coef_p_value < self.signif_level
                ],
                "cox_exp(coef)": cox_hazard_increase,
                "passing_cox_p_value": passing_cox_assumption[0],
                "passing_cox_signif_p_value": passing_cox_assumption[0]
                < self.signif_level,
                "passing_cox_signig_p_value_flag": emoji_dict[
                    passing_cox_assumption[0] < self.signif_level
                ],
                "cox_concordance_index": cox_concordance_index,
                "cox_log_likelihood": cox_log_likelihood,
                "cox_AIC_partial": cox_AIC_partial,
                "restricted_mean_surv_time": rmst,
            },
            index=[0],
        )

        # add values to summary dataframe
        self.df_survival_cox_summary = pd.concat(
            objs=[self.df_survival_cox_summary, df_column_survival], ignore_index=True
        )

        # create respective attributes
        self.df_cph_feature_cox = df_column_survival
        self.cph_feature_cox = cph

    def plot_threshold_analysis_(
        self, df_column_threshold_survival, cph_threshold, km_above, km_below, ax
    ) -> None:
        """
        Plot threshold analysis
        """

        # import required libraries
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # check if user defined a threshold_title
        if self.threshold_title is not None:
            # plot title
            ax.set_title(f"{self.threshold_title}")

        # check if user wants title with statistics
        elif self.plot_stats:
            # create custom message to display on plot
            msg = f"""Column: {df_column_threshold_survival.loc[0, "column"]}
Cox assumptions: {df_column_threshold_survival.loc[0, "passing_cox_assumptions_signif_p_value"]} [p-value={df_column_threshold_survival.loc[0, "passing_cox_assumptions_p_value"]:.2f}]
Cox coef signif p-value: {df_column_threshold_survival.loc[0, "cox_coef_signif_p_value"]} [p-value={df_column_threshold_survival.loc[0, "cox_coef_p_value"]:.2f}]
Cox hazard-ratio: exp(coef) = {df_column_threshold_survival.loc[0, "cox_exp(coef)"]:.2f}
Cox hazard variation: exp(coef) - 1 = {df_column_threshold_survival.loc[0, "cox_exp(coef)"] - 1:.2f}
Cox variation in surviral: 1/exp(coef) -1 = {(1 / df_column_threshold_survival.loc[0, "cox_exp(coef)"]) -1:.2f}
Log-rank signif p-value: {df_column_threshold_survival.loc[0, "log_rank_signif_p_value"]} [p-value={df_column_threshold_survival.loc[0, "log_rank_p_value"]:.2f}]
Surv in time signif p-value: {df_column_threshold_survival.loc[0, "survival_in_time_signif_p_value"]} [p-value={df_column_threshold_survival.loc[0, "survival_in_time_p_value"]:.2f}]
RMST above over below percent: {df_column_threshold_survival.loc[0, "rmst_above_over_below_%"]:.1f}%"""
            # plot title
            ax.set_title(f"{msg}", fontweight="normal")

        # check if user want to plot threshold baseline
        if self.plot_threshold_baseline:
            # plot cox threshold survival
            cph_threshold.baseline_survival_.plot(
                label="Cox baseline", linestyle="dotted", color="black", ax=ax
            )

        # plot km threhold survival curves
        km_above.plot_survival_function(
            label=df_column_threshold_survival.loc[0, "above_label"],
            color=self.blue_strong,
            ax=ax,
        )
        km_below.plot_survival_function(
            label=df_column_threshold_survival.loc[0, "below_label"],
            color=self.pink_strong,
            ax=ax,
        )

        # get legend data
        (
            current_handles,
            current_labels,
        ) = ax.get_legend_handles_labels()  # get current handles and labels
        new_handles = list(current_handles[:])

        # # assign a flag in case threshold type is boolean
        # print(self.threshold_type) #df_column_threshold_survival.loc[0, 'threshold_label'] == "boolean"

        # check if variable is boolean and not above and/or below label was defined
        if (
            (df_column_threshold_survival.loc[0, "threshold_label"] == "boolean")
            and (self.above_label == "Above threshold")
            and (self.below_label == "Below threshold")
        ):
            # reassign above_label and below_label
            self.above_label, self.below_label = (
                f"{self.feature_col} = True",
                f"{self.feature_col} = False",
            )

        # check if user want to plot threshold baseline
        if self.plot_threshold_baseline:
            # define labels
            new_labels = list(
                [self.baseline_labels, self.above_label, self.below_label]
            )
        # user don't want to plot threshold baseline
        else:
            # define labels
            new_labels = list([self.above_label, self.below_label])

        # define plot details
        ax.set_ylabel(self.y_axis_label)
        ax.set_xlabel(self.x_axis_label)
        if self.threshold_legend is None:
            ax.legend(
                new_handles,
                new_labels,
                title=f"Threshold: {df_column_threshold_survival.loc[0, 'threshold_label']}",
                loc="lower left",
            )
        else:
            ax.legend(
                new_handles, new_labels, title=self.threshold_legend, loc="lower left"
            )
        ax.set_xticks(
            [
                *range(
                    self.dataframe[self.duration_col].min(),
                    self.dataframe[self.duration_col].max() + 1,
                )
            ]
        )
        ax.set_yticks([*np.arange(0.0, 1.1, 0.1)])
        ax.set_ylim(0, 1.05)

    def plot_cox_analysis_(self, df_column_cox_survival, cph_cox, ax) -> None:
        """
        Plot cox analysis
        """

        # import required libraries
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # check if user defined a cox_title
        if self.cox_title is not None:
            # plot title
            ax.set_title(f"{self.cox_title}")

        # check if user wants title with statistics
        elif self.plot_stats:
            # define message to plot
            msg = f"""Column: {df_column_cox_survival.loc[0, "column"]}
Cox assumptions: {df_column_cox_survival.loc[0, "passing_cox_signif_p_value"]} [p-value: {df_column_cox_survival.loc[0, "passing_cox_p_value"]:.2f}]
Cox coef signif p-value: {df_column_cox_survival.loc[0, "cox_coef_signif_p_value"]} [p-value: {df_column_cox_survival.loc[0, "cox_coef_p_value"]:.2f}]
Cox hazard-ratio: exp(coef) = {df_column_cox_survival.loc[0, "cox_exp(coef)"]:.2f}
Cox hazard variation: exp(coef) - 1 = {df_column_cox_survival.loc[0, "cox_exp(coef)"] - 1:.2f}
Cox variation in surviral: 1/exp(coef) -1 = {(1 / df_column_cox_survival.loc[0, "cox_exp(coef)"]) -1:.2f}
Cox Concordance index: {df_column_cox_survival.loc[0, "cox_concordance_index"]:.2f}
Cox AIC partial: {df_column_cox_survival.loc[0, "cox_AIC_partial"]:.2f}
Cox log likelihood: {df_column_cox_survival.loc[0, "cox_log_likelihood"]:.2f}
KM restricted mean surv time: {df_column_cox_survival.loc[0, "restricted_mean_surv_time"]:.2f}"""

            # define plot details
            ax.set_title(f"{msg}", fontweight="normal")

        # check if user want to define cox partial values to plot
        if self.cox_partial_values is None:

            # check if variable is boolean
            if df_column_cox_survival.loc[0, "bool_col"]:
                # define values to plot
                cox_partial_values_plot = [False, True]

            # column is not boolean
            else:
                # get column quantiles to plot
                cox_partial_values_plot = list(
                    set(
                        self.dataframe[self.feature_col]
                        .quantile(q=[0.05, 0.25, 0.5, 0.75, 0.95])
                        .tolist()
                    )
                )
                cox_partial_values_plot = [
                    int(item)
                    if self.round_cox_quantiles == 0
                    else round(item, self.round_cox_quantiles)
                    for item in cox_partial_values_plot
                ]
                cox_partial_values_plot.sort(reverse=True)

        # user wants specific values on cox plot
        else:
            # sort user specific values
            cox_partial_values_plot = self.cox_partial_values
            cox_partial_values_plot.sort(reverse=True)

        # create a mapping dict for color map
        cmap_dict = {
            1: ["#045893"],
            2: ["#045893", "#B40C0D"],
            3: ["#045893", "#108010", "#B40C0D"],
            4: ["#045893", "#108010", "#DB6100", "#B40C0D"],
            5: ["#984EA3", "#045893", "#108010", "#DB6100", "#B40C0D"],
            6: ["#636363", "#984EA3", "#045893", "#108010", "#DB6100", "#B40C0D"],
        }

        # define color map according to number of lines to plot
        color_map = cmap_dict[len(cox_partial_values_plot)]

        # check if variable is boolean -> two lines to plot
        if self.dataframe[self.feature_col].dtype == "bool":
            # reverse order of color map
            color_map.sort(reverse=True)

        # plot cox curves for quantiles
        cph_cox.plot_partial_effects_on_outcome(
            covariates=[self.feature_col],
            values=cox_partial_values_plot,
            y="survival_function",
            plot_baseline=False,
            cmap=ListedColormap(color_map),
            ax=ax,
        )

        # check if user want to plot cox baseline
        if self.plot_cox_baseline:
            # plot cox baseline survival
            cph_cox.baseline_survival_.plot(linestyle="dotted", color="black", ax=ax)

        # get legend data
        (
            current_handles,
            current_labels,
        ) = ax.get_legend_handles_labels()  # get current handles and labels
        new_handles = list(current_handles[:])
        # check if user want cox baseline
        if self.plot_cox_baseline:
            # check if user defined a specific column name
            if self.cox_col_name_legend is None:
                # define labels
                new_labels = list(
                    [self.baseline_labels]
                    + [item.replace("=", " = ") for item in current_labels[:-1]]
                )
            # no specific column name was defined
            else:
                # define labels
                new_labels = list(
                    [self.baseline_labels]
                    + [
                        item.replace("=", " = ").replace(
                            self.feature_col, self.cox_col_name_legend
                        )
                        for item in current_labels[:-1]
                    ]
                )
        # user don't want cox baseline
        else:
            # check if user defined a specific column name
            if self.cox_col_name_legend is None:
                # define labels
                new_labels = list(
                    [self.baseline_labels]
                    + [item.replace("=", " = ") for item in current_labels[:]]
                )
            # no specific column name was defined
            else:
                # define labels
                new_labels = list(
                    [self.baseline_labels]
                    + [
                        item.replace("=", " = ").replace(
                            self.feature_col, self.cox_col_name_legend
                        )
                        for item in current_labels[:]
                    ]
                )

        # define new handles with baseline as first item
        new_handles = [new_handles[-1]] + new_handles[:-1]

        # check if variable is boolean
        if self.dataframe[self.feature_col].dtype == "bool":
            # define new handles and labels
            new_handles = [new_handles[0]] + [new_handles[2]] + [new_handles[1]]
            new_labels = [new_labels[0]] + [new_labels[2]] + [new_labels[1]]
            # display legend
            ax.legend(new_handles, new_labels, loc="lower left")
        # variable is not boolean
        else:
            # define legend
            ax.legend(new_handles, new_labels, loc="lower left")

        # define plot details
        ax.set_ylabel(self.y_axis_label)
        ax.set_xlabel(self.x_axis_label)
        ax.set_xticks(
            [
                *range(
                    self.dataframe[self.duration_col].min(),
                    self.dataframe[self.duration_col].max() + 1,
                )
            ]
        )
        ax.set_yticks([*np.arange(0.0, 1.1, 0.1)])
        ax.set_ylim(0, 1.05)


def categorical_survival_inspection(
    dataframe: pd.DataFrame,
    duration_col: str,
    event_col: str,
    feature_col: str,
    feature_values: list = None,
    max_duration: int = None,
    title: str = None,
    figsize: tuple = (10, 5),
    y_axis_label: str = "Survival curve",
    x_axis_label: str = "Timeline",
    color_map: Union[str, list] = None,
    ci_show_flag: bool = None,
    show_censors: bool = False,
    saving_path: str = None,
    signif_level: float = 0.05,
    display_hypothesis_tests: bool = False,
    display_life_table_stats: bool = False,
    display_life_table: bool = False,
) -> None:
    """
    It plots a survival curve (based on Kaplan-Meier estimator) for all unique values
    on the dataframe column (of the values specified via feature_values params).

    It can also calculate (and display) the Multivariate Logrank Test as well as
    Pairwise Logrank Test among the survival curves.

    It has some plots customizations as well as options to
    display lifetables and statistics of survival analysis


    Args
        dataframe: a pandas.DataFrame with the data to use for survival analysis.
        duration_col: a str with the column name of durations information for survival analysis.
        event_col: a str with the column name of event occurence information for survival analysis.
        feature_col: a str with the column name of the feature to plot different survival curves.
        feature_values: a list with values of the feature_col column to plot different survival curves.
        max_duration: an int with the maximum x-axis of the duration information of survival curve.
        title: a str with the title of the figure to plot.
        figsize: a tuple with the figure size to plot.
        y_axis_label: a str with the y-axis label of the figure to plot.
        x_axis_label: a str with the x-axis label of the figure to plot.
        color_map: a string (with cmap name) or a list (of colors) to use for the different survival curves.
        ci_show_flag: a bool to indicate if confidence intervals are to be plot for survival curves.
        show_censors: a bool to indicate if censorsed data is to be plot for survival curves.
        saving_path: a str with the path to save figure.
        signif_level: a float with the confidence interval for the analysis.
        display_hypothesis_tests: a bool to indicate if hypothesis test results are to be displayed.
        display_life_table_stats: a bool to indicate if life table statistics are to be plot for survival curves.
        display_life_table: a bool to indicate if life table is to be plot for survival curves.

    Return
        None: a NoneType object
    """

    # import required libraries
    from itertools  import combinations
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import get_cmap
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
    from lifelines.utils import restricted_mean_survival_time

    # define figure layout
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, tight_layout=True)

    # create a mapping dict for color map
    cmap_dict = {
        1: ["#045893"],
        2: ["#045893", "#B40C0D"],
        3: ["#045893", "#108010", "#B40C0D"],
        4: ["#045893", "#108010", "#DB6100", "#B40C0D"],
        5: ["#984EA3", "#045893", "#108010", "#DB6100", "#B40C0D"],
        6: ["#636363", "#984EA3", "#045893", "#108010", "#DB6100", "#B40C0D"],
    }

    # check if testing values was input
    if feature_values is None:
        # get unique values on feature columns
        unique_values = dataframe[feature_col].unique().tolist()

    # testing values was input
    else:
        # define unique values as input values
        unique_values = feature_values

    # check if color_map as input
    if color_map is None:
        # check length of unique values
        if len(unique_values) <= 6:
            # create a color to map values
            custom_cmap = cmap_dict[len(unique_values)]
        # length > 6 and <= 10
        elif len(unique_values) <= 10:
            # get colors from tab10
            cmap = get_cmap("tab10")
            # get a list of available colors to plot
            custom_cmap = cmap(np.linspace(0, 1, len(unique_values)))
        #
        else:
            # get colors from tab10
            cmap = get_cmap("gist_ncar_r")
            # get a list of available colors to plot
            custom_cmap = cmap(np.linspace(0, 1, len(unique_values)))

    # color_map is a string with plt cmap
    elif isinstance(color_map, str):
        # get colors from tab10
        cmap = get_cmap(color_map)
        # get a list of available colors to plot
        custom_cmap = cmap(np.linspace(0, 1, len(unique_values)))

    # check if color_map input is a list of colors
    elif isinstance(color_map, list):
        # check length of list
        if len(color_map) == len(unique_values):
            # create a color to map values
            custom_cmap = color_map
        # length of input list is wrong
        else:
            # raise exception with message error
            raise ValueError(
                "Input color_map list must have as many values as there are unique values in the given column."
                f"\nThere are {len(color_map)} value(s) in color map input but there are {len(unique_values)} unique value(s) on the given column."
            )

    # create an empty dataframe to store life table statistics
    df_life_stats = pd.DataFrame(
        columns=[
            "feature",
            "unique value",
            "total censored + total observed",
            "total censored",
            "total observed",
            "total_observed / (total censored + total observed)",
            "RMST min duration",
            "RMST max duration",
            "RMST [t=min duration, t=max_duration] [closed interval]",
            "RMST [t=min duration, t=max_duration] / (total_observed / (total censored + total observed))",
        ]
    )

    # iterate over unique values of feature columns
    for idx, value in enumerate(unique_values):

        # filter dataframe
        df_plot = dataframe[dataframe[feature_col] == value]

        # instanciate KM model
        kmf = KaplanMeierFitter(alpha=signif_level)
        # fit model to data
        kmf.fit(
            durations=df_plot[duration_col],
            event_observed=df_plot[event_col],
        )

        # check ci_show flag
        if ci_show_flag is None:
            # get a flag to show ci_intervals
            ci_show_flag = len(unique_values) < 5

        # plot
        kmf.plot_survival_function(
            label=value,
            color=custom_cmap[idx],
            ci_show=ci_show_flag,
            show_censors=show_censors,
            censor_styles={"ms": 5, "marker": ".", "markeredgecolor": "black"},
            ax=ax,
        )

        # check if a max timeline value of interest was input
        if max_duration is None:
            # define max duration as the max duration available value
            max_duration = dataframe[duration_col].max()

        # check if user wants to display statistics
        if display_life_table_stats:
            # define event table as a dataframe
            df_event_table = kmf.event_table

            # define a variable to column names with statistics of interest
            rate_obs_total = "total_observed / (total censored + total observed)"
            RMST = "RMST [t=min duration, t=max_duration] [closed interval]"
            rate_RMST_rate_obs_total = "RMST [t=min duration, t=max_duration] / (total_observed / (total censored + total observed))"

            # create a dataframe with life table statistics
            df_life_stats_unique_value = pd.DataFrame(
                data={
                    "feature": feature_col,
                    "unique value": value,
                    "total censored + total observed": df_event_table["entrance"].max(),
                    "total censored": df_event_table["censored"].sum(),
                    "total observed": df_event_table["observed"].sum(),
                    rate_obs_total: df_event_table[
                        "observed"
                    ].sum()
                    / df_event_table["entrance"].max(),
                    "RMST min duration": 0,
                    "RMST max duration": max_duration,
                    RMST: restricted_mean_survival_time(
                        kmf, max_duration
                    ),
                    rate_RMST_rate_obs_total: restricted_mean_survival_time(
                        kmf, max_duration
                    )
                    / (
                        df_event_table["observed"].sum()
                        / df_event_table["entrance"].max()
                    ),
                },
                index=[0],
            )

            # concatenate life table stats to dataframe with all stats
            df_life_stats = pd.concat(
                objs=[df_life_stats, df_life_stats_unique_value],
                axis=0,
                ignore_index=True,
            )

        # check if user wants to check life table
        if display_life_table:
            print(
                f"\n{'*'*49}\n\n"
                f'Life table for unique value "\033[1m{value}\033[0m" of column "\033[1m{feature_col}\033[0m":\n'
            )
            display(kmf.event_table)

    # check if user wants to display statistics
    if display_life_table_stats:
        # sort dataframe
        df_life_stats = df_life_stats.sort_values(
            by=[
                RMST,
                rate_obs_total,
                rate_RMST_rate_obs_total, 
            ],
            ascending=[False, True, False],
            ignore_index=True,
        )
        # display dataframe
        print(
            f"\n{'*'*49}\n\n"
            f'Statistics of life tables for column "\033[1m{feature_col}\033[0m":\n'
        )
        # make sure scientific notation is appropriate to interpret results
        with pd.option_context('display.float_format', lambda x: f'{x:,.3f}'):
            # display dataframe
            display(df_life_stats)

        # print header to new section
        print(
        f'\n\t\033[1mRate statistics, on the form "stats[numerator] / stats[denominator]", between unique values of the above dataframe:\033[0m\n'
        )

        # create a dataframe to hold rate information
        df_survival_rates = pd.DataFrame(columns=["numerator", "denominator", rate_obs_total, RMST, rate_RMST_rate_obs_total])

        # iterate over all pairs of unique values
        for index, pair in enumerate( combinations(iterable=df_life_stats["unique value"].tolist(), r=2) ):
            # define numerator and denominator
            numerator = pair[0]
            denominator = pair[1]
            
            # get statistics from life table stats
            rate_obs_total_num = df_life_stats.loc[
                df_life_stats["unique value"] == numerator, rate_obs_total
            ].tolist()[0],
            rate_obs_total_denom = df_life_stats.loc[
                df_life_stats["unique value"] == denominator, rate_obs_total
            ].tolist()[0],
            RMST_num = df_life_stats.loc[
                df_life_stats["unique value"] == numerator, RMST
            ].tolist()[0],
            RMST_denom = df_life_stats.loc[
                df_life_stats["unique value"] == denominator, RMST
            ].tolist()[0],
            rate_RMST_rate_obs_total_num = df_life_stats.loc[
                df_life_stats["unique value"] == numerator, rate_RMST_rate_obs_total
            ].tolist()[0],
            rate_RMST_rate_obs_total_denom = df_life_stats.loc[
                df_life_stats["unique value"] == denominator, rate_RMST_rate_obs_total
            ].tolist()[0],

            # add values to df_survival_rates
            df_survival_rates.loc[index, "numerator"] = numerator
            df_survival_rates.loc[index, "denominator"] = denominator
            df_survival_rates.loc[index, rate_obs_total] = rate_obs_total_num[0] / rate_obs_total_denom[0]
            df_survival_rates.loc[index, RMST] = RMST_num[0] / RMST_denom[0]
            df_survival_rates.loc[index, rate_RMST_rate_obs_total] = rate_RMST_rate_obs_total_num[0] / rate_RMST_rate_obs_total_denom[0]

        # sort dataframe
        df_survival_rates = df_survival_rates.sort_values(
            by=[
                "RMST [t=min duration, t=max_duration] [closed interval]",
                "total_observed / (total censored + total observed)",
                "RMST [t=min duration, t=max_duration] / (total_observed / (total censored + total observed))",
            ],
            ascending=[False, True, False],
            ignore_index=True,
        )   
        # make sure scientific notation is appropriate to interpret results
        with pd.option_context('display.float_format', lambda x: f'{x:,.3f}'):
            # display dataframe
            display(df_survival_rates)

    # define plot details
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_xticks([*range(0, max_duration + 1)])
    ax.set_yticks([*np.arange(0.0, 1.1, 0.1)])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, max_duration + 0.5)
    if title is not None:
        ax.set_title(f"{title}")
    else:
        ax.set_title(f"{feature_col}")

    # get labels and handles from image
    current_handles, current_labels = ax.get_legend_handles_labels()
    new_handles = list(current_handles[:])
    if isinstance(current_labels[0], (int, float)):
        new_handles = [new_handles[i] for i in np.argsort(current_labels)]
        new_labels = np.sort(current_labels)
    else:
        new_labels = current_labels
    new_labels = [f"{feature_col} = {label}" for label in new_labels[:]]
    ax.legend(new_handles, new_labels, loc="lower left")

    # check if summary is to be print
    if display_hypothesis_tests:
        # perform multivariate logrank test
        mult_variate_t = multivariate_logrank_test(
            event_durations=dataframe[duration_col],
            groups=dataframe[feature_col],
            event_observed=dataframe[event_col],
            t_0=-1,
        )

        # performe pairwise logrank test
        pairwise_t = pairwise_logrank_test(
            event_durations=dataframe[duration_col],
            groups=dataframe[feature_col],
            event_observed=dataframe[event_col],
            t_0=-1,
        )

        # print reports
        print(
            f"{'*'*49}\n\n"
            "\033[1m"
            "\t\t"
            f"feature: {feature_col.upper()}"
            "\033[0m"
            "\033[1m"
            "\n\n\n"
            f"---> MULTIVARIATE LOGRANK TEST"
            "\033[0m"
            "\n\n"
            "\033[1m"
            "Null hypothesis: there are no difference between groups\n"
            "This test is a generalization of the logrank_test"
            "\033[0m"
            "\n\n"
        )
        # get multi variate results
        df_multi_variate_t = mult_variate_t.summary
        # change index name
        df_multi_variate_t.index = ["multivariate_logrank_test"]
        print(df_multi_variate_t.to_string(float_format=lambda x: f"{x:.3f}"))

        # mult_variate_t.print_summary()
        print(
            #            f"{'*'*49}\n\n"
            "\n\n"
            "\033[1m"
            "---> PAIRWISE LOGRANK TEST"
            "\033[0m"
            "\n\n"
            "\033[1m"
            "Null hypothesis: there are no difference between groups\n"
            "This is a logrank test pairwisely for all n≥2 unique groups"
            "\033[0m"
            "\n\n"
        )
        # get pairwise variate results
        df_pairwise_t = pairwise_t.summary
        print(df_pairwise_t.to_string(float_format=lambda x: f"{x:.3f}"), f"\n\n{'*'*49}\n")

    # check if user want to save figure
    if saving_path is not None:

        # save figure
        plt.savefig(saving_path, facecolor="white", bbox_inches="tight", dpi=900)

    # display figure
    plt.show()

    return df_life_stats
