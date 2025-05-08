from typing import List
from matplotlib.ticker import MaxNLocator
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


def horizontal_bar_plot(
        data_frame: pd.DataFrame,
        x_col: str,
        y_col: str,
        title:  str = None,
        values: str = None,
        label_x_offset: int = 500
) -> None:
    """
    Method to generate a horizontal bar plot from a DataFrame.

    :param data_frame: The DataFrame containing the data to be plotted.
    :param x_col: The column in the DataFrame for the x-axis (horizontal bars).
    :param y_col: The column in the DataFrame for the y-axis (bar lengths).
    :param title: The title of the plot.
    :param values: The label for the x-axis.
    :param label_x_offset: Offset for positioning the value labels next to the bars.
    :return: None.
    """

    data_frame = data_frame.sort_values(by=y_col, ascending=False)

    plt.figure(figsize=(15, 6))
    bars = plt.barh(data_frame[x_col], data_frame[y_col], edgecolor='black', color='lightgrey')
    plt.xlabel(values)
    plt.title(title)
    plt.gca().invert_yaxis()

    for bar in bars:
        width = bar.get_width()
        label_x = width + label_x_offset
        label_y = bar.get_y() + bar.get_height() / 2
        plt.text(label_x, label_y, int(width), ha='center', va='center', color='black')

    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.show()

def combined_box_bar_plot(
        data_frame1: pd.DataFrame,
        data_frame2: pd.DataFrame,
        x_col1: str,
        y_col1: str,
        x_col2: str,
        y_col2: str,
        order: List[str],
        title1: str = None,
        title2: str = None,
        x_label1: str = None,
        y_label1: str = None,
        x_label2: str = None,
        y_label2: str = None
) -> None:
    """
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    sns.barplot(data=data_frame1, x=x_col1, y=y_col1, palette='Pastel1', edgecolor='black', color='lightgrey', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_xlabel(x_label1)
    ax1.set_ylabel(y_label1)
    ax1.set_title(title1)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')

    for p in ax1.patches:
        height = int(round(p.get_height()))
        ax1.annotate(str(int(round(height))), (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    sns.boxplot(data=data_frame2, x=x_col2, y=y_col2, palette='Pastel1', order=order, ax=ax2)
    ax2.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax2.set_xlabel(x_label2)
    ax2.set_ylabel(y_label2)
    ax2.set_title(title2)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_area_stacked(
        data_frame: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        legend_title: str = None
) -> None:
    """
    Method to plot a stacked area chart from the data in a DataFrame.

    :param data_frame: The input dataframe containing the data.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :param legend_title: Title of the legend.
    :return: None.
    """

    df_plot = data_frame[:-1]
    df_plot = df_plot.transpose()
    
    ax = df_plot.plot(kind='area', stacked=True, cmap='tab20b',  figsize=(14, 6))
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=len(data_frame.columns)))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
   
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, frameon=False, title=legend_title)
    
    plt.show()

def plot_boxplot_distribution(
        data_frame: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        figsize: tuple = (8, 6)
) -> None:
    """
    Method to generate a boxplot to visualize distribution in a DataFrame.

    :param data_frame: The input dataframe containing the data.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :param figsize: Size of the figure (width, height).
    :return: None.
    """

    plt.figure(figsize=figsize)
    
    data_frame.boxplot()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

def plot_percentage_bars(
        data_frame: pd.DataFrame,
        columns: List[str] = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to generate a bar plot to visualize the relative
    values (in percentage) of selected columns.

    :param data_frame: The input dataframe containing the data.
    :param columns: The list of column names to plot the percentage bars.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :return: None.
    """

    num_bars = len(data_frame[columns].columns)
    fig = plt.figure(figsize=(16, 4) if num_bars >= 5 else (5, 4))

    ax = fig.gca()
    ax.spines[['top', 'right']].set_visible(False)

    total_sum = data_frame[columns].iloc[-1].sum()

    for i, (col, values) in enumerate(data_frame[columns].items()):
        x = [i]
        y = [values.iloc[-1]]

        plt.bar(x, y, edgecolor='black', color='none')
        plt.text(x[0], y[0], f'{(y[0] / total_sum * 100):.2f}%', ha='center', va='bottom', fontsize=7) if y[0] != 0 else None

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=30)

    x_ticks = range(num_bars)
    x_tick_labels = data_frame[columns].columns
    plt.xticks(x_ticks, x_tick_labels, ha='center')
    
    plt.show()

def plot_barplot_distribution_by_year(
        data_frame: pd.DataFrame,
        x_label: str = None,
        y_label: str = None
) -> None:
    """
    Method to plot bar plots for each year in the data frame,
    showing the distribution of data.

    :param data_frame: The input dataframe containing the data.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :return: None.
    """

    df_plot = data_frame.iloc[:, :-1]
    years = df_plot.columns
    total_counts = df_plot.sum(axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(17, 6), sharey=True)
    
    for i, year in enumerate(years):
        ax = axes[i]
        ax = df_plot[year].plot(kind='bar', ax=ax, edgecolor='black', color='lightgray')
        ax.set_title(year)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        total_count = total_counts[year]
        for rect in ax.patches:
            height = rect.get_height()
            percentage = (height / total_count) * 100
            ax.annotate(f'{percentage:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            
    plt.tight_layout()

    plt.show()

def perform_z_test(
    sample_data: (np.ndarray, pd.Series), 
    population_data: (np.ndarray, pd.Series), 
    alpha: float = 0.05
)-> str:
    
    """
    Method to perform a two-sided z-test to compare sample data to a population.

    :param sample_data: NumPy array or Pandas Series of sample data.
    :param population_data: NumPy array or Pandas Series of population data.
    :param alpha: Significance level.
    :return: A string indicating the result of the test.
    """
    
    z_statistic, p_value = sm.stats.ztest(sample_data, population_data, alternative='two-sided')

    if p_value < alpha:
        result = "Reject the null hypothesis. There is a significant difference between the sample and the population."
    else:
        result = "Fail to reject the null hypothesis. There is no significant difference between the sample and the population."

    return print(f"Calculated z-statistic value = {z_statistic}, corresponding p-value = {p_value}.\n{result}")

def calculate_confidence_interval(
    confidence: float,
    sample_mean: float,
    sample_std: float,
    sample_size: int
)-> tuple:
    
    """
    Method to calculate a confidence interval for a normally distributed population.

    :param confidence: The desired confidence level.
    :param sample_mean: The sample mean (x-bar).
    :param sample_std: The sample standard deviation (sigma).
    :param sample_size: The sample size (n).
    :return: A tuple (lower_bound, upper_bound) representing the confidence interval.
    """
    z_score = stats.norm.interval(confidence)[1]
    sigma_over_root_n = sample_std / np.sqrt(sample_size)
    lower_bound = sample_mean - z_score * sigma_over_root_n
    upper_bound = sample_mean + z_score * sigma_over_root_n
    return lower_bound, upper_bound

def plot_scatter_plot(
        x_data: List[float],
        y_data: List[float],
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        alpha: float = 1.0
) -> None:
    """
    Method to create a scatter plot to visualize the relationship between two variables.

    Parameters:
    :param x_data: Data for the x-axis.
    :param y_data: Data for the y-axis.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :param alpha: Transparency of data points.
    :return: None.
    """

    plt.figure(figsize=(12, 6))
    plt.scatter(x_data, y_data, color='blue', marker='o', alpha=alpha)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_correlation_heatmap(
        correlation_matrix: pd.DataFrame
) -> None:
    """
    Method to plot a correlation heatmap.

    :param correlation_matrix: The correlation matrix as a DataFrame.
    :return: None.
    """
    sns.heatmap(correlation_matrix, cmap='gray', annot=True, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def find_corr_pairs(
        data_frame: pd.DataFrame,
        correlation_type: str
) -> None:
    """
    Method to find unique correlated pairs in a dataframe based on the specified correlation type.
    
    :param data_frame: The input dataframe containing the data.
    :param correlation_type: The type of correlation to find. Possible values:
         - "positive_strong",
         - "positive_moderate",
         - "weak",
         - "negative_moderate",
         - "negative_strong".
    :return: None.
    """

    correlation_unstacked = data_frame.unstack()
    correlation_pairs_sorted = correlation_unstacked.sort_values(kind="quicksort")

    threshold_ranges = {
        "positive_strong": (0.6, 1),
        "positive_moderate": (0.2, 0.6),
        "weak": (-0.2, 0.2),
        "negative_moderate": (-0.6, -0.2),
        "negative_strong": (-1, -0.6)
    }

    if correlation_type in threshold_ranges:
        threshold_range = threshold_ranges[correlation_type]
        corr_pairs = correlation_pairs_sorted[(correlation_pairs_sorted > threshold_range[0]) & (correlation_pairs_sorted < threshold_range[1])]
        print(f"{correlation_type.replace('_', ' ').title()} correlated pairs (between {threshold_range[0]} and {threshold_range[1]}):")
        selected_pairs = []
        for pair, score in corr_pairs.items():
            sorted_pair = sorted(pair)
            if sorted_pair not in selected_pairs:
                print(f"Pair: {sorted_pair}, score: {score}")
                selected_pairs.append(sorted_pair)
    else:
        print("Invalid correlation type.")

def perform_two_sample_t_test(
    sample_data_1: (np.ndarray, pd.Series),
    sample_data_2: (np.ndarray, pd.Series),
    alpha: float = 0.05
)-> str:
    
    """
    Method to perform a two-sample t-test and provide a conclusion.

    :param sample_data1: First sample data.
    :param sample_data2: Second sample data.
    :param alpha: Significance level (default is 0.05).
    :return: A result based on the t-test result.
    """
    
    t_statistic, p_value = stats.ttest_ind(sample_data_1, sample_data_2)

    if p_value < alpha:
        result = 'Reject the null hypothesis. There is a significant difference between the means of two sample data.'
    else:
        result = 'Fail to reject the null hypothesis. There is no significant difference between the means of two sample data.'
    
    return print(f"Calculated t-statistic value = {t_statistic}, corresponding p-value = {p_value}.\n{result}")
