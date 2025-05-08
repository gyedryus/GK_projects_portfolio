from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_hist_distribution(
        data_frame: pd.DataFrame,
        column_name: str,
        bins: int,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to generate a plot to visualize distribution of a specified column.

    :param data_frame: The input dataframe containing the data.
    :param column_name: The name of the column to plot the histogram.
    :param bins: he number of bins to use in the histogram.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :return: None.
    """
    plt.hist(data_frame[column_name], bins=bins, color='lightgrey', edgecolor='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()


def plot_boxplot_distribution(
        data_frame: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to generate a boxplot to visualize distribution in a DataFrame.

    :param data_frame: The input dataframe containing the data.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :return: None.
    """

    data_frame.boxplot()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

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

    ax = df_plot.plot(kind='area', stacked=True, cmap='tab20b')
    ax.spines[['top', 'right']].set_visible(False)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, frameon=False, title=legend_title)

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

    df_plot = data_frame[:-1]
    years = df_plot.columns
    total_counts = df_plot.sum(axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=len(years), figsize=(17, 6))

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

    plt.tight_layout()

    plt.show()


def plot_percentage_bar_chart(
        data_frame: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to  plot a bar chart with percentage labels on top of each bar.

    :param data_frame: The input dataframe containing the data.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the legend.
    :return: None.
    """

    total_sum = data_frame.sum().sum()

    ax = data_frame.plot(kind='bar', edgecolor='black', color='lightgray', legend=False, width=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total_sum) * 100
        ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2, height), xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    ax.set_xlim(left=-1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    plt.show()


def plot_average_answer_length(
        data_frame: pd.DataFrame,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to plot a line graph of the average answer length.

    :param data_frame: The input dataframe containing the data.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the legend.
    :return: None.
    """

    data_frame.plot(kind='line', marker='o')
    plt.xticks(data_frame.index)
    plt.ylim(0, data_frame.values.max() * 1.1)

    plt.legend(frameon=False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

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


def plot_hexbin_scatter(
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to plot a hexbin scatter plot to visualize
    the relationship between two numerical variables.

    :param x_data: The input data for the x-axis.
    :param y_data: The input data for the y-axis.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :return: None.
    """

    plt.hexbin(x_data, y_data, gridsize=20, cmap='binary', alpha=0.3)
    plt.colorbar(label='Count')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

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
