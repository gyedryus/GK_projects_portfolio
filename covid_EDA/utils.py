from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def linear_plot_of_changes(
        data_frame: pd.DataFrame,
        selected_columns: List[str],
        start_date: str = None,
        end_date: str = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        show_legend: bool = True,
        legend_title: str = None,
        alpha: float = 1.0
) -> None:
    """
    Method to generate a line plot to visualize the changes in selected columns over time.

    :param data_frame: The input dataframe containing the data.
    :param selected_columns: A list of column names to plot.
    :param start_date: Start date of the time range to include in the plot.
    :param end_date: End date of the time range to include in the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :param show_legend: Whether to display the legend.
    :param legend_title: Title for the legend.
    :param alpha: Transparency level of the lines.
        Value ranges from 0.0 to 1.0.
    :return: None.
    """

    if start_date and end_date:
        df = data_frame.loc[start_date:end_date]
    elif start_date and end_date == None:
        df = data_frame.loc[start_date:]
    elif start_date == None and end_date:
        df = data_frame.loc[:end_date]
    else:
        df = data_frame

    fig, ax = plt.subplots(figsize=(16, 6))
    for column in selected_columns:
        plt.plot(df.index, df[column].diff(), label=column, alpha=alpha)

    ax.spines[['top', 'right']].set_visible(False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=30) if title else None

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.split('_')[-1] for label in labels]
        ax.legend(handles, new_labels, title=legend_title)

    plt.xticks(rotation=45)
    plt.show()


def bar_plot_of_changes(
        data_frame: pd.DataFrame,
        selected_columns: List[str],
        start_date: str = None,
        end_date: str = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        show_legend: bool = True,
        legend_title: str = None,
        alpha: float = 1.0
) -> None:
    """
    Method to generate a bar plot to visualize the changes in selected columns over time.

    :param data_frame: The input dataframe containing the data.
    :param selected_columns: A list of column names to plot.
    :param start_date: Start date of the time range to include in the plot.
    :param end_date: End date of the time range to include in the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :param show_legend: Whether to display the legend.
    :param legend_title: Title for the legend.
    :param alpha: Transparency level of the lines.
        Value ranges from 0.0 to 1.0.
    :return: None.
    """

    if start_date and end_date:
        df = data_frame.loc[start_date:end_date]
    elif start_date and end_date == None:
        df = data_frame.loc[start_date:]
    elif start_date == None and end_date:
        df = data_frame.loc[:end_date]
    else:
        df = data_frame

    fig, ax = plt.subplots(figsize=(16, 6))
    for column in selected_columns:
        plt.bar(df.index, df[column].diff(), label=column, alpha=alpha)

    ax.spines[['top', 'right']].set_visible(False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=30) if title else None

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.split('_')[-1] for label in labels]
        ax.legend(handles, new_labels, title=legend_title)

    plt.xticks(rotation=45)
    plt.show()


def linear_plot(
        data_frame: pd.DataFrame,
        selected_columns: List[str],
        start_date: str = None,
        end_date: str = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None,
        show_legend: bool = True,
        legend_title: str = None
) -> None:
    """
    Method to generate a line plot to visualize the selected columns over time.

    :param data_frame: The input dataframe containing the data.
    :param selected_columns: A list of column names to plot.
    :param start_date: Start date of the time range to include in the plot.
    :param end_date: End date of the time range to include in the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :param show_legend: Whether to display the legend.
    :param legend_title: Title for the legend.
    :return: None.
    """

    if start_date and end_date:
        df = data_frame.loc[start_date:end_date]
    elif start_date and end_date == None:
        df = data_frame.loc[start_date:]
    elif start_date == None and end_date:
        df = data_frame.loc[:end_date]
    else:
        df = data_frame

    fig, ax = plt.subplots(figsize=(16, 6))
    for column in selected_columns:
        plt.plot(df.index, df[column], label=column)

    ax.spines[['top', 'right']].set_visible(False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=30)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [label.split('_')[-1] for label in labels]
        ax.legend(handles, new_labels, title=legend_title)

    plt.xticks(rotation=45)
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
    fig = plt.figure(figsize=(10, 4) if num_bars >= 5 else (5, 4))

    ax = fig.gca()
    ax.spines[['top', 'right']].set_visible(False)

    total_sum = data_frame[columns].iloc[-1].sum()

    renamed_columns = [col.split('_')[-1] for col in data_frame[columns].columns]

    for i, (col, values) in enumerate(data_frame[columns].items()):
        x = [i]
        y = [values.iloc[-1]]

        plt.bar(x, y, edgecolor='black', color='none')
        plt.text(x[0], y[0], f'{(y[0] / total_sum * 100):.2f}%', ha='center', va='bottom') if y[0] != 0 else None

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=30)
    plt.xticks(range(len(data_frame[columns].columns)), renamed_columns)

    plt.show()


def plot_average_bars(
        data_frame: pd.DataFrame,
        group_by: str,
        avg_parameter: str,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to generate a bar chart showing the average values
    of a specific parameter grouped by a given column.

    :param data_frame: The input dataframe containing the data.
    :param group_by: The column name to group the data by.
    :param avg_parameter: The parameter for which the average value
        will be calculated and visualized.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :return: None.
    """

    data_frame = data_frame[data_frame[avg_parameter] > 0]
    if group_by == 'age':
        data_frame['age_new'] = data_frame['age'].apply(lambda x: int(x.split('s')[0]))
        group_by = 'age_new'
        
    group_avg = data_frame.groupby(group_by)[avg_parameter].mean().reset_index()
    num_bars = len(group_avg)
    fig = plt.figure(figsize=(12, 4) if num_bars >= 5 else (5, 4))
    
    ax = sns.barplot(x=group_by, y=avg_parameter, data=group_avg)
    ax.spines[['top', 'right']].set_visible(False)
    
    for index, row in group_avg.iterrows():
        value = row[avg_parameter]
        if np.isfinite(value) and not np.isnan(value):
            if value or value > 0:
                plt.bar(row.name, value, edgecolor='black', color='lightgrey')
                ax.text(row.name, value, f'{value:.2f}', ha='center', va='bottom')
                
    labels = [tick.get_text() + 's' for tick in ax.get_xticklabels()]
    ax.set_xticklabels(labels)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=30)

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
