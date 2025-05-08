from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator


def plot_percentage_bars(
        data_frame: pd.DataFrame,
        column: str,
        x_label: str = None,
        y_label: str = None,
        title: str = None
) -> None:
    """
    Method to generate a bar plot to visualize the relative
    values (in percentage) of a selected column.

    :param data_frame: The input dataframe containing the data.
    :param column: The column name to plot the percentage bars.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param title: Title of the plot.
    :return: None.
    """

    value_counts = data_frame[column].value_counts()
    total = value_counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        value_counts.index,
        value_counts.values,
        edgecolor='black',
        color='lightgrey'
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{(height/total*100):.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                   )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    plt.show()

def plot_boxplots(
    data_frame: pd.DataFrame,
    target_column: str
) -> None:
    """
    Generates box plots for each numerical column in the DataFrame
    against a target column.

    :param df: The input DataFrame containing the data.
    :param target_column: The name of the column to be used as the x-axis.
    :return: None.
    """
    
    plt.figure(figsize=(20, 10))
    num_cols: List[str] = [
        col for col in data_frame.columns if col != target_column
    ]

    for idx, column in enumerate(num_cols):
        plt.subplot(len(num_cols) // 6 + 1, 6, idx + 1)
        sns.boxplot(
            x=target_column, y=column, data=data_frame, color='lightgrey', 
            boxprops=dict(edgecolor='black'), 
            whiskerprops=dict(color='black'), 
            capprops=dict(color='black'), 
            medianprops=dict(color='black')
        )

        formatted_title = column.replace('_', ' ').title()
        plt.title(f"Distribution of {formatted_title}")

    plt.tight_layout()
    plt.show()

def plot_feature_scatter_plots(
    data_frame: pd.DataFrame,
    features: Optional[List[str]] = None,
    hue: Optional[str] = None
) -> None:
    """
    Generates scatter plot pair plots for specified features in the DataFrame
    with an optional hue factor. If no features are specified, scatter plots for
    all features in the DataFrame are generated. KDE plots are included on the
    diagonal to visualize the distribution of each variable.

    :param data_frame: The input DataFrame containing the data.
    :param features: Optional list of features to plot. If None, all features in
                     the DataFrame are plotted.
    :param hue: Optional string specifying a column name to use for color-coding
                data points according to its unique values.
    :return: None.    
    """
    
    palette = "Greys" if hue else ["lightgrey"]
    
    plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'}
    if features:
        sns.pairplot(
            data_frame, vars=features, hue=hue, diag_kind="kde",
            plot_kws=plot_kws, palette=palette
        )
    else:
        sns.pairplot(
            data_frame, hue=hue, diag_kind="kde",
            plot_kws=plot_kws, palette=palette
        )

    plt.show()

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame
) -> None:
    """
    Method to plot rectangular data as a color-encoded matrix,
    ensuring that 0 is always represented as white.
    
    :param correlation_matrix: The correlation matrix as a DataFrame.
    :return: None.
    """
    
    grey_cmap = LinearSegmentedColormap.from_list(
        'custom_grey', [(0, 'black'), (0.5, 'white'), (1, 'black')]
    )
    
    plt.figure(figsize=(8, 6))
    
    vmax = max(correlation_matrix.max().max(), abs(correlation_matrix.min().min()))
    vmin = -vmax
    
    sns.heatmap(
        correlation_matrix, cmap=grey_cmap, annot=True, fmt=".2f",
        vmin=vmin, vmax=vmax
    )
    plt.title('Correlation Heatmap')
    plt.show()

def find_corr_pairs(
    data_frame: pd.DataFrame,
    correlation_types: List[str]
) -> None:
    """
    Method to find and print unique correlated pairs in a dataframe
    based on specified correlation types.
    
    :param data_frame: The input dataframe containing the data.
    :param correlation_types: A list of correlation types to find, including:
                               "positive_strong", "positive_moderate", "weak",
                               "negative_moderate", "negative_strong".
    """

    corr_matrix = data_frame.corr(numeric_only=True)
    corr_unstacked = corr_matrix.unstack()
    corr_pairs_sorted = corr_unstacked.sort_values(kind="quicksort")

    threshold_ranges = {
        "positive_strong": (0.6, 1),
        "positive_moderate": (0.2, 0.6),
        "weak": (-0.2, 0.2),
        "negative_moderate": (-0.6, -0.2),
        "negative_strong": (-1, -0.6)
    }

    for correlation_type in correlation_types:
        if correlation_type in threshold_ranges:
            threshold_range = threshold_ranges[correlation_type]
            corr_pairs = corr_pairs_sorted[
                (corr_pairs_sorted > threshold_range[0]) & (corr_pairs_sorted <= threshold_range[1])
            ]
            
            print(
                f"{correlation_type.replace('_', ' ').title()} correlated pairs"
                f"(between {threshold_range[0]} and {threshold_range[1]}):"
            )
            
            selected_pairs = []
            for (pair1, pair2), score in corr_pairs.items():
                if pair1 != pair2:
                    sorted_pair = tuple(sorted([pair1, pair2]))
                    if sorted_pair not in selected_pairs:
                        print(f"Pair: {sorted_pair}, Score: {score}")
                        selected_pairs.append(sorted_pair)
        else:
            print(f"Invalid correlation type: {correlation_type}.")

def perform_t_tests(
    data_frame: pd.DataFrame, 
    alpha: float, 
    status_column: str, 
    low_status: pd.DataFrame, 
    high_status: pd.DataFrame
) -> pd.DataFrame:
    """
    Method to perform T-tests to compare all numeric features
    between two groups within a DataFrame.

    :param data_frame: DataFrame containing wine data.
    :param alpha: Significance level for the T-tests.
    :param status_column: Name of the column indicating the group status.
    :param low_status: DataFrame containing the subset of data for the 'low' status group.
    :param high_status: DataFrame containing the subset of data for the 'high' status group.

    :return: A DataFrame with T-test results, including feature names, T-statistics, 
             P-values, and significance status.
    """

    t_test_results: List[Dict[str, float]] = []

    for column in data_frame.select_dtypes(
        include=['float64', 'int64']
    ).columns.drop(status_column, errors='ignore'):
        t_statistic, p_value = stats.ttest_ind(
            low_status[column], high_status[column],
            nan_policy='omit'
        )
        
        t_test_results.append({
            'Feature': column,
            'T-Statistic': t_statistic,
            'P-Value': p_value,
            'Significance': 'Yes' if p_value < alpha else 'No'
        })
    
    results_df = pd.DataFrame(t_test_results)
    
    return results_df

def calculate_confidence_intervals_z(
    data_frame: pd.DataFrame,
    confidence: float = 0.95
) -> pd.DataFrame:
    """
    Method to calculates confidence intervals for each numeric feature
    in the DataFrame using the Z-distribution.
    
    :param data_frame: DataFrame containing the data.
    :param confidence: Confidence level for the confidence intervals.
    
    :return: A DataFrame with features as rows and the lower and upper bounds
             of the confidence intervals as columns.
    """
    
    ci_df = pd.DataFrame(
        columns=['Mean', 'Lower CI', 'Upper CI']
    )
    
    features = data_frame.select_dtypes(include=[np.number]).columns
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    for feature in features:
        sample_data = data_frame[feature].dropna()
        sample_mean = sample_data.mean()
        sample_std = sample_data.std()
        sample_size = len(sample_data)
        
        sigma_over_root_n = sample_std / np.sqrt(sample_size)
        
        lower_bound = sample_mean - z_score * sigma_over_root_n
        upper_bound = sample_mean + z_score * sigma_over_root_n
        
        ci_df.loc[feature] = [sample_mean, lower_bound, upper_bound]
    
    return ci_df

def plot_qq(
    data_frame: pd.DataFrame,
    features: List[str]
)->None:
    """
    Method to generates QQ plots for the specified features
    against a normal distribution.

    :param data_frame: DataFrame containing the data.
    :param features: List of feature names to generate QQ plots for.
    :return: None.
    """
    
    num_plots = len(features)
    num_rows = num_plots // 6 + (num_plots % 6 > 0)
    fig, axs = plt.subplots(nrows=num_rows, ncols=6, figsize=(20, 4 * num_rows))
    axs = axs.flatten()
    
    for i, feature in enumerate(features):
        stats.probplot(data_frame[feature], dist="norm", plot=axs[i])
        axs[i].set_title('QQ plot for '+ feature)
    
    for ax in axs[len(features):]:
        ax.set_visible(False)
        
    plt.tight_layout()
    plt.show()

def apply_yeo_johnson(
    data_frame: pd.DataFrame,
    features:List[str]
)->pd.DataFrame:
    """
    Method to apply the Yeo-Johnson transformation to specified features
    of a DataFrame to make data more normally distributed. This method
    modifies the input DataFrame in-place and also returns it.

    :param data_frame: A DataFrame containing the data to be transformed.
    :param features: A list of strings specifying the column names of the features
                     to be transformed.
    :return: The transformed DataFrame with the specified features altered.
    """
    
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    data_transformed = pt.fit_transform(data_frame[features])
    data_frame[features] = data_transformed
    return data_frame

def evaluate_linear_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
)->None:
    """
    Evaluates a fitted linear regression model by calculating and printing
    the Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    and R-squared (R2) values. Also, generates plots for Actual vs. Predicted
    values and the Residuals.
    
    :param  model: Fitted linear regression model.
    :param X_test: Test features.
    :param y_test: True values for the test set.
    :return: None.
    """
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, c='0.4')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    line_start = min(y_test.min(), y_pred.min())
    line_end = max(y_test.max(), y_pred.max())
    plt.plot([line_start, line_end], [line_start, line_end], 'k--')
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, c='0.4', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, color='k', linestyle='--')
    
    plt.tight_layout()
    plt.show()

def evaluate_linear_model_rounded(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray
)->None:
    """
    Evaluates a fitted linear regression model with rounded predictions,
    printing out the MSE, RMSE, and R-squared (R2) values based on the rounded
    predictions. Additionally, generates a scatter plot of actual vs. predicted
    (rounded) values and a residuals plot for rounded predictions.
    
    :param model: Fitted linear regression model.
    :param X_test: Test features.
    :param y_test: True values for the test set.
    :return: None.
    """
    
    y_pred = model.predict(X_test)
    y_pred_rounded = np.round(y_pred)
    
    mse_rounded = mean_squared_error(y_test, y_pred_rounded)
    rmse_rounded = np.sqrt(mse_rounded)
    r2_rounded = r2_score(y_test, y_pred_rounded)
    
    print(f"Mean Squared Error (MSE) with Rounded Predictions: {mse_rounded}")
    print(f"Root Mean Squared Error (RMSE) with Rounded Predictions: {rmse_rounded}")
    print(f"R-squared (R2) with Rounded Predictions: {r2_rounded}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_rounded, alpha=0.5, c='0.4')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values (Rounded)')
    plt.title('Actual vs. Predicted Values')
    line_start = min(y_test.min(), y_pred_rounded.min())
    line_end = max(y_test.max(), y_pred_rounded.max())
    plt.plot([line_start, line_end], [line_start, line_end], 'k--')
    
    plt.subplot(1, 2, 2)
    residuals_rounded = y_test - y_pred_rounded
    plt.scatter(y_pred_rounded, residuals_rounded, alpha=0.5, c='0.4')
    plt.xlabel('Predicted Values (Rounded)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.axhline(y=0, linestyle='--', c='0')
    
    plt.xticks(np.unique(y_pred_rounded))
    
    plt.tight_layout()
    plt.show()

def remove_outliers(
    data_frame: pd.DataFrame,
    threshold: float = 3.0
)->pd.DataFrame:
    """
    Method to remove outliers from a DataFrame based on the Z-score method.
    Any feature with a Z-score greater than the specified threshold will be
    considered an outlier.
    
    :param data_frame: DataFrame containing the data to be processed.
    :param threshold: A float representing the Z-score threshold to identify outliers.
    :return: DataFrame with outliers removed.
    """
    
    z_scores = np.abs((data_frame - data_frame.mean()) / data_frame.std())
    
    return data_frame[(z_scores < threshold).all(axis=1)]

def print_confidence_interval_t(
    scores: np.array,
    alpha: float = 0.95
)->None:
    """
    Method to print the confidence interval for an array of scores
    using the t-distribution.

    :param scores: An array containing the scores.
    :param alpha: Confidence level for the confidence intervals, default is 0.95.
    :return: None.
    """

    mean_score = np.mean(scores)
    sem = stats.sem(scores)
    dof = len(scores) - 1
    t_score = stats.t.ppf((1 + alpha) / 2, dof)
    
    ci_lower = mean_score - t_score * sem
    ci_upper = mean_score + t_score * sem
    
    print(
        f"With {alpha*100:.0f}% confidence, the true prediction accuracy"
        f" of the linear regression model for predicting wine target variable"
        f" lies between {100*ci_lower:.2f}% and {100*ci_upper:.2f}%, based on "
        f"current red wine dataset."
    )

def model_feature_importance(
    model: BaseEstimator,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Method to create a DataFrame containing the feature importance
    coefficients from a linear model, adds a column for the absolute
    values of the coefficients, and sorts the features by their
    absolute coefficient values in descending order.

    :param model: The fitted linear model object from which to extract coefficients.
    :param X: The DataFrame used for model training, to extract feature names.
    :return: A DataFrame with columns for the feature names, their coefficients
    from the model, and the absolute values of the coefficients, sorted by
    the absolute values in descending order.
    """

    feature_importance = pd.DataFrame(
        model.coef_,
        index=X.columns,
        columns=["Coefficient"]
    )
    
    feature_importance["Abs_Coefficient"] = feature_importance["Coefficient"].abs()

    sorted_features = feature_importance.sort_values(
        by="Abs_Coefficient",
        ascending=False
    )
    
    return sorted_features

def plot_feature_importances_horizontal(
    data_frame: pd.DataFrame,
    column: str 
) -> None:
    """
    Method to plot horizontal bars of the coefficient values for all features.

    :param data_frame: DataFrame containing coefficients.
    :param column: Column name to plot.
    :return: None.
    """

    if column not in data_frame.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    sorted_dataframe = data_frame.sort_values(by=column, ascending=True)

    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        data=sorted_dataframe, 
        y=sorted_dataframe.index, 
        x=column, 
        orient='h', 
        color='lightgrey', 
        edgecolor='black'
    )

    plt.title(f'Feature Importance by {column}')
    plt.xlabel(f'{column} Value')
    plt.ylabel('Feature')
    
    for p in barplot.patches:
        width = p.get_width()
        plt.text(
            width + (sorted_dataframe[column].max() - sorted_dataframe[column].min()) * 0.005,
            p.get_y() + p.get_height() / 2,
            f'{width:.2f}',
            va='center',
            fontsize=9
        )
    
    plt.show()

def create_confidence_interval_df_t(
    data_array: np.array,
    confidence_level: float = 0.95
)->pd.DataFrame:
    """
    Method to create a DataFrame containing the mean of the data
    and the confidence interval.

    :param data_array: An array of data points.
    :param confidence_level: The confidence level for the interval.
    "return pd.DataFrame: A DataFrame with the mean and confidence interval.
    """
    
    mean_val = np.mean(data_array)

    sem = stats.sem(data_array)

    n = len(data_array)

    margin_of_error = sem * stats.t.ppf((1 + confidence_level) / 2., n - 1)

    confidence_interval = (mean_val - margin_of_error, mean_val + margin_of_error)

    data_frame = pd.DataFrame({
        'Mean Value': [mean_val],
        'Lower Bound CI': [confidence_interval[0]],
        'Upper Bound CI': [confidence_interval[1]]
    })

    return data_frame
