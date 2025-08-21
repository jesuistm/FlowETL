import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from sklearn.ensemble import IsolationForest

def missing_values(columns : Dict[str, Any], abstraction : pd.DataFrame, features_schema : Dict[str, str]) -> pd.DataFrame:
    """
    Handle missing values following the configuration specified within the plan
    
    **Parameters**
    - `columns` : Dictionary mapping the columns where missing values should be handled to the respective handling strategy
    - `abstraction` : Pandas dataframe to be processed by this function
    - `features_schema` : Dictionary mapping the abstraction's columns to their respective inferred FlowETL data type

    **Column Configuration Structure Example**

    ```
    {
        column_name: {
            detect (optional): [list of values considered missing for this column],
            strategy (required): handling strategy,
            user_value (optional, depending on strategy): value for user-defined imputation
        }
    }
    ```

    **Available Strategies**
    - `drop_row`: Remove rows containing missing values
    - `drop_column`: Remove column if the majority of values are missing
    - `impute_user`: Replace with user-provided value
    - `impute_auto`: Automatically impute based on inferred column type
    - `mean`: Fill with column mean (numeric columns only)
    - `median`: Fill with column median (numeric columns only)
    - `mode`: Fill with column mode
    - `forward_fill`: Forward fill missing values
    - `backward_fill`: Backward fill missing values
    """
    
    # Keep track of columns to drop to avoid modifying dictionary during iteration
    columns_to_drop = []
    
    for column, config in columns.items():
        
        # Skip if column was already dropped
        if column not in abstraction.columns:
            continue
        
        # attempt to get the user-specified missing values for this column, otherwise 
        # default to NaN or "", which is how pandas interprets missing values for numers and strings
        detectables = config.get('detect', [np.nan, ""])

        # read the handling strategy defined for the current column
        strategy = config.get('strategy', 'impute_auto')

        # read the inferred type for the current column
        inferred_type = features_schema.get(column, 'Complex')

        # read the user value for imputation if provided
        user_value = config.get('user_value', 'MISSINGVALUE')

        # create mask for missing values in this column
        # we use both isna and isin to handle the problem where "np.nan != np.nan" in Pandas
        missing_mask = abstraction[column].isna() | abstraction[column].isin(detectables)
        
        # handle the missing value based on the strategy
        if strategy == 'drop_row':
            abstraction.drop(abstraction[missing_mask].index, inplace=True)

        # check whether the current column is composed mainly of missing values, defined in the 'detectables' list
        elif strategy == 'drop_column' and ((abstraction[column].isna() | abstraction[column].isin(detectables)).sum() > len(abstraction[column]) / 2):
            # mark column for removal
            columns_to_drop.append(column)

        elif strategy == 'impute_user':
            abstraction.loc[missing_mask, column] = user_value

        elif strategy == 'impute_auto':
            
            # impute automatically using the inferred FlowETL type for this column

            if inferred_type == 'Number':
                abstraction.loc[missing_mask, column] = 0
                # the .fillna method targets the entire column, hence further cell iterations are redundant
                continue

            elif inferred_type == 'String':
                abstraction.loc[missing_mask, column] = "N/A"
                continue

            elif inferred_type == 'Date':
                abstraction.loc[missing_mask, column] = "1/1/2000"
                continue

            elif inferred_type == 'Boolean':
                # replace with the majority value for this column
                if not abstraction[column].mode().empty:
                    mode_value = abstraction[column].mode()[0]
                    abstraction.loc[missing_mask, column] = mode_value
                continue

            elif inferred_type == 'Complex':
                # use a general placeholder for complex column types
                abstraction.loc[missing_mask, column] = "MISSINGVALUE"
                continue

        elif strategy == 'mean':
            mean_value = abstraction[column].mean()
            abstraction.loc[missing_mask, column] = mean_value

        elif strategy == 'median':
            median_value = abstraction[column].median()
            abstraction.loc[missing_mask, column] = median_value

        elif strategy == 'mode':
            if not abstraction[column].mode().empty:
                mode_value = abstraction[column].mode()[0]
                abstraction.loc[missing_mask, column] = mode_value

        elif strategy == 'forward_fill':
            abstraction[column] = abstraction[column].ffill()

        elif strategy == 'backward_fill':
            abstraction[column] = abstraction[column].bfill()

    # Drop columns that were marked for removal
    for column in columns_to_drop:
        abstraction.drop(column, axis=1, inplace=True)

    return abstraction


def duplicate_instances(abstraction : pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicated rows from the abstraction, leveraging Pandas' built-in `drop_duplicates` method
    **Parameters**
    - `abstraction` : Pandas dataframe to be processed by this function
    """
    return abstraction.drop_duplicates()


def outliers_anomalies(columns : Dict[str, Any], abstraction : pd.DataFrame, features_schema : Dict[str, str]) -> pd.DataFrame:
    """
    Detects and handles outliers and anomalies with column-specific strategies.

    **Parameters**
    - `columns`: Dictionary mapping column names to their outlier handling configuration
    - `abstraction` : Pandas dataframe to be processed by this function
    - `features_schema` : Dictionary mapping the abstraction's columns to their respective inferred FlowETL data type

    **Column Configuration Structure Example**
    ```
    {
        column_name: {
            normal_values (optional): definition of normal values (see below),
            strategy (required): handling strategy,
            user_value (optional, depending on strategy): value for user-defined imputation
        }
    }
    ```
    **Normal Values Possibilities (Detection Step)**
    - List: Explicit list of acceptable values
    - Range: Min/max bounds for acceptable values
    - Condition: Boolean expression defining normal values
    - Auto : automatically detect outliers for numerical features only using an Isolation Forest

    **Strategies:**
    - `drop`: Remove rows containing outliers
    - `impute_user`: Replace with user-provided value
    - `impute_auto`: Automatic imputation based on inferred FlowETL column type. 
    - `mean`: Replace with column mean (numeric only)
    - `median`: Replace with column median (numeric only)
    - `mode`: Replace with column's statistical mode
    """

    # collect all rows to drop (for columns using 'drop' strategy)
    rows_to_drop = set()

    for column, config in columns.items():

        # get normal values definition (how to detect outliers), default to "auto", 
        # which only attempts to find outliers in numerical columns
        normal_values = config.get('normal_values', 'auto')

        # get the outliers handling strategy
        strategy = config.get('strategy', 'impute_auto')

        # get the inferred FlowETL type for this column
        inferred_type = features_schema.get(column, 'Complex')

        # get the user value to be used for user-defined imputation of outliers
        user_value = config.get('user_value', 'OUTLIER_REPLACED')

        # create mask for outliers/anomalies in this column
        outlier_mask = _detect_outliers(abstraction[column], normal_values, inferred_type)

        # handle outliers based on strategy
        if strategy == 'drop':
            # collect rows with outliers to drop them all at once later
            rows_to_drop.update(abstraction[outlier_mask].index.tolist())

        elif strategy == 'impute_user':
            abstraction.loc[outlier_mask, column] = user_value
            
        elif strategy == 'impute_auto':
            # auto impute based on inferred column type
            if inferred_type == 'Number':
                # use median for numeric outliers (more robust than mean)
                # the ~ operator is the NOT operator for arrays, meaning we compute the median of all non-outlier cells
                median_value = abstraction.loc[~outlier_mask, column].median()
                abstraction.loc[outlier_mask, column] = median_value
                
            elif inferred_type == 'String':
                abstraction.loc[outlier_mask, column] = "ANOMALY_REPLACED"
                
            elif inferred_type == 'Date':
                # use mode (most common date) for date outliers
                if not abstraction.loc[~outlier_mask, column].mode().empty:
                    mode_value = abstraction.loc[~outlier_mask, column].mode()[0]
                    abstraction.loc[outlier_mask, column] = mode_value
                else:
                    abstraction.loc[outlier_mask, column] = "1/1/2000"
                    
            elif inferred_type == 'Boolean':
                # use mode for boolean outliers
                if not abstraction.loc[~outlier_mask, column].mode().empty:
                    mode_value = abstraction.loc[~outlier_mask, column].mode()[0]
                    abstraction.loc[outlier_mask, column] = mode_value
                    
            elif inferred_type == 'Complex':
                abstraction.loc[outlier_mask, column] = "ANOMALY_REPLACED"
                
        elif strategy == 'mean':
            # only for numeric columns
            if inferred_type == 'Number':
                mean_value = abstraction.loc[~outlier_mask, column].mean()
                abstraction.loc[outlier_mask, column] = mean_value
                
        elif strategy == 'median':
            # only for numeric columns
            if inferred_type == 'Number':
                median_value = abstraction.loc[~outlier_mask, column].median()
                abstraction.loc[outlier_mask, column] = median_value
                
        elif strategy == 'mode':
            if not abstraction.loc[~outlier_mask, column].mode().empty:
                mode_value = abstraction.loc[~outlier_mask, column].mode()[0]
                abstraction.loc[outlier_mask, column] = mode_value

    # drop all collected rows at once (if any columns used 'drop' strategy)
    if rows_to_drop:
        abstraction.drop(list(rows_to_drop), inplace=True)
    
    return abstraction


def _detect_outliers(series: pd.Series, normal_values: Union[str, List, Dict], inferred_type: str) -> pd.Series:
    """
    Helper function to detect outliers based on `normal_values` definition
    
    **Parameters**
    - `series`: the pandas series to analyze
    - `normal_values`: definition of normal values (auto, list, dict, or condition string)
    - `inferred_type`: the inferred flowetl data type
    
    Returns a boolean mask where True indicates an outlier
    """
    
    # handle different types of normal_values definitions
    if isinstance(normal_values, str) and normal_values.lower() == 'auto':

        # auto-detect outliers using isolation forest (for numeric data only)
        if inferred_type == 'Number':

            # remove missing values for analysis
            clean_series = series.dropna()
            
            if len(clean_series) < 10:  # need sufficient data for isolation forest
                return pd.Series(False, index=series.index)
            
            # reshape for sklearn (needs 2d array)
            X = clean_series.values.reshape(-1, 1)
            
            # apply isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # create boolean mask (isolationforest returns -1 for outliers, 1 for inliers)
            outlier_mask_clean = outlier_labels == -1
            
            # map back to original series index
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask.loc[clean_series.index] = outlier_mask_clean
            
            return outlier_mask
        else:
            # for non-numeric, no outliers detected in auto mode
            return pd.Series(False, index=series.index)
            
    elif isinstance(normal_values, list):
        # explicit list of acceptable values - anything not in list is outlier
        # return the index of all values which are NOT in the normal values
        return ~series.isin(normal_values)
        
    elif isinstance(normal_values, dict):
        # range-based detection (expecting keys like 'min', 'max')
        outlier_mask = pd.Series(False, index=series.index)

        # check for minimum value in normal_values and mark as outlier if less than min
        if 'min' in normal_values:
            outlier_mask |= series < normal_values['min']
        # check for maximum value in normal_values and mark as outlier if greater than max
        if 'max' in normal_values:
            outlier_mask |= series > normal_values['max']
            
        return outlier_mask
        
    elif isinstance(normal_values, str):
        # assume it's a condition string that can be evaluated
        # this is more advanced and potentially unsafe - handle with care
        try:
            # create a safe namespace for evaluation
            namespace = {'series': series, 'pd': pd, 'np': np}
            # evaluate the condition - should return a boolean mask
            normal_mask = eval(normal_values, {"__builtins__": {}}, namespace)
            return ~normal_mask  # invert to get outlier mask
        except:
            # if evaluation fails, assume no outliers
            return pd.Series(False, index=series.index)
    
    # default: no outliers detected
    return pd.Series(False, index=series.index)