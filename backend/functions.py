import numpy as np
import pandas as pd
from typing import Any, Dict

def missing_values(columns : Dict[str, Any], abstraction : pd.DataFrame, features_schema : Dict[str, str]) -> pd.DataFrame:
    """
    Handle missing values following the configuration handles within the plan
    
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
        inferred_type = features_schema.get(f'{column}', 'Complex')

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