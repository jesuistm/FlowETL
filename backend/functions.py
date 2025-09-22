import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Tuple
from sklearn.ensemble import IsolationForest
import logging 
import json

def compute_missing_values_count(abstraction: pd.DataFrame, columns_config: Dict[str, Any], logger) -> Dict[str, float]:
    """
    Compute and return the count of missing values for each column specified in the `columns_config` dictionary. 
    Result follows a format like { column name : count of missing values }
    """
    logger_prefix = "Util [compute missing values count]"
    logger.info(f"{logger_prefix} - Entered function")

    artifact = {
        column : sum(abstraction[column].isna() | abstraction[column].isin(config.get('detect', [np.nan, ""]))) 
        for column, config in columns_config.items()
    }

    logger.info(f"{logger_prefix} - Relevant artifact(s) : {json.dumps(artifact, indent=2)}. Exiting function")
    return artifact


def compute_outlier_values_count(abstraction: pd.DataFrame, flowetl_schema: Dict[str, str], columns_config: Dict[str, Any], logger) -> Dict[str, float]:
    """
    Compute and return the count of outliers for each column specified in the `columns_config` dictionary
    Result follows a format like { column name : count of outlier values }
    """

    logger_prefix = "Util [compute outlier values count]"
    logger.info(f"{logger_prefix} - Entered function")

    artifact = { 
        column : sum(_detect_outliers(abstraction[column], config.get('normal_values', 'auto'), flowetl_schema.get(column, 'Complex'), logger)) 
        for column, config in columns_config.items()
    }

    logger.info(f"{logger_prefix} - Relevant artifact(s) : {json.dumps(artifact, indent=2)}. Exiting function")
    return artifact

    
def compute_data_quality(abstraction: pd.DataFrame, flowetl_schema: Dict[str, str],pipeline: List[Any], logger) -> Dict[str, str]:
    """
    Compute a data quality report for the input abstraction. The report structure is as follows: 

    Report contents:
    - Dataset-wide information : dimensions, % of missing values, % of duplicate entries, % of outliers/anomalies 
    - Column-specific information : flowetl type, % of missing values, % of outliers/anomalies
    """

    logger_prefix = "Util [compute data quality]"
    logger.info(f"{logger_prefix} - Entered function")

    report = { 
        'dimensions' : [abstraction.shape[0], abstraction.shape[1]], # number of rows, number of columns 
        'missing_values_percent' : 0,
        'duplicate_entries_percent' : round((abstraction.duplicated().sum() / len(abstraction)) * 100,2),
        'outlier_values_percent' : 0,
        'column_specific' : {} 
    }

    # number of cells missing/outliers across the entire dataframe
    missing_values_count = 0
    outlier_values_count = 0

    # compute column-specific data quality information
    for node in pipeline:

        # extract configuration
        node_type = node.get('node_type', None)
        node_id = node.get('id', None)
        columns = node.get('columns', None)

        logger.info(f"{logger_prefix} - Computing data quality information for node with id {node_id} and type {node_type}")

        if node_type == "MissingValues":
            # need to return an object like { col1 : count of missing values, col2 : count of missing values, etc } 
            # to compute the total number of missing cells, simply sum all the values in the response
            result = compute_missing_values_count(abstraction, columns, logger)
            missing_values_count = sum(list(result.values()))

            # convert each count in the results.values to a percentage of the number of rows in the abstraction
            result = { key : round((value/abstraction.shape[0]) * 100,2) for key, value in result.items() }
            report['column_specific']['missing'] = result                                                               

        if node_type == "OutliersAndAnomalies":
            # similarly, here we must return an object like { col : count of outliers }
            # to compute thr total number of missing cells, sum the values in the response
            result = compute_outlier_values_count(abstraction, flowetl_schema, columns, logger)
            outlier_values_count = sum(list(result.values()))

            # convert each count in the results.values to a percentage of the number of rows in the abstraction
            result = { key : (value/abstraction.shape[0]) * 100 for key, value in result.items() }

            report['column_specific']['outliers'] = result                                                              

    # compute the percentage of dataframe cells considered missing
    report['missing_values_percent'] = round((missing_values_count/abstraction.size) * 100, 2)

    # compute the percentage of outlier values in the dataframe
    report['outlier_values_percent'] = round((outlier_values_count/abstraction.size) * 100, 2)

    logger.info(f"{logger_prefix} - Relevant artifact(s) : {json.dumps(report, indent=2)}. Exiting function")
    return report


def apply_pipeline(abstraction: pd.DataFrame, flowetl_schema: Dict[str, str],pipeline: List[Any], logger) -> Tuple[pd.DataFrame, List[Exception]]:
    """
    Apply the pipeline of flowetl functions to the input abstraction.
    Collect any exceptions and return them alongside the result.
    """

    logger_prefix = "Util [apply pipeline]"
    logger.info(f"{logger_prefix} - Entered function")

    exceptions: List[Exception] = []

    for node in pipeline:
        try:
            # extract configuration
            node_type = node.get('node_type', None)
            node_id = node.get('node_id', None)
            columns = node.get('columns', None)
            source = node.get('source', None)
            target = node.get('target', None)
            function = node.get('function', None)
            drop_source = node.get('drop_source', None)
            condition = node.get('condition', None)

            logger.info(f"{logger_prefix} - Processing node with id : {node_id}")

            # route to appropriate function
            if node_type == "MissingValues":
                abstraction = missing_values(columns=columns, abstraction=abstraction, features_schema=flowetl_schema, logger=logger)

            elif node_type == "Duplicates":
                abstraction = duplicate_instances(abstraction=abstraction, logger=logger)

            elif node_type == "OutliersAndAnomalies":
                abstraction = outliers_anomalies(columns=columns, abstraction=abstraction, features_schema=flowetl_schema, logger=logger)

            elif node_type == "DeriveColumn":
                abstraction = derive_column(
                    abstraction=abstraction,
                    source=source,
                    target=target,
                    function=function,
                    drop_source=drop_source,
                    logger=logger
                )

            elif node_type == "DropRow":
                abstraction = drop_rows(abstraction=abstraction, condition=condition, logger=logger)

            logger.info(f"{logger_prefix} - Successfully applied node with id {node_id}")

        except Exception as e:
            logger.error(f"{logger_prefix} - Error in node {node_id}. Exception collected within method. Exception details: {e}")
            exceptions.append(e)
            # continue to next node instead of failing

    logger.info(f"{logger_prefix} - Pipeline applied to abstraction. Exiting function")
    return abstraction, exceptions
    

def drop_rows(abstraction: pd.DataFrame, condition: str, logger) -> pd.DataFrame:
    """
    Drop all rows within the abstraction that meet the input condition

    **Parameters**
    - `abstraction` : Pandas dataframe to be processed by this function
    - `condition` : Pandas function which takes in a dataframe's row and returns a boolean value
    """

    logging_prefix = "Util [drop rows]"
    logger.info(f"{logging_prefix} - Entered function")

    abstraction = abstraction.copy()
    
    try:
        logger.info(f"{logging_prefix} - Converting artifact 'drop row mask' into runnable function")
        func = eval(condition) 
        # compute boolean mask over the abstraction
        logger.info(f"{logging_prefix} - Conversion succesful. Computing boolean mask over whole dataset")
        mask = func(abstraction) # booolean mask
    except Exception as e:
      logger.error(f"{logging_prefix} - Error occured while converting artifact 'drop row mask' into runnable function")
      raise Exception("Error occured while converting artifact 'drop row mask' into runnable function")

    # drop rows where the mask is True
    processed_abstraction = abstraction[~mask]
    logger.info(f"{logging_prefix} - Removed rows from dataset using boolean mask.Exiting function")

    return processed_abstraction


def duplicate_instances(abstraction : pd.DataFrame, logger) -> pd.DataFrame:
    """
    Removes duplicated rows from the abstraction, leveraging Pandas' built-in `drop_duplicates` method
    **Parameters**
    - `abstraction` : Pandas dataframe to be processed by this function
    """
    
    logging_prefix = "Util [drop duplicated instances]"
    logger.info(f"{logging_prefix} - Entered function. Removing all duplicated rows from dataset")
    processed_abstraction = abstraction.drop_duplicates()
    logger.info(f"{logging_prefix} - Removal succesfull. Exiting function")
    return processed_abstraction

    
def missing_values(columns : Dict[str, Any], abstraction : pd.DataFrame, features_schema : Dict[str, str], logger) -> pd.DataFrame:
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

    logging_prefix = "Util [missing values handler]"
    logger.info(f"{logging_prefix} - Entering function")

    # create a copy to avoid modifying the original dataframe
    abstraction = abstraction.copy()
    
    # Keep track of columns to drop to avoid modifying dictionary during iteration
    columns_to_drop = []
    
    for column, config in columns.items():
        
        # skip if column was already dropped
        if column not in abstraction.columns:
            logger.warning(f"{logging_prefix} - Column '{column}' not in dataset, skipped.")
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

        logger.info(f"{logging_prefix} - Processing column '{column}' using strategy '{strategy}' and inferred column type of '{inferred_type}'")

        # create mask for missing values in this column
        # we use both isna and isin to handle the problem where "np.nan != np.nan" in Pandas
        missing_mask = abstraction[column].isna() | abstraction[column].isin(detectables)
        logger.info(f"{logging_prefix} - Compute missing values mask using user-specified detecable missing values")

        # handle the missing value based on the strategy
        if strategy == 'drop_row':
            abstraction = abstraction.drop(abstraction[missing_mask].index)
            logger.info(f"{logging_prefix} - Dropped {missing_mask.sum()} rows from dataset")

        # check whether the current column is composed mainly of missing values, defined in the 'detectables' list
        elif strategy == 'drop_column' and ((abstraction[column].isna() | abstraction[column].isin(detectables)).sum() > len(abstraction[column]) / 2):
            # mark column for removal
            columns_to_drop.append(column)
            logger.info(f"{logging_prefix} - Marked column '{column}' for removal due to 50%+ missing values")

        elif strategy == 'impute_user':
            abstraction.loc[missing_mask, column] = user_value
            logger.info(f"{logging_prefix} - Imputed column '{column}' using user-value '{user_value}")

        elif strategy == 'impute_auto':
            # automatically impute missing values based on the column type inferred by flowetl
            if inferred_type == 'Number':
                abstraction.loc[missing_mask, column] = 0
                logger.info(f"{logging_prefix} - Auto-imputed '{column}' with 0")
                continue

            elif inferred_type == 'String':
                abstraction.loc[missing_mask, column] = "N/A"
                logger.info(f"{logging_prefix} - Auto-imputed '{column}' with 'N/A'")
                continue

            elif inferred_type == 'Date':
                abstraction.loc[missing_mask, column] = "1/1/2000"
                logger.info(f"{logging_prefix} - Auto-imputed '{column}' with '1/1/2000'")
                continue

            elif inferred_type == 'Boolean':
                if not abstraction[column].mode().empty:
                    mode_value = abstraction[column].mode()[0]
                    abstraction.loc[missing_mask, column] = mode_value
                    logger.info(f"{logging_prefix} - Auto-imputed '{column}' with mode '{mode_value}'")
                continue

            elif inferred_type == 'Complex':
                abstraction.loc[missing_mask, column] = "MISSINGVALUE"
                logger.info(f"{logging_prefix} - Auto-imputed '{column}' with placeholder 'MISSINGVALUE'")
                continue

        elif strategy == 'mean':
            mean_value = abstraction[column].mean()
            abstraction.loc[missing_mask, column] = mean_value
            logger.info(f"{logging_prefix} - Imputed '{column}' with mean value of {mean_value}")

        elif strategy == 'median':
            median_value = abstraction[column].median()
            abstraction.loc[missing_mask, column] = median_value
            logger.info(f"{logging_prefix} - Imputed '{column}' with median value of {median_value}")

        elif strategy == 'mode':
            if not abstraction[column].mode().empty:
                mode_value = abstraction[column].mode()[0]
                abstraction.loc[missing_mask, column] = mode_value
                logger.info(f"{logging_prefix} - Imputed '{column}' with mode value of {mode_value}")

        elif strategy == 'forward_fill':
            abstraction[column] = abstraction[column].ffill()
            logger.info(f"{logging_prefix} - Applied forward fill on '{column}'")

        elif strategy == 'backward_fill':
            abstraction[column] = abstraction[column].bfill()
            logger.info(f"{logging_prefix} - Applied backward fill on '{column}'")

    # drop columns that were marked for removal
    if columns_to_drop:
        abstraction = abstraction.drop(columns=columns_to_drop)
        logger.info(f"{logging_prefix} - Dropped columns '{json.dumps(columns_to_drop)} from dataset")

    logger.info(f"{logging_prefix} - Exiting function")
    return abstraction


def outliers_anomalies(columns : Dict[str, Any], abstraction : pd.DataFrame, features_schema : Dict[str, str], logger) -> pd.DataFrame:
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

    logging_prefix = "Util [outlier and anomaly values handler]"
    logger.info(f"{logging_prefix} - Entering function")

    # create a copy to avoid modifying the original dataframe
    abstraction = abstraction.copy()

    # collect all rows to drop (for columns using 'drop' strategy)
    rows_to_drop = set() # this stores the indices

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

        logger.info(f"{logging_prefix} - Handing outliers over column '{column}' of type '{inferred_type}' using strategy '{strategy}' and normal values '{json.dumps(normal_values)}'")

        # create mask for outliers/anomalies in this column
        outlier_mask = _detect_outliers(abstraction[column], normal_values, inferred_type)
        logger.info(f"{logging_prefix} - Computed outlier mask over column '{column}' using normal values '{json.dumps(normal_values)}'")

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
        abstraction = abstraction.drop(list(rows_to_drop))
        logging.info(f"{logging_prefix} - Dropped all rows containing an outlier")
    
    logger.info(f"{logging_prefix} - Exiting function")
    return abstraction


def derive_column(abstraction : pd.DataFrame, source : Union[str, List], target : Union[str, List], function : str, drop_source : bool, logger) -> pd.DataFrame:
    """
    Handle all column operations including creation, transformation, merging, splitting, renaming, and dropping.

    **Parameters**
    - `abstraction` : Pandas dataframe to be processed by this function
    - `source`: Single source column name (for split/transform/rename/drop operations) or list of source columns (for merge operations)
    - `target`: Single target column name (for create/transform/rename operations) or list of target columns (for split operations)
    - `function`: Pandas-compatible transformation lambda code/expression
    - `drop_source`: Boolean - whether to drop source column(s) after operation

    **Operation Types:**

    #### 1. Merge Columns
    Combine multiple source columns into one new target column.
    - Set: `source` (list), `target`, `function` (merging logic)
    - Set: `drop_source=true` to remove source columns
    
    #### 2. Split Column
    Split one source column into multiple target columns.
    - Set: `source`, `target` (list), `function` (splitting logic)
    - Set: `drop_source=true` to remove source column

    #### 3. Create Column
    Create new column from existing column(s), keeping the source.
    - Set: `source`, `target`, `function`
    - Set: `drop_source=false` to keep source columns

    #### 4. Standardize/Transform Column
    Apply transformation to column in-place.
    - Set: `source`, `target` (same as source), `function`
    - Set: `drop_source=false` to apply transformation in place

    #### 5. Rename Column
    Change column name without transformation.
    - Set: `source`, `target`
    - No `function` needed
    - Set: `drop_source=false` to simply rename the column

    #### 6. Drop Column
    Remove column from dataset.
    - Set: `source`, `drop_source=true`
    - No `target` or `function` needed

    **Function Examples:**
    - Concatenation: `lambda row: row['col1'] + ' ' + row['col2']`
    - Mathematical: `lambda x: x * 100`
    - Conditional: `lambda x: 'High' if x > 100 else 'Low'`
    - Date extraction: `lambda x: pd.to_datetime(x).year`
    - Splitting: `lambda x: x.split(',')`
    """

    logging_prefix = "Util [derive column]"
    logger.info(f"{logging_prefix} - Entering function")
    
    # create a copy to avoid modifying the original dataframe
    abstraction = abstraction.copy()
    
    # operation 1: merge columns
    if isinstance(source, list) and isinstance(target, str):
        logging.info(f"{logging_prefix} - Performing merge operation. Source : {json.dumps(source)} -> Target : {target}")
        # apply function row-wise for merging
        abstraction[target] = abstraction.apply(eval(function), axis=1)

        if drop_source:
            # drop source columns if specified
            abstraction = abstraction.drop(columns=source)
            logging.info(f"{logging_prefix} - Source columns dropped")

        logging.info(f"{logging_prefix} - Merge operation complete.")
        return abstraction

    # operation 2 : split columns 
    elif isinstance(source, str) and isinstance(target, list):

        logging.info(f"{logging_prefix} - Performing split operation. Source : {source} -> Target : {json.dumps(target)}")

        # apply function and split result into multiple columns
        split_data = abstraction[source].apply(eval(function))
        
        # handle the split results
        if hasattr(split_data.iloc[0], '__iter__') and not isinstance(split_data.iloc[0], str):
            # if function returns iterable (like list from split)
            split_df = pd.DataFrame(split_data.tolist(), index=abstraction.index)

            # assign to target columns
            for i, col_name in enumerate(target):
                abstraction[col_name] = split_df.iloc[:, i] if i < split_df.shape[1] else None

        else:
            # if function doesn't return iterable, assign same value to all targets
            for col_name in target:
                abstraction[col_name] = split_data
    
        if drop_source:
            abstraction = abstraction.drop(columns=[source])
            logging.info(f"{logging_prefix} - Source columns dropped")
        
        logging.info(f"{logging_prefix} - Split operation complete")
        return abstraction

    # operation 5 : rename column 
    elif function is None and isinstance(source, str) and isinstance(target, str) and source != target:
        logging.info(f"{logging_prefix} - Performing rename operation. Source : {source} -> Target : {target}")
        abstraction = abstraction.rename(columns={source: target})
        logging.info(f"{logging_prefix} - Rename operation complete")
        return abstraction
    
    # operations 3 & 4 : create & transform columns 
    elif isinstance(source, str) and isinstance(target, str):
        logging.info(f"{logging_prefix} - Performing creation or transform operation. Source : {json.dumps(source)} -> Target : {json.dumps(target)}")

        # check if we're dealing with multiple source columns referenced in the function
        if '[' in str(function) and ']' in str(function):
            # we check if square brackets appear in lambda - meaning it must access multiple columns
            abstraction[target] = abstraction.apply(eval(function), axis=1)
        else:
            # column-wise operation
            abstraction[target] = abstraction[source].apply(eval(function))
        if drop_source and source != target:
            # drop source if different from target and drop_source is True
            abstraction = abstraction.drop(columns=[source])
            logging.info(f"{logging_prefix} - Source columns dropped")

        logging.info(f"{logging_prefix} - Create or Transform operation complete.")
        return abstraction
    
    # operation 6 : drop column 
    elif drop_source and (target is None and function is None):
        logging.info(f"{logging_prefix} - Performing drop column(s) operation. Source : {json.dumps(source)}")
        abstraction = abstraction.drop(columns=source)
        logging.info(f"{logging_prefix} - Drop column(s) operation complete.")
        return abstraction
    
    logger.info(f"{logging_prefix} - Exiting function")
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
        # NOTE : this is more advanced and potentially unsafe - handle with care. Potential risk of prompt/code injection attacks
        try:
            # create a safe namespace for evaluation
            namespace = {'series': series, 'pd': pd, 'np': np}
            # evaluate the condition - should return a boolean mask
            normal_mask = eval(normal_values, { "__builtins__": {} }, namespace)
            return ~normal_mask  # invert to get outlier mask
        except:
            # if evaluation fails, assume no outliers
            return pd.Series(False, index=series.index)
    
    # default: no outliers detected
    return pd.Series(False, index=series.index)