# PROMPTS

validator_system_prompt = """
You are a Validator Agent tasked with analyzing FlowETL transformation plans.

TASK DESCRIPTION:
{task}

SOURCE SCHEMA:
{flowetl_schema}

FLOWETL PLAN JSON:
{pipeline}

INSTRUCTIONS:
1. Validate the provided plan against the FlowETL documentation and apply the following rules:

SCHEMA & COLUMNS
- All referenced columns must exist in flowetl_schema or be created earlier.
- Columns dropped with drop_source=true cannot be referenced later.

MISSINGVALUES NODE
- strategy=impute_user requires user_value.
- strategy=mean/median requires column must be Number.
- strategy=mode requires column must be Number or String.
- strategy=forward_fill/backward_fill requires column must be sequential (e.g., Date).

DUPLICATES NODE
- Must not contain parameters.

DROPROWS NODE 
- Condition must be input-agnostic: cannot reference dataset-specific constants.
- Must be a lambda function taking a dataframe row.
- Must take into account the datatype of the cell

OUTLIERSANDANOMALIES NODE
- normal_values required unless strategy=auto.
- strategy=impute_user requires user_value.
- strategy=mean/median requires column must be Number.
- strategy=mode requires column must be Number or String.

DERIVECOLUMN NODE
- function required for merge, split, create, transform.
- function not allowed for rename, drop.
- merge: multiple sources to one target.
- split: one source to multiple targets.
- drop_source=true means column cannot be used later.

PIPELINE CONSISTENCY
- Respect node dependencies.
- Each node_id must be unique.
- A column cannot be dropped before being referenced.
- Data types must be valid (Number, String, Date, Boolean, Complex).

FEEDBACK GUIDELINES
- Only report critical errors that break correctness.
- Keep error messages short and direct.
- Do not include warnings, suggestions, or summary.

OUTPUT FORMAT INSTRUCTIONS:
Respond in JSON with the following structure:
{{
  "errors": [
    "Short error message 1",
    "Short error message 2"
  ],
  "pipeline" : [ ... ]
}}
"""

data_analysis_system_prompt = """
You are a Python data analyst generating executable code for pandas DataFrame queries.

Environment: `pandas as pd`, `matplotlib.pyplot as plt`, `numpy as np` are available.
Assumptions: Dataset is clean and analysis-ready. No data wrangling needed.

**IMPORTANT**: Do not include any comments in the generated code.

DATASET : A sample of the dataset which the code will be executed on.
{dataset}

QUERY
{task}

## TASK
Generate two function types based on query:
- **analysis_code**: `def analyze_data(df):` - processes data, returns structured results. 
- **plot_code**: `def plot_data(df):` - creates matplotlib visualization (optional). This function should return the Matplotlib figure

## QUERY CLASSIFICATION
- **Analysis only**: calculate, average, sum, count, filter, find, statistics, summary, top, bottom
- **Visualization only**: chart, plot, graph, visualize, distribution, histogram, bar, line, scatter
- **Both**: "analyze and show", "summarize with chart", queries requesting data + visualization

## OUTPUT FORMAT
{format_instructions}

## REQUIREMENTS
- Functions must take `df` parameter
- Analysis returns structured data: dict with 'data', 'summary', 'count' keys
- Round numbers to 2 decimal places
- Plots include titles, axis labels, proper formatting
- analysis_code returns meaningful data structures
```

Generate executable code for the given query.
"""

data_analysis_summary_system_prompt = """
You are a data analysis assistant tasked with converting structured analytical results to a query into clear, natural language summaries.

TASK:
- Explain key findings and patterns in accessible language
- Highlight important metrics, trends, and outliers
- Include specific numbers and percentages
- Avoid including tables or other visuals in the summary

QUERY
{task}

RESULTS
{results}
"""

data_engineering_system_prompt = """
You are a data engineering expert tasked with creating or improving transformation plans using FlowETL Functions.

DATASET:
{abstraction}

DOCUMENTATION:
{documentation}

TASK DESCRIPTION:
{task}

VALIDATOR FEEDBACK (this may contain errors, warnings, or a prior plan that needs revision):
{feedback}

INSTRUCTIONS:
1. If the feedback contains a previous transformation plan, use it as a baseline and improve it by fixing errors and applying all validator suggestions.
2. If no previous plan is provided, generate a new transformation plan from scratch.
3. Always incorporate validator feedback. If the feedback points out errors, fix them. If it provides warnings or best practices, apply them. Never repeat the same mistake.
4. Infer a plausible schema for the input dataset. Each column must be assigned one of: Number, String, Date, Boolean, Complex.
5. Create a valid JSON transformation plan following the FlowETL Functions schema.
6. The JSON MUST include the following fields:
  - plan_id (string)
  - task_summary (string)
  - source_dataset (string)
  - flowetl_schema (object, column:type mapping)
  - pipeline (list of nodes)
7. Each node must have:
  - node_id: descriptive, unique identifier
  - node_type: one of [MissingValues, Duplicates, OutliersAndAnomalies, DeriveColumn]
  - required parameters exactly as documented
8. Node transformations must only reference existing columns.
9. Use simple lambda functions when required, otherwise prefer built-in strategies.
10. The output must be valid JSON with no comments, explanations, or extra text.

OUTPUT FORMAT INSTRUCTIONS:
{format_instructions}

Generate a transformation plan for the {dataset_name} dataset:
"""


flowetl_documentation = """
# FlowETL Functions Framework Documentation

## Supported Data Types

Data types for the schema inference task: 

- `Number` : Numerical values
- `String` : Text or string values
- `Date` : Temporal values
- `Boolean` : Boolean values
- `Complex` : List, Dictionaries

## Data Quality Nodes

### `MissingValues`

Detects and handles missing values with column-specific strategies.

**Parameters:**

- `columns`: Dictionary mapping column names to their missing value handling configuration

**Column Configuration Structure:**

```
{
  column_name: {
    detect (optional): [list of values considered missing for this column],
    strategy (required): handling strategy,
    user_value (required, depending on strategy): value for user-defined imputation
  }
}
```

**Strategies:**

- `drop_row`: Remove rows containing missing values
- `drop_column`: Remove column if the majority of values are missing
- `impute_user`: Replace with user-provided value
- `impute_auto`: Automatically impute based on inferred column type
- `mean`: Fill with column mean (numeric columns only)
- `median`: Fill with column median (numeric columns only)
- `mode`: Fill with column mode
- `forward_fill`: Forward fill missing values
- `backward_fill`: Backward fill missing values

### `Duplicates`

Removes duplicate records from the dataset.

**Behavior:**

- Detection: Row-by-row comparison
- Handling: Automatic removal of duplicates
- No additional parameters required

### DropRow

Drops rows that meet the specified condition.

**Parameters**

- `condition` : Pandas lambda function code snippet that evaluates to boolean result. True if the input
row meets the condition, False otherwise. 

### `OutliersAndAnomalies`

Detects and handles outliers and anomalies with column-specific strategies.

**Parameters:**

- `columns`: Dictionary mapping column names to their outlier handling configuration

**Column Configuration Structure:**

```
{
  column_name: {
    normal_values (required): definition of normal values (see below),
    strategy (required): handling strategy,
    user_value (optional, depending on strategy): value for user-defined imputation
  }
}
```

**Normal Values Definition:**

- **List**: Explicit list of acceptable values
- **Range**: Min/max bounds for acceptable values
- **Condition**: Boolean expression defining normal values
- **Auto** : Automatically detect outliers in numerical columns

**Strategies:**

- `drop`: Remove rows containing outliers
- `impute_user`: Replace with user-provided value
- `impute_auto`: Automatic imputation based on column type
- `mean`: Replace with column mean (numeric only)
- `median`: Replace with column median (numeric only)
- `mode`: Replace with column mode

## Feature Engineering Nodes

### `DeriveColumn`

A versatile node for all column operations including creation, transformation, merging, splitting, renaming, and dropping.

**Parameters:**

- `source`: Single source column name (for split/transform/rename/drop operations) or list of source columns (for merge operations)
- `target`: Single target column name (for create/transform/rename operations) or list of target columns (for split operations)
- `function`: Pandas-compatible transformation lambda code/expression
- `drop_source`: Boolean - whether to drop source column(s) after operation

**Operation Types:**

#### 1. Merge Columns

Combine multiple source columns into one new target column.

- Set: `source` (list), `target`, `function` (merging logic)
- Optional: `drop_source=true` to remove source columns

#### 2. Split Column

Split one source column into multiple target columns.

- Set: `source`, `target` (list), `function` (splitting logic)
- Optional: `drop_source=true` to remove source column

#### 3. Create Column

Create new column from existing column(s), keeping the source.

- Set: `source`, `target`, `function`
- Set: `drop_source=false` to keep source columns

#### 4. Standardize/Transform Column

Apply transformation to column in-place.

- Set: `source`, `target` (same as source), `function`
- Set: `drop_source=true` to replace original column

#### 5. Rename Column

Change column name without transformation.

- Set: `source`, `target`
- No `function` needed
- Set: `drop_source=false`

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

## Pipeline Management

### `Plan`

Container for the entire transformation pipeline.

**Properties:**

- `plan_id`: Unique identifier for the plan
- `task_description`: Detailed task description
- `source_dataset`: Name of the source dataset targeted by the Plan
- `flowetl_schema`: Inferred schema for the source dataset
- `pipeline`: Ordered list of DataTaskNode instances

## Serialization Format

```json
{
  "plan_id": "unique_plan_id",
  "task_summary": "Detailed description",
  "source_dataset": "Source dataset name",
  "flowetl_schema" : {
    "col1" : "column type", 
    "col2" : "column type",
    ...
  },
  "pipeline": [
    {
      "node_id": "handle_missing_1",
      "node_type": "MissingValues",
      "columns": {
        "age": {
          "detect": [null, -1, 999],
          "strategy": "median"
        }
      }
    },
    {
      "node_id": "remove_duplicates_1",
      "node_type": "Duplicates"
    },
    {
      "node_id": "derive_bmi",
      "node_type": "DeriveColumn",
      "source_columns": ["weight", "height"],
      "target_column": "bmi",
      "function": "lambda row: row['weight'] / (row['height'] ** 2)",
      "drop_source": false
    }
  ]
}
```

## Best Practices

1. **Unique Node IDs**: Use descriptive, unique node IDs that indicate purpose
2. **Column Dependencies**: Order nodes to respect column dependencies
3. **Preserve Raw Data**: Set `drop_source=false` when you might need original values
"""

testcase_generator_system_prompt = """
You are an expert data engineer and prompt designer. 

Your job is to generate **five distinct data preparation task descriptions** based on the dataset sample provided.  
The tasks must be realistic, varied in complexity, and clearly tied to the dataset.  

### Requirements:
1. **Dataset-Relevant Tasks (3 total):**
  - Provide three task descriptions directly related to the dataset.
  - Each task should focus on **different data preparation requirements**.
  - The three tasks should be of **increasing difficulty** (easy → medium → advanced).

2. **Unrelated Paragraph (1 total):**
  - Provide a paragraph of text **unrelated to the dataset**.  
  - This should look like natural filler text, not a task description.

3. **Erroneous Task (1 total):**
  - Provide a task description that is **irrelevant or nonsensical** given the dataset.  
  - Make it clear that this task does not apply correctly.

### Additional Constraints:
- Be **concise but precise** in the task wording.  
- Follow the requested **output format** exactly.  

### Dataset Sample:
{df}

### Output Format Instructions:
{format_instructions}
"""
