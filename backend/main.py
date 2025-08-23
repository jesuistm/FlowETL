from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.pydantic_models import Plan, DataAnalystRequest, DataEngineerRequest, Analysis
import pandas as pd
import json
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# set logging to DEBUG mode
logging.basicConfig(level=logging.INFO)

# load and extract environment variable
load_dotenv()
key = os.environ["OPENAI_API_KEY"]

app = FastAPI()

# setup OpenAI client
llm = ChatOpenAI(api_key=key,  model="gpt-5",  temperature=0.0)

# configure API middleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# FlowETL functions documentation used to inform the plan generation process
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

## Pipeline Management

### `Plan`

Container for the entire transformation pipeline.

**Properties:**

- `plan_id`: Unique identifier for the plan
- `task_description`: Detailed task description
- `source_dataset`: Name of the source dataset targeted by the Plan
- `source_schema`: Inferred schema for the source dataset
- `pipeline`: Ordered list of DataTaskNode instances

## Serialization Format

```json
{
  "plan_id": "unique_plan_id",
  "task_summary": "Detailed description",
  "source_dataset": "Source dataset name",
  "source_schema" : {
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

# prompt include both the FlowETL Functions documentation and the task instruction for the data engineering LLM
data_engineering_system_prompt = """
You are a data engineering expert tasked with creating transformation plans using the available FlowETL Functions.

DATASET:
{dataset}

DOCUMENTATION:
{documentation}

TASK DESCRIPTION:
{task_description}

INSTRUCTIONS:
1. Analyze the task description and knowledge base to understand the data transformation requirements
2. Using the FlowETL supported data types, infer a plausible schema for the input dataset
3. Create a complete JSON transformation plan using FlowETL Functions
4. Follow the node ordering rules from the documentation
5. Use descriptive node IDs that clearly indicate their purpose
6. Include all required fields in the JSON output
7. Ensure the JSON is valid and follows the exact schema shown in the documentation

OUTPUT FORMAT INSTRUCTIONS
{format_instructions}

Generate a transformation plan for the {source_dataset} dataset:
"""

# setup data engineering output parser to follow pydantic schema 
data_engineering_output_parser = JsonOutputParser(pydantic_object=Plan)

# define data engineering prompt builder
data_engineering_prompt_builder = PromptTemplate( 
  template=data_engineering_system_prompt, 
  input_variables=["task_description", "documentation", "dataset", "source_dataset"], 
  partial_variables={"format_instructions": data_engineering_output_parser.get_format_instructions()} 
) 

# define data engineering LangChain chain
data_engineering_chain = data_engineering_prompt_builder | llm | data_engineering_output_parser

@app.post("/transform")
async def transform_data(request: DataEngineerRequest) -> Dict[str, Any]:
  try:
    # reconstruct DataFrame from JSON
    abstraction = pd.DataFrame(json.loads(request.abstraction))

    # extract the source dataset name 
    source_dataset = request.source_dataset

    # extract task description from request payload
    task_description = request.task_description

    # invoke the plan construction chain
    result = data_engineering_chain.invoke({ 
      "task_description": task_description, 
      "documentation" : flowetl_documentation, 
      "source_dataset" : source_dataset,
      "dataset" : abstraction
    })

    return { "plan" : result }

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")


# define intial prompt template to obtain both the analysis and plotting code snippets for the query
data_analysis_system_prompt = """
You are a Python data analyst generating executable code for pandas DataFrame queries.

Environment: `pandas as pd`, `matplotlib.pyplot as plt`, `numpy as np` are available.
Assumptions: Dataset is clean and analysis-ready. No data wrangling needed.

**IMPORTANT**: Do not include any comments in the generated code.

DATASET
{dataset}

QUERY
{query}

## TASK
Generate two function types based on query:
- **analysis_code**: `def analyze_data(df):` - processes data, returns structured results
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
{query}

RESULTS
{results}
"""

# define the output parser for the data analysis task
data_analysis_output_parser = JsonOutputParser(pydantic_object=Analysis)

# define the prompt builder for the data analysis task
data_analysis_prompt_builder = PromptTemplate( 
  template=data_analysis_system_prompt, 
  input_variables=["dataset", "query"], 
  partial_variables={"format_instructions": data_analysis_output_parser.get_format_instructions()} 
) 

# define the data analysis results summary prompt builder
data_analysis_summary_prompt_builder = PromptTemplate( 
  template=data_analysis_summary_system_prompt, 
  input_variables=["results", "query"], 
) 

# define the data analysis Langchain chain
data_analysis_chain = data_analysis_prompt_builder | llm | data_analysis_output_parser

# define the query result summarise Langchain chain
summariser_chain = data_analysis_summary_prompt_builder | llm | StrOutputParser()

@app.post("/analyze")
def analyze_data(request : DataAnalystRequest) -> Dict[str, Any]:
  try:
    # extract the abstracted dataset sample and the user query
    abstraction = pd.DataFrame(json.loads(request.abstraction))
    query = request.query

    # invoke the plan construction chain
    result = data_analysis_chain.invoke({ "query" : query, "dataset" : abstraction })
    
    analysis_code = result["analysis_code"]
    plot_code = result.get("plot_code", None)

    # extract the code analysis function
    namespace = {}
    exec(analysis_code, namespace)
    runnable_function = namespace['analyze_data'] # called 'analyze_data' as per the llm prompt

    # execute the function to obtain the analysis results
    data_analysis_result = runnable_function(abstraction)

    # run the summary creation chain to convert the results of the data analysis to natural language
    summary = summariser_chain.invoke({ "query" : query, "results" : data_analysis_result })

    # return the plotting function and the natural language summary to the frontend
    return { "plot_code" : plot_code, "summary" : summary }

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")
