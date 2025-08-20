from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.pydantic_models import Plan, FrontEndRequest
import pandas as pd
import json
from typing import Dict, Any
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# set logging to DEBUG mode
logging.basicConfig(level=logging.INFO)

# load and extract environment variable
load_dotenv()
key = os.environ["OPENAI_API_KEY"]

app = FastAPI()

# configure API middleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# FlowETL functions documentation used to inform the plan generation process
flowetl_documentation = """
# FlowETL Functions Framework Documentation

## Data Quality Nodes

### `MissingValues`

Detects and handles missing values with column-specific strategies.

**Parameters:**

- `columns`: Dictionary mapping column names to their missing value handling configuration

**Column Configuration Structure:**

```
{
  column_name: {
    detect: [list of values considered missing for this column],
    strategy: handling strategy,
    user_value: value for user-defined imputation
  }
}
```

**Strategies:**

- `drop_row`: Remove rows containing missing values
- `drop_column`: Remove column if threshold percentage of values are missing
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
    normal_values: definition of normal values (see below),
    strategy: handling strategy,
    user_value: value for user-defined imputation
  }
}
```

**Normal Values Definition:**

- **List**: Explicit list of acceptable values
- **Range**: Min/max bounds for acceptable values
- **Condition**: Boolean expression defining normal values
- **Auto**: If not specified, uses Isolation Forest for numeric columns

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
- `function`: Pandas-compatible transformation code/expression
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
  "source_schema" : the inferred schema for the source dataset
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

# prompt include both the FlowETL Functions documentation and the task instruction for the LLM
system_prompt_template = """
You are a data engineering expert tasked with creating transformation plans using the available FlowETL Functions.

DOCUMENTATION:
{documentation}

TASK DESCRIPTION:
{task_description}

INSTRUCTIONS:
1. Analyze the task description and knowledge base to understand the data transformation requirements
2. Create a complete JSON transformation plan using FlowETL Functions
3. Follow the node ordering rules from the documentation
4. Use descriptive node IDs that clearly indicate their purpose
5. Include all required fields in the JSON output
6. Ensure the JSON is valid and follows the exact schema shown in the documentation

{format_instructions}

Generate a transformation plan for the {source_dataset} dataset:
"""

# setup OpenAI client
llm = ChatOpenAI(api_key=key,  model="gpt-5",  temperature=0.0)

# setup output parser to follow pydantic schema 
output_parser = JsonOutputParser(pydantic_object=Plan)

# define prompt structure
prompt_template = PromptTemplate( 
    template=system_prompt_template, 
    input_variables=["task_description", "documentation"], 
    partial_variables={"format_instructions": output_parser.get_format_instructions()} 
) 

# define LangChain chain to assemble the prompt, call the llm, and parse the output
chain = prompt_template | llm | output_parser

@app.post("/")
async def process_abstraction(request: FrontEndRequest) -> Dict[str, Any]:
    try:
        # reconstruct DataFrame from JSON
        abstraction = pd.DataFrame(json.loads(request.abstraction))

        # extract the source dataset name 
        source_dataset = request.source_dataset

        # extract task description from request payload
        task_description = request.task_description

        # invoke the plan construction chain
        result = chain.invoke({ "task_description": task_description, "documentation" : flowetl_documentation, "source_dataset" : source_dataset })

        return { "plan" : result }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")