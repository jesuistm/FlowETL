from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.pydantic_models import Plan, DataAnalystRequest, DataEngineerRequest, Analysis
import pandas as pd
import json
import numpy as np
from backend.functions import *
from backend.prompts import *
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
llm = ChatOpenAI(api_key=key,  model="gpt-4.1",  temperature=0.0)

# configure API middleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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

    logging.info("Received complete abstraction")

    # take the min between 10% sample of the abstraction or 25 rows - this makes processing quicker
    # we assume that the plan generated will be successfully applied to the entire dataset
    sample_size = min(25, int(len(abstraction) * 0.1)) 
    sampled_abstraction = abstraction.sample(n=sample_size).to_json()

    # extract the source dataset name and task description from request payload
    source_dataset = request.source_dataset
    task_description = request.task_description

    # invoke the plan construction chain
    result = data_engineering_chain.invoke({ 
      "task_description": task_description, 
      "documentation" : flowetl_documentation, 
      "source_dataset" : source_dataset,
      "dataset" : sampled_abstraction
    })

    logging.info("Successfully invoked the data engineering chain")

    # extract the data engineering pipeline and the dataset schema from the generated plan
    pipeline = result['pipeline']
    features_schema = result['source_schema']

    logging.info("Successfully extracted pipeline and schema from chain")
    logging.info(json.dumps(pipeline, indent=2))
    logging.info(json.dumps(features_schema, indent=2))

    # apply the pipeline onto the abstracted dataset
    for node in pipeline:

      # for the current node, extract all possible configuration attributes
      node_type = node.get('node_type', None)
      node_id = node.get('node_id', None)
      columns = node.get('columns', None)
      source = node.get('source', None)
      target = node.get('target', None)
      function = node.get('function', None)
      drop_source = node.get('drop_source', None)

      logging.info(f"Processing node with ID : {node_id}")

      # based on the node type, call one of the functions and configure it using the node's attributes
      if node_type == "MissingValues":
        abstraction = missing_values(columns=columns, abstraction=abstraction, features_schema=features_schema)

      if node_type == "Duplicates":
        abstraction = duplicate_instances(abstraction=abstraction)

      if node_type == "OutliersAndAnomalies":
        abstraction = outliers_anomalies(columns=columns, abstraction=abstraction, features_schema=features_schema)

      if node_type == "DeriveColumn":
        abstraction = derive_column(abstraction=abstraction, source=source, target=target, function=function, drop_source=drop_source)

      logging.info(f"Successfully applied the node, {len(abstraction.index)}")

    logging.info("Applied the pipeline successfully, sending the processed abstraction to frontend")

    return { "processed_abstraction" : abstraction.to_json(orient='records') } # serialise the df by converting it to json

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")


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
