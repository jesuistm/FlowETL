import json
import logging
import uuid
from typing import Any, Dict, Literal

from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import END, START, StateGraph
from functions import *
from models import DataAnalystRequest, DataEngineerRequest, GraphState
from prompts import *
from chains_utils import *

MAX_RETRIES = 3

# console handler - write API logs to console
console_handler = logging.StreamHandler()
console_handler.setLevel("INFO")
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logger = logging.getLogger("flowetl_logger")
logger.setLevel(logging.INFO)

# prevent duplicate handlers in app reload
if not logger.hasHandlers():
  logger.addHandler(console_handler)

app = FastAPI()

# custom middleware for per-request logging. Each request gets its own log file
class RequestLoggingMiddleware(BaseHTTPMiddleware):

  async def dispatch(self, request: Request, call_next):
        
    # create unique log file for this request
    # generate request ID as timestamp down to seconds + first segment of a UUID (e.g., T2025-09-21-15-45-30_ID8b7f8c5f)
    # T_ refers to the request timestamp, ID refers to the request ID
    request_id = f"T{datetime.now():%Y-%m-%d-%H-%M-%S}_ID{str(uuid.uuid4()).split('-')[0]}"

    # create the logs folder if it doesnt exist
    LOGS_DIR = os.getenv("LOGS_DIR", "flowetl_logs")
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    logfile = f"{LOGS_DIR}/{request_id}.log"

    # store the request_id in request.state so endpoints can access it
    request.state.request_id = request_id

    # create the file handler to allow write to log file
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # attach file handler
    logger.addHandler(file_handler)

    logger.info(request.url.path)

    try:
      response = await call_next(request)
      response.headers["flowetl-request-id"] = request_id

    finally:
      # detach & close file handler after request
      logger.removeHandler(file_handler)
      file_handler.close()

    return response

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
  CORSMiddleware, 
  allow_origins=["*"], 
  allow_credentials=True, 
  allow_methods=["*"], 
  allow_headers=["*"]
)


def planner(state: GraphState) -> GraphState:

  logging_prefix = f"Planner [{state['iterations']}]"
  logger.info(f"{logging_prefix} - Entered Planner")
  logger.info(f"{logging_prefix} - Input pipeline : {json.dumps(state.get('pipeline', None), indent=2)}")

  # extract any feedback from the previous validation round
  feedback = state.get("errors", None)
  feedback_text = json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else None

  logger.info(f"{logging_prefix} : feedback - {'No feedback received from Validator' if not feedback else feedback_text}")
  next_state = dict(state)
  iteration = state.get("iterations", -1)

  # compute a plan at the start of the cycle or if there is any previous feedback
  if iteration == 1 or feedback:

    # extract artifacts required by the planning agent
    task = state.get("task", None)
    dataset_name = state.get("dataset_name", None) 
    abstraction = state.get("abstraction", None)

    logger.info(f"{logging_prefix} - Invoking plan generation chain")

    # assemble the prompt and invoke the plan generation chain
    try:
      result = data_engineering_chain.invoke({ 
        "task": task, 
        "documentation" : flowetl_documentation, 
        "dataset_name" : dataset_name, 
        "abstraction" : abstraction, 
        "feedback" : feedback_text 
      })
    except Exception as e:
      raise Exception(f"Error occured during data engineering chain invokation: {str(e)}")
      
    logger.info(f"{logging_prefix} - Chain succesfully invoked")

    # extract the pipeline of flowetl functions generated and update the graph state
    new_pipeline = result.get('pipeline', None)
    next_state["pipeline"] = new_pipeline
    next_state["flowetl_schema"] = result.get('flowetl_schema', None)

    logger.info(f"{logging_prefix} - Output pipeline : {json.dumps(new_pipeline, indent=2)}")

  # if there is no feedback given, the plan is valid
  next_state["is_valid"] = feedback == []

  logger.info(f"{logging_prefix} - Generated plan is { 'valid' if feedback == [] else 'not valid'}")
  logger.info(f"{logging_prefix} - Graph state updated. Exiting Planner.")

  return next_state


def validator(state : GraphState) -> GraphState:

  logging_prefix = f"Validator [{state['iterations']}]"
  logger.info(f"{logging_prefix} - Entered Validator")

  # extract required artifacts for validation from previous state
  task = state.get("task", None)
  pipeline = state.get("pipeline", None)
  flowetl_schema = state.get("flowetl_schema", None)

  logger.info(f"{logging_prefix} - Invoking validation chain")

  try:
    # assemble validation prompt and extract any generated feedback
    result = plan_validation_chain.invoke({ 
      "task": task,  "flowetl_schema" : flowetl_schema, "pipeline" : pipeline
    })
  except Exception as e:
    raise Exception(f"Error occured during within the plan validation chain : {str(e)}")

  logger.info(f"{logging_prefix} - Validation process complete")

  # update the state and return
  next_state = dict(state)
  validation =  result.get("validation", None)
  next_state["validation"] = validation

  # check if the validator spotted any validation, if none then we exit the graph
  if validation:
    logger.info(f"{logging_prefix} - Validation errors : {json.dumps(validation)}")
    iters = next_state.get("iterations")
    next_state["iterations"] = iters + 1
  else:
    # no validation returned by the validator, therefore the plan must be valid
    logger.info(f"{logging_prefix} - No validation errors encountered, plan is syntactically and logically valid.")
    next_state["is_valid"] = True

  logger.info(f"{logging_prefix} - Graph state updated. Exiting Validator.")
  return next_state


# this function enables routing within the langgraph based on the output of the validator node
def router(state: GraphState) -> Literal["CONTINUE", "END"]:
  validation = state.get("validation", {})
  iters = state.get("iterations", 0)

  if validation and iters <= MAX_RETRIES:
    # allow 3 iterations before failing the graph 
    return "CONTINUE"
  else:
    # graph run out of iterations to compute a valid plan or the plan is valid
    return "END"
  

def build_graph():
  graph = StateGraph(GraphState)
  graph.add_node("planner", planner)
  graph.add_node("validator", validator)

  graph.add_edge(START, "planner")
  graph.add_edge("planner", "validator")
  graph.add_conditional_edges("validator", router, { "CONTINUE": "planner", "END": END })
  
  return graph.compile()


@app.post("/transform")
async def transform_data(request_payload: DataEngineerRequest, request: Request) -> Dict[str, Any]:

  try:
  
    logging_prefix = "Endpoint /transform"

    try:
      # reconstruct DataFrame from JSON
      abstraction = pd.DataFrame(json.loads(request_payload.abstraction))
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while ingesting full dataset: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while ingesting full dataset: {str(e)}")

    logger.info(f"{logging_prefix} - Received full dataset")

    # take the min between 10% sample of the abstraction or 25 rows - this makes processing quicker
    # we assume that the plan generated will be successfully applied to the entire dataset
    sample_size = min(25, int(len(abstraction) * 0.1)) 

    try:
      sampled_abstraction = abstraction.sample(n=sample_size).to_json()
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while sampling dataset: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while sampling dataset: {str(e)}")

    # extract the source dataset name and task description from request payload
    dataset_name = request_payload.dataset_name
    task = request_payload.task

    logger.info(f"{logging_prefix} - Extracted sample of size {sample_size} from dataset '{dataset_name}'")
    logger.info(f"{logging_prefix} - Received the following data engineering task : '{task}'")

    try:
      # configure input for the langgraph
      inputs: GraphState = { 
        "task": task, "dataset_name": dataset_name, "abstraction" : sampled_abstraction, "iterations": 1, "is_valid" : False
      }
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered in graph: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered in graph: {str(e)}")
   

    logger.info(f"{logging_prefix} - Triggering Planner-Validator graph")
    final_state = build_graph().invoke(inputs)

    logger.info(f"{logging_prefix} - Exited graph.")
    # check whether the planner and validator failed to synthetise a plan
    is_valid = final_state.get("is_valid", False)

    if not is_valid:
      logger.error(f"{logging_prefix} - Plan could not be generated within bounded iterations.")
      raise Exception("plan could not be generated within bounded iterations")
    
    else:
      logger.info(f"{logging_prefix} - Generated plan is valid.")

    pipeline = final_state.get("pipeline", None) # we expect this to not fail, since the validator and pydantic models should work ok
    flowetl_schema = final_state.get("flowetl_schema", None)

    logger.info(f"{logging_prefix} - Post-graph plan : {json.dumps(pipeline, indent=2, ensure_ascii=False)}")
    logger.info(f"{logging_prefix} - Post-graph FlowETL schema : {json.dumps(flowetl_schema, indent=2, ensure_ascii=False)}")

    # compute the data quality report on the input (before-transformation) pipeline

    try:
      data_quality_report_before = compute_data_quality(abstraction, flowetl_schema, pipeline, logger)
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while computing pre-transformation data quality report: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while computing pre-transformation data quality report: {str(e)}")
   
    logger.info(f"{logging_prefix} - Computed data quality report pre-transformation : {json.dumps(data_quality_report_before, indent=2)}")

    try:
      # apply the pipeline onto the abstracted dataset
      abstraction, errors = apply_pipeline(abstraction, flowetl_schema, pipeline, logger)
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while applying pipeline: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while applying pipeline: {str(e)}")
   
    # TODO - make sure the errors (if any) are passed to the validation agent 

    try:
      data_quality_report_after= compute_data_quality(abstraction, flowetl_schema, pipeline, logger)
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while computing post-transformation data quality report: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while computing post-transformation data quality report: {str(e)}")

    logger.info(f"{logging_prefix} - Computed data quality report post-transformation : {json.dumps(data_quality_report_after, indent=2)}")

    return { 
      "processed_abstraction" : abstraction.to_json(orient='records'),
      "data-quality-before" : data_quality_report_before, 
      "data-quality-after" : data_quality_report_after,
    } 

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"failed to process data: {str(e)}", headers={"X-Request-ID": request.state.request_id})

@app.post("/analyze")
def analyze_data(request_payload : DataAnalystRequest, request : Request) -> Dict[str, Any]:

  try:
    logging_prefix = "Endpoint /analyze"
  
    try:
      # extract the abstracted dataset sample and the user query
      abstraction = pd.DataFrame(json.loads(request_payload.abstraction))
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while loading dataset: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while loading dataset: {str(e)}")

    task = request_payload.task

    logger.info(f"{logging_prefix} - Received full dataset")
    logger.info(f"{logging_prefix} - Received the following data analysis task : '{task}'")

    logger.info(f"{logging_prefix} - Triggering data analysis chain")

    try:
      # invoke the plan construction chain
      result = data_analysis_chain.invoke({ "task" : task, "dataset" : abstraction })
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while computing query results: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while computing query results: {str(e)}")


    logger.info(f"{logging_prefix} - Chain completed succesfully, artifacts retured.")
    
    analysis_code = result.get("analysis_code", None)
    plot_code = result.get("plot_code", None)

    try:
      logger.info(f"{logging_prefix} - Attempting to convert artifact 'analysis code' into runnable function")
      # extract the code analysis function
      namespace = {}
      exec(analysis_code, namespace)
      runnable_function = namespace['analyze_data'] # called 'analyze_data' as per the llm prompt
      logger.info(f"{logging_prefix} - Conversion succesfull")

    except Exception:
      logger.error(f"{logging_prefix} - Error occured while converting artifact 'analysis code' into runnable function")
      raise Exception("Error occured while converting artifact 'analysis code' into runnable function")
    
    logger.info(f"{logging_prefix} - Computing data analysis results")
    # execute the function to obtain the analysis results
    data_analysis_result = runnable_function(abstraction)
    logger.info(f"{logging_prefix} - Results obtained and invoking the result summarisation chain")

    try:
      summary = summariser_chain.invoke({ "task" : task, "results" : data_analysis_result })
    except Exception as e:
      logger.error(f"{logging_prefix} - Error encountered while invoking summarisation chain: {str(e)}")
      raise Exception(f"{logging_prefix} - Error encountered while invoking summarisation chain: {str(e)}")


    logger.info(f"{logging_prefix} - Summary computed succesfully")

    # return the plotting function and the natural language summary to the frontend
    return { 
      "plot_code" : plot_code, 
      "summary" : summary,
    }

  except Exception as e:
    raise HTTPException(
      status_code=400, detail=f"Failed to process data: {str(e)}", headers={"X-Request-ID": request.state.request_id}
    )