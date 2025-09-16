import json
import logging
import uuid
from typing import Any, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import END, START, StateGraph

from backend.functions import *
from backend.models import DataAnalystRequest, DataEngineerRequest, GraphState
from backend.prompts import *
from backend.chains_utils import *

MAX_RETRIES = 3
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
logging.basicConfig(level=logging.INFO, filename=f"logs/{uuid.uuid4()}.logs", format="%(asctime)s - %(levelname)s - %(message)s")


def planner(state: GraphState) -> GraphState:

  logging_prefix = f"Planner [iteration {state['iterations']}"
  logging.info(f"{logging_prefix} - Pipeline : {json.dumps(state.get("pipeline", []), indent=2)}")

  # extract any feedback from the previous validation round
  feedback = state.get("errors", [])
  feedback_text = json.dumps(feedback, ensure_ascii=False, indent=2) if feedback else None

  logging.info(f"{logging_prefix} - {'No feedback received from Validator' if not feedback else feedback_text}")

  if feedback:
    # extract artifacts required by the planning agent
    task = state.get("task", None)
    dataset_name = state.get("dataset_name", None) 
    abstraction = state.get("abstraction", None)

    logging.info(f"{logging_prefix} - Generating new plan")

    # assemble the prompt and invoke the plan generation chain
    result = data_engineering_chain.invoke({ 
      "task": task, "documentation" : flowetl_documentation, "dataset_name" : dataset_name, "abstraction" : abstraction, "feedback" : feedback_text 
    })

    # extract the pipeline of flowetl functions generated and update the graph state
    next_state = dict(state)
    next_state["pipeline"] = result['pipeline']
    next_state["flowetl_schema"] = result['flowetl_schema']

  # if there is no feedback given, the plan is valid
  next_state["is_valid"] = feedback == []

  logging.info(f"{logging_prefix} - Graph state updated. Exiting Planner.")
  return next_state


def validator(state : GraphState) -> GraphState:

  logging_prefix = f"Validator [iteration {state['iterations']}"

  # extract required artifacts for validation from previous state
  task = state.get("task", None)
  pipeline = state.get("pipeline", None)
  flowetl_schema = state.get("flowetl_schema")

  logging.info(f"{logging_prefix} - Initialising validation process")

  # assemble validation prompt and extract any generated feedback
  result = plan_validation_chain.invoke({ "task": task,  "flowetl_schema" : flowetl_schema, "pipeline" : pipeline })

  logging.info(f"{logging_prefix} - Validation process complete")

  # update the state and return
  next_state = dict(state)
  validation =  result.get("validation", {})
  next_state["validation"] = validation

  # check if the validator spotted any validation, if none then we exit the graph
  if len(validation) > 0:
    iters = next_state.get("iterations", 0)
    next_state["iterations"] = iters + 1
  else:
    # no validation returned by the validator, therefore the plan must be valid
    next_state["is_valid"] = True

  logging.info(f"{logging_prefix} - Graph state updated. Exiting Validator.")
  return next_state


# this function enables routing within the langgraph based on the output of the validator node
def router(state: GraphState) -> Literal["ERROR", "DONE", "FAIL"]:
  validation = state.get("validation", {})
  iters = state.get("iterations", 0)

  if len(validation) > 0:
    if iters <= MAX_RETRIES:
      # allow 3 iterations before failing the graph 
      return "CONTINUE"
    else:
      # graph run out of iterations to compute a valid plan
      return "END"
    
  # plan is validated
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
async def transform_data(request: DataEngineerRequest) -> Dict[str, Any]:
  try:
    # reconstruct DataFrame from JSON
    abstraction = pd.DataFrame(json.loads(request.abstraction))

    logging.info("received complete abstraction")

    # take the min between 10% sample of the abstraction or 25 rows - this makes processing quicker
    # we assume that the plan generated will be successfully applied to the entire dataset
    sample_size = min(25, int(len(abstraction) * 0.1)) 
    sampled_abstraction = abstraction.sample(n=sample_size).to_json()

    # extract the source dataset name and task description from request payload
    dataset_name = request.dataset_name
    task = request.task

    # configure input for the langgraph
    inputs: GraphState = { "task": task, "dataset_name": dataset_name, "abstraction" : sampled_abstraction, "iterations": 1, "is_valid" : False }

    logging.info("triggering planner-validator graph")
    final_state = build_graph().invoke(inputs)

    # check whether the planner and validator failed to synthetise a plan
    is_valid = final_state.get("is_valid", False)

    if not is_valid:
      logging.error("plan could not be generated", exc_info=True)
      raise Exception("plan could not be generated within bounded iterations")

    pipeline = final_state.get("pipeline") # we expect this to not fail, since the validator and pydantic models should work ok
    flowetl_schema = final_state.get("flowetl_schema")

    logging.info("validated plan")
    logging.info(json.dumps(pipeline, indent=2, ensure_ascii=False))
    logging.info("flowetl schema")
    logging.info(json.dumps(flowetl_schema, indent=2, ensure_ascii=False))

    # compute the data quality report on the input (before-transformation) pipeline
    data_quality_report_before = compute_data_quality(abstraction, flowetl_schema, pipeline)

    logging.info(f"Data quality report before : {data_quality_report_before}")

    # apply the pipeline onto the abstracted dataset
    abstraction, errors = apply_pipeline(abstraction, flowetl_schema, pipeline)

    # TODO - make sure the errors (if any) are passed to the validation agent 
    logging.error("############### IMPLEMENT THE TODO ##################")
    logging.error(f"Errors encountered : {errors}", exc_info=True)

    # compute the data quality report on the transformed dataframe
    data_quality_report_after = compute_data_quality(abstraction, flowetl_schema, pipeline)

    logging.info(f"Data quality report after : {data_quality_report_after}")

    return { 
      "processed_abstraction" : abstraction.to_json(orient='records'),
      "data-quality-before" : data_quality_report_before, 
      "data-quality-after" : data_quality_report_after
    } 

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"failed to process data: {str(e)}")


@app.post("/analyze")
def analyze_data(request : DataAnalystRequest) -> Dict[str, Any]:
  try:
    # extract the abstracted dataset sample and the user query
    abstraction = pd.DataFrame(json.loads(request.abstraction))
    task = request.task

    logging.info("Received query and dataset for analysis")

    # invoke the plan construction chain
    result = data_analysis_chain.invoke({ "task" : task, "dataset" : abstraction })

    logging.info("Compute the result for the dataset and query")
    
    analysis_code = result["analysis_code"]
    plot_code = result.get("plot_code", None)

    # extract the code analysis function
    namespace = {}
    exec(analysis_code, namespace)
    runnable_function = namespace['analyze_data'] # called 'analyze_data' as per the llm prompt

    # execute the function to obtain the analysis results
    data_analysis_result = runnable_function(abstraction)

    # run the summary creation chain to convert the results of the data analysis to natural language
    summary = summariser_chain.invoke({ "task" : task, "results" : data_analysis_result })

    # return the plotting function and the natural language summary to the frontend
    return { "plot_code" : plot_code, "summary" : summary }

  except Exception as e:
    raise HTTPException(status_code=400, detail=f"Failed to process data: {str(e)}")