from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from models import Plan, Analysis, Feedback, TaskDescriptionTestcases
from prompts import *

# LLM CLIENT
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-coder:6.7b")

llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=MODEL_NAME, temperature=0.0)

# LLM OUTPUT PARSERS 
data_engineering_output_parser = JsonOutputParser(pydantic_object=Plan)
data_analysis_output_parser = JsonOutputParser(pydantic_object=Analysis)
plan_validation_output_parser = JsonOutputParser(pydantic_object=Feedback)
testcases_output_parser = JsonOutputParser(pydantic_object=TaskDescriptionTestcases)

# LLM PROMPT BUILDERS
data_engineering_prompt_builder = PromptTemplate( 
  template=data_engineering_system_prompt, 
  input_variables=["task", "documentation", "abstraction", "dataset_name"], 
  partial_variables={"format_instructions": data_engineering_output_parser.get_format_instructions()} 
) 

plan_validation_prompt_builder = PromptTemplate( 
  template=validator_system_prompt, 
  input_variables=["task", "schema", "plan"]
) 

data_analysis_prompt_builder = PromptTemplate( 
  template=data_analysis_system_prompt, 
  input_variables=["dataset", "task"], 
  partial_variables={"format_instructions": data_analysis_output_parser.get_format_instructions()} 
) 

data_analysis_summary_prompt_builder = PromptTemplate( 
  template=data_analysis_summary_system_prompt, 
  input_variables=["results", "task"], 
) 

testcases_prompt_builder = PromptTemplate( 
  template=testcase_generator_system_prompt, 
  input_variables=["df"], 
  partial_variables={"format_instructions": testcases_output_parser.get_format_instructions()}
) 


# CHAINS
data_analysis_chain = data_analysis_prompt_builder | llm | data_analysis_output_parser

data_engineering_chain = data_engineering_prompt_builder | llm | data_engineering_output_parser

plan_validation_chain = plan_validation_prompt_builder | llm | plan_validation_output_parser

data_analysis_chain = data_analysis_prompt_builder | llm | data_analysis_output_parser

summariser_chain = data_analysis_summary_prompt_builder | llm | StrOutputParser()

testcases_generator_chain = testcases_prompt_builder | llm | testcases_output_parser