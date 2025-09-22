"""
This script parses through the datasets folder, and for each testcase it sends a 
request to the FlowETL API. For each request, it logs metrics such as time elapsed, and status code.
Requests are grouped by dataset name

NOTE : this script requires the API to be running on port 8000 on localhost, unless the API is hosted elsewhere
"""

import json
import logging
import requests
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":

    logging.warning("Running this script will be an expensive operation")
    confirmation = str(input("Are you sure? (y/n) :"))

    if confirmation == "y":
            
        # load task descriptions locally
        with open("tasks.json", "r") as file:
            all_tasks = json.load(file)

        # results containing test outcome for each task, grouped by dataset name
        results = {}

        for filepath in Path("datasets").iterdir():
            
            dataset_name = str(filepath).split("\\")[-1]
            dataset_tasks = all_tasks.get(dataset_name, {})
            dataset_contents = pd.read_csv(filepath)
            results[dataset_name] = {} # group response by dataset name

            for difficulty, task in dataset_tasks.items():

                logging.info(f"Prompting API - dataset [{dataset_name}], difficulty [{difficulty}], task '{task}'")

                try:
                    response = requests.post(
                        "http://localhost:8000/transform", 
                        json={"dataset_name" : dataset_name,"abstraction" : dataset_contents.to_json(),"task": task}
                    )

                    response_body = response.json()
                    status_code = response_body.get("status_code", None)
                    request_id = response_body.get("response_id", None)

                    if status_code == 200:
                        logging.info(f"Request successful. Check runtime log with ID : {request_id}")

                    # save the response status code 
                    results[dataset_name][difficulty] = status_code

                except requests.exceptions.RequestException as e:
                    logging.error("Error occurred while sending API request, testcase has been skipped.", exc_info=True)

        # save results
        with open("report.json", "w") as file:
            json.dump(results, file, indent=2)

        logging.info("Comprehensive prompting completed. Results saved to 'report.json'")
        
    else:
        logging.info("Operation aborted.")