import json
import logging
import requests
import pandas as pd
from pathlib import Path
import argparse

"""
file for running and managing tests
flags

--newrun --> run the testing script
--cleanup --> removes all orphaned (outdated) log files
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_tests():
    with open("tasks.json", "r") as file:
        all_tasks = json.load(file)

    results = {}

    for filepath in Path("datasets").iterdir():

        dataset_name = filepath.name
        dataset_tasks = all_tasks.get(dataset_name, {})
        dataset_contents = pd.read_csv(filepath)
        results[dataset_name] = {}

        for difficulty, task in dataset_tasks.items():
            logging.info(f"Prompting API - dataset [{dataset_name}], difficulty [{difficulty}], task '{task}'")
            try:
                response = requests.post(
                    "http://localhost:8000/transform",
                    json={
                        "dataset_name": dataset_name,
                        "abstraction": dataset_contents.to_json(),
                        "task": task,
                    },
                )

                response_body = response.json()
                status_code = response.status_code
                request_id = response.headers.get("flowetl-request-id", None)

                if status_code == 200:
                    logging.info(f"Request successful. Check runtime log with ID : {request_id}")
                else:
                    logging.error(f"Request failed. Check runtime log with ID : {request_id}")

                results[dataset_name][difficulty] = [status_code, request_id]

            except requests.exceptions.RequestException:
                logging.error("Error occurred while sending API request, testcase has been skipped.", exc_info=True)

            break
        break

    with open("report.json", "w") as file:
        json.dump(results, file, indent=2)

    logging.info("Comprehensive prompting completed. Results saved to 'report.json'")


def cleanup_logs():
    """
    cleanup orphaned logs - these are logs whose ID does not appear in the latest report.json file
    """
    log_dir = Path("../logs")
    report_file = Path("report.json")

    if not log_dir.exists():
        logging.warning("No logs folder found, skipping cleanup.")
        return

    valid_request_ids = set()

    if  report_file.exists():

        with open(report_file, "r") as file:

            report = json.load(file)

            for dataset, tasks in report.items():

                for difficulty, values in tasks.items():

                    if len(values) == 2:

                        _, request_id = values

                        if request_id in log_dir.iterdir():

                            logging.warning(f"orphaned log found {request_id}")
                            valid_request_ids.add(request_id)

    for logfile in log_dir.iterdir():

        log_id = logfile.stem

        if log_id not in valid_request_ids:

            logfile.unlink()
            logging.info(f"Deleted log: {logfile.name}")
        else:
            logging.info(f"Kept log: {logfile.name}")

    logging.info("Cleanup completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlowETL API tester")
    parser.add_argument("--newrun", action="store_true", help="Run the full test suite")
    parser.add_argument("--cleanup", action="store_true", help="Remove orphaned log files")
    args = parser.parse_args()

    if args.cleanup:
        logging.info("Removing all orphaned log files")
        cleanup_logs()

    if args.newrun:
        logging.warning("Running all tests will be an expensive operation")
        run_tests()

    if not args.cleanup and not args.newrun:
        logging.info("No flags provided. Use --help for available options.")
