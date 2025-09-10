import json
import logging
import pandas as pd
from pathlib import Path
from backend.chains_utils import testcases_generator_chain

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    logging.warning("Running this file will overrid the current testcases. Please confirm that this is your desired action.")
    confirmation = str(input("Generate new testcases? Y/N : "))

    if confirmation == "Y":
        testcases = {}

        for filepath in Path("datasets").iterdir():
            filename = str(filepath.name).split("\\")[-1]
            logging.info(f"Generating testcases for {filename}")
            df = pd.read_csv(filepath).head()
            testcases[filename] = testcases_generator_chain.invoke({ "df": df.to_json() })
            
        with open("tescases.json", "w") as file:
            json.dump(testcases, file, indent=2)

        logging.info("Saved testcases to 'testcases.json'")
    else:
        logging.info("Action aborted")