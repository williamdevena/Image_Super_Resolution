

import logging
import os
from datetime import datetime

from src import baselines, logging_utilities


def main():
    """
    Executes the all the stages of the project
    """

    ## INITIAL CONFIGURATIONS
    if not os.path.exists("./project_log"):
        os.mkdir("./project_log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler("project_log/assignment.log"),
            logging.StreamHandler()
        ]
    )
    logging.info((('-'*70)+'\n')*5)
    logging_utilities.print_name_stage_project("IMAGE SUPER RESOLUTION")
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logging.info(f"DATE AND TIME OF EXCUTION: {dt_string}")




    ## BASELINES TESTING
    baselines.baselines()


if __name__ == "__main__":
    main()
