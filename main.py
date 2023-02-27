

import logging
import os

from src import baselines, logging_utilities


def main():

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

    ## BASELINES TESTING
    baselines.baselines()


if __name__ == "__main__":
    main()
