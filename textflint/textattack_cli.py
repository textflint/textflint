"""

TextFlint Command Arg Parsing Main Function
=============================================
"""

# !/usr/bin/env python
import argparse
from .engine import Engine
from .adapter import get_config


def main():
    """
    This is the main command line parer and entry function to use TextAttack
    via command lines.

    textflint <command> [<args>]

    :param string command: dataset, config, model
    :param string [<args>]: depending on the command string

    """
    parser = argparse.ArgumentParser(
        "TextFlint CLI",
        usage="[python -m] textflint <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help='file path of json/csv file which satisfy TextFlint requirements',
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help='file path of json file or Config object',
    )

    parser.add_argument(
        "--task",
        type=str,
        default='UT',
        help='task name which will be helpful without config input',
    )

    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help='path to the python file containing '
             'the FlintModel instance which named "model"',
    )

    # run !
    args = parser.parse_args()

    engine = Engine()
    config = args.config if args.config else get_config(args.task)
    engine.run(args.dataset, config)


if __name__ == "__main__":
    main()
