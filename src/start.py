import argparse
import subprocess
import sys
from loguru import logger


PROCESS_TO_RUN = {
    "train_process": "train",
    "infer_process": "infer"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run training, inference tasks, or a Streamlit web UI.",
        add_help=False,
    )
    parser.add_argument(
        "--task", choices=["train", "infer"], help="Task to run: 'train' or 'infer'."
    )
    parser.add_argument(
        "--webui", action="store_true", help="Launch the Streamlit web UI."
    )

    # Forward unknown arguments to train.py, infer.py,
    # or Streamlit without explicit parsing in start.py
    known_args, remaining_argv = parser.parse_known_args()

    # Remaining unknown args will be passed to train/infer script or Streamlit
    return known_args, remaining_argv


def launch_and_monitor_subprocess(process_name, args):
    processes = {}
    process_name = PROCESS_TO_RUN.get(process_name)
    if process_name is None:
        logger.error(f"Invalid script key - '{process_name}'")
        return
    process = subprocess.Popen(
        ["python", f"syngen/{process_name}.py"] + args, start_new_session=True, shell=False
    )
    processes[process_name] = process
    process.wait()
    return_code = process.returncode
    if return_code == -9:
        logger.warning(
            f"\n\nWARNING!\n{process_name} was terminated, "
            f"most likely due to OOM event.\nReturn code: {return_code}"
        )
    elif return_code != 0:
        logger.warning(
            f"\n\nWARNING!\n{process_name} exited with an error.\n"
            f"Return code: {return_code}"
        )
    else:
        logger.info(f"{process_name} completed successfully.")


def main():
    known_args, remaining_argv = parse_args()

    # Check if the Streamlit web UI should be launched
    if known_args.webui:
        # Adjust the path to your Streamlit application script if necessary
        command = ["streamlit", "run", "syngen/streamlit_app/run.py"] + remaining_argv
        subprocess.run(command, check=True)
        return
    elif known_args.task == "train":
        # Construct the command to run the training script
        launch_and_monitor_subprocess("train_process", remaining_argv)
    elif known_args.task == "infer":
        # Construct the command to run the inference script
        launch_and_monitor_subprocess("infer_process", remaining_argv)
    else:
        print(
            "Unknown command. Use --task=train, --task=infer, or --webui.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
