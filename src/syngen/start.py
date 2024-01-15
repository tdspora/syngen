import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run training, inference tasks, or a Streamlit web UI.", add_help=False
    )
    parser.add_argument("--task", choices=["train", "infer"], help="Task to run: 'train' or 'infer'.")
    parser.add_argument("--webui", action="store_true", help="Launch the Streamlit web UI.")

    # Forward unknown arguments to train.py, infer.py, or Streamlit without explicit parsing in start.py
    known_args, remaining_argv = parser.parse_known_args()

    # Remaining unknown args will be passed to train/infer script or Streamlit
    return known_args, remaining_argv


def main():
    known_args, remaining_argv = parse_args()

    # Check if the Streamlit web UI should be launched
    if known_args.webui:
        # Adjust the path to your Streamlit application script if necessary
        command = ["streamlit", "run", "syngen/streamlit_app/run.py"] + remaining_argv
    elif known_args.task == "train":
        # Construct the command to run the training script
        command = ["python", "syngen/train.py"] + remaining_argv
    elif known_args.task == "infer":
        # Construct the command to run the inference script
        command = ["python", "syngen/infer.py"] + remaining_argv
    else:
        print("Unknown command. Use --task=train, --task=infer, or --webui.", file=sys.stderr)
        sys.exit(1)

    # Run the command with any additional arguments
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
