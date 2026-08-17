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
        description="Run training or inference tasks.",
        add_help=False,
    )
    parser.add_argument(
        "--task", choices=["train", "infer"], help="Task to run: 'train' or 'infer'."
    )

    # Forward unknown arguments to train.py or infer.py without explicit parsing here
    known_args, remaining_argv = parser.parse_known_args()

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
    try:
        process.wait()
    except KeyboardInterrupt:
        logger.info(f"\nKeyboard interrupt received, terminating {process_name}...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return
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

    if known_args.task == "train":
        launch_and_monitor_subprocess("train_process", remaining_argv)
    elif known_args.task == "infer":
        launch_and_monitor_subprocess("infer_process", remaining_argv)
    else:
        print(
            "Unknown command. Use --task=train or --task=infer.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
