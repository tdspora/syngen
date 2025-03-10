import yaml
import subprocess
import os
import re
import csv
import sys
import json


class InferenceRunner:
    def __init__(self, config_path):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path

        with open(self.config_path, 'r') as config_file:
            self.config = json.load(config_file)

        self.configurations = self.config['configurations']
        self.csv_path = self.config['csv_path']
        self.batch_memory_csv_path = self.config['batch_memory_csv_path']
        self.original_yaml_path = self.config['original_yaml_path']
        self.table_name = self.config['table_name'] if 'table_name' in self.config else None

        # Load the original YAML to get the default table name
        with open(self.original_yaml_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)

        if not self.table_name:
            self.default_table_name = next(key for key in self.yaml_data if key != 'global')
            self.table_name = self.default_table_name

        # Add the parent directory of the script to Python path
        sys.path.append(os.path.dirname(self.script_dir))

    def create_yaml_file(self, table_name, infer_size, batch_size, run_parallel):
        # Load the original YAML
        with open(self.original_yaml_path, 'r') as file:
            self.yaml_data = yaml.safe_load(file)

        # Extract the default table name (excluding 'global')
        self.default_table_name = next(key for key in self.yaml_data if key != 'global')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f"default_table_name: {self.default_table_name}")

        # Update the YAML data
        if table_name in self.yaml_data:
            self.yaml_data[table_name]['infer_settings']['size'] = infer_size
            self.yaml_data[table_name]['infer_settings']['batch_size'] = batch_size
            self.yaml_data[table_name]['infer_settings']['run_parallel'] = run_parallel
        else:
            # Use the default table name from the template
            self.yaml_data[self.default_table_name]['infer_settings']['size'] = infer_size
            self.yaml_data[self.default_table_name]['infer_settings']['batch_size'] = batch_size
            self.yaml_data[self.default_table_name]['infer_settings']['run_parallel'] = run_parallel
            table_name = self.default_table_name

        # Create a new YAML file for this configuration
        new_yaml_path = os.path.join(self.script_dir, 'metadata', f'EPMCTDM-5946_{table_name}_{infer_size}_{batch_size}.yaml')
        with open(new_yaml_path, 'w') as file:
            yaml.dump(self.yaml_data, file)

        return new_yaml_path

    def append_result_to_csv(self, result):
        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                "table_name",
                "infer_size",
                "batch_size",
                "number_of_batches",
                "run_parallel",
                "Method_handle_Time_(sec)",
                "Memory_usage_(%)",
                "run_type"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(result)

        print(f"Result appended to {self.csv_path}")

    def read_log_file(self, log_file_path):
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as file:
                return file.read()
        return ""

    def extract_info_from_logs(self, log_content):
        memory_usage_match = re.search(r"Finished processing all batches\. Memory usage: (\d+\.\d+)%", log_content)
        execution_time_match = re.search(r"Function 'handle' executed in (\d+\.\d+) seconds", log_content)

        memory_usage = float(memory_usage_match.group(1)) if memory_usage_match else None
        execution_time = float(execution_time_match.group(1)) if execution_time_match else None

        return memory_usage, execution_time

    def run_inference_and_extract_info(self, table_name, infer_size, batch_size, run_parallel):
        print(f"\nStarting inference for table_name={table_name}, infer_size={infer_size}, batch_size={batch_size}, run_parallel={run_parallel}")

        # Update the YAML data
        if table_name in self.yaml_data:
            self.yaml_data[table_name]['infer_settings']['size'] = infer_size
            self.yaml_data[table_name]['infer_settings']['batch_size'] = batch_size
            self.yaml_data[table_name]['infer_settings']['run_parallel'] = run_parallel

        # Create a new YAML file for this configuration
        new_yaml_path = os.path.join(self.script_dir, 'metadata', f'EPMCTDM-5946_{table_name}_{infer_size}_{batch_size}.yaml')
        with open(new_yaml_path, 'w') as file:
            yaml.dump(self.yaml_data, file)

        # Run the inference command
        command = f"python3 -m start --task infer --metadata_path {new_yaml_path} --log_level TRACE"
        print(f"Running command: {command}")

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        output = []
        for line in process.stdout:
            print(line, end='')  # Print each line in real-time
            output.append(line)  # Store the line for later processing

        process.wait()

        # Join all output lines into a single string
        full_output = ''.join(output)

        # Extract information from the output
        memory_usage, execution_time = self.extract_info_from_logs(full_output)

        # Calculate number of batches
        num_batches = infer_size // batch_size
        if infer_size % batch_size != 0:
            num_batches += 1

        results_of_run = {
            "table_name": table_name,
            "infer_size": infer_size,
            "batch_size": batch_size,
            "number_of_batches": num_batches,
            "run_parallel": str(run_parallel).lower(),
            "Method_handle_Time_(sec)": execution_time,
            "Memory_usage_(%)": memory_usage,
            "run_type": "script_run"
        }

        print('Results of run:')
        print(results_of_run)

        if memory_usage is None or execution_time is None:
            print("\nWARNING: Could not find memory usage or execution time information.")
            print("Last 20 lines of subprocess output:")
            for line in output[-20:]:
                print(line, end='')

        return results_of_run

    def run_all_experiments(self):
        for config in self.configurations:
            table_name = config.get("table_name", next(key for key in self.yaml_data if key != 'global'))
            size = int(config["infer_settings"]['size'])
            batch_size = int(config["infer_settings"]['batch_size'])
            run_parallel = config["infer_settings"].get('run_parallel', True)
            result = self.run_inference_and_extract_info(table_name, size, batch_size, run_parallel)
            self.append_result_to_csv(result)
            print(f"Completed: table_name={table_name}, infer_size={size}, batch_size={batch_size}")
            print("\n" + "="*50 + "\n")

        print("All experiments completed.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')

    runner = InferenceRunner(config_path)
    runner.run_all_experiments()
