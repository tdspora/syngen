import os
import toml
from streamlit.web import bootstrap


def start():
    dir_path = os.path.dirname(__file__)
    file_path = os.path.join(dir_path, "run.py")
    config_path = os.path.join(dir_path, ".streamlit", "config.toml")

    config = toml.load(config_path)

    config_options = {}
    for section in config:
        for setting, value in config[section].items():
            option_name = f"{section}_{setting}"
            config_options[option_name] = value

    bootstrap.load_config_options(config_options)
    bootstrap.run(file_path, False, [], flag_options={})
