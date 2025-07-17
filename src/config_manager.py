import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """loads the yaml configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"error: config file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"error parsing yaml file: {e}")
        return {}

if __name__ == '__main__':
    config = load_config("../../config.yaml")
