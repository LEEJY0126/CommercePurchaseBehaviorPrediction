import os,yaml
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
config_path = os.path.join(PROJECT_PATH, "src/config/config.yaml")

class Config :
    def __init__ (self) :
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.data = config["data"]
        self.train = config["train"]
        self.model = config['model']

