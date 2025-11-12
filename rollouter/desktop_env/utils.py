import omegaconf

def load_config(config_path: str = "configs/config.yaml"):
    return omegaconf.OmegaConf.load(config_path)
