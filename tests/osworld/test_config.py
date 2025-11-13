import hydra


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    print("config", config, type(config))

if __name__ == "__main__":
    main()