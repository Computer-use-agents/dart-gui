from transformers import GenerationConfig
MODEL_PATH = "../UI-TARS-1.5-7B"

generation_config = GenerationConfig.from_pretrained(MODEL_PATH)