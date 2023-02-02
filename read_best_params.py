import json
import os


def read_best_params(model, key_name, sr=0.1, cr=2.0, tr=0.0):
    dir_prefix = os.getcwd()
    file_path = "/res/ndcg/sim_{}.json".format(key_name)
    if key_name == "sr":
        key = sr
    elif key_name == "cr":
        key = cr
    else:
        key = tr
    with open(dir_prefix + file_path, "r") as f:
        config = json.load(f)
        for model_config in config["models"]:
            if model == model_config["name"]:
                for param in model_config["params"]:
                    if param[key_name] == key:
                        return param
    raise Exception("invalid ")
