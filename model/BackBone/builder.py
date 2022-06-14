import tensorflow as tf

def BackBoneBuild(config):
    if config["model_config"]["backbone"]["name"].upper() in ["MOBILEDET"]:
        if config["model_config"]["backbone"]["modelType"].upper() == "CPU":
            from model.BackBone.MobileDet import MobileDetCPU
            backbone = MobileDetCPU
        elif config["model_config"]["backbone"]["modelType"].upper() == "GPU":
            from model.BackBone.MobileDet import MobileDetGPU
            backbone = MobileDetGPU

    elif config["model_config"]["backbone"]["name"].upper() in ["MV3", "MOBILENETV3", "MOBILENET3"]:
        if config["model_config"]["backbone"]["modelSize"].upper() == "SMALL":
            from model.BackBone.MobilenetV3 import MobileNetV3Small
            backbone = MobileNetV3Small
        else:
            from model.BackBone.MobilenetV3 import MobileNetV3Large
            backbone = MobileNetV3Large
    else:
        raise ValueError("Not implemented yet or misspelled")

    return backbone
    