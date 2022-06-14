import tensorflow as  tf

def NeckBuild(config):
    if config["model_config"]["neck"]["name"].upper() in ["PFH", "PFHLITE"]:
        from model.Neck.PFH import PFH
        return PFH
    elif config["model_config"]["neck"]["name"].upper() in ["FPN"]:
        from model.Neck.FPN import FPN
        return FPN
    else:
        raise ValueError(config["model_config"]["neck"]["name"] + " is not implemented yet or misspelled")