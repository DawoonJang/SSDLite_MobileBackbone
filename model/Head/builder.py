import tensorflow as  tf

def HeadBuild(config):
    if config["model_config"]["head"]["name"].upper() in ["SSD", "SSDLITE"]:
        from model.Head.SSD import SSD
        return SSD
    else:
        raise ValueError(config["model_config"]["head"]["name"] + " is not implemented yet or misspelled")