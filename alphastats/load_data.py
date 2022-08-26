from alphastats.loader.AlphaPeptLoader import AlphaPeptLoader
from alphastats.loader.DIANNLoader import DIANNLoader
from alphastats.loader.FragPipeLoader import FragPipeLoader
from alphastats.loader.MaxQuantLoader import *


def load_data(file, type, **kwargs):
    type = type.lower()
    if type == "maxquant":
        loader = MaxQuantLoader(file=file, **kwargs)
    elif type == "alphapept":
        loader = AlphaPeptLoader(file=file, **kwargs)
    elif type == "diann":
        loader = DIANNLoader(file=file, **kwargs)
    elif type == "fragpipe":
        loader = FragPipeLoader(file=file, **kwargs)
    else:
        raise ValueError(
            f"type: {type} is invalid. Choose from maxquant, alphapept, diann, fragpipe"
        )
    return loader
