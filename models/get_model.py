from .hybridsn import hybridsn
from .gscvit import gscvit


def get_model(model_name, dataset_name, patch_size):
    if model_name == 'hybridsn':
        model = hybridsn(dataset_name, patch_size)

    elif model_name == 'gscvit':
        model = gscvit(dataset_name)

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return model
