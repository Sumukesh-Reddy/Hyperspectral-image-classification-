import torch.optim as optim


def load_scheduler(model_name, model):
    optimizer, scheduler = None, None

    if model_name == 'hybridsn':
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = None

    elif model_name == 'gscvit':
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
        scheduler = None

    else:
        raise KeyError("{} model is not supported yet".format(model_name))

    return optimizer, scheduler
