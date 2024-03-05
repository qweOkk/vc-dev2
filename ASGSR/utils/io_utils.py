import importlib
import torch

import torchaudio


def load_waveform_numpy():
    # read a waveform, return numpy array
    pass


def load_waveform_torch(path):
    # read a waveform, return torch tensor (1, T)
    waveform, sample_rate = torchaudio.load(path)

    return waveform, sample_rate


def save_waveform_numpy():
    # save a waveform, input is numpy array
    pass


def save_waveform_torch(path, waveform, sample_rate):
    # save a waveform, input is torch tensor
    torchaudio.save(path, waveform, sample_rate)


def load_voxcelebtrainer_model(model_name, model_config, params_path):
    model = importlib.import_module('voxceleb_trainer.models.' + model_name).__getattribute__('MainModel')
    model = model(**model_config)
    loaded_state = torch.load(params_path)
    new_dict = {}

    # deal with the model saved by voxceleb_trainer
    for name, param in loaded_state.items():
        if '__S__.' in name:
            new_name = name.replace('__S__.', '')
            new_dict[new_name] = param

    # only load model parameters which in saved parameters
    keys_to_remove = []
    for name in new_dict:
        if name not in model.state_dict():
            print(f"Warning: Parameter '{name}' not found in the model. Skipping parameter.")
            keys_to_remove.append(name)
    for name in keys_to_remove:
        del new_dict[name]
    model.load_state_dict(new_dict)
    return model


def load_yaml_to_dict(path):
    import yaml
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
