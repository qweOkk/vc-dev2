import torch
import sys
import warnings


warnings.filterwarnings('ignore')
sys.path.append('../')
sys.path.append('.')
import importlib
from ASGSR.attack.attack_config import config as attack_config
def load_voxcelebtrainer_model(model_name, model_config, params_path):
    model = importlib.import_module('ASGSR.voxceleb_trainer.models.' + model_name).__getattribute__('MainModel')
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

def get_baseline_model(model_name):
    model_config_path = attack_config.model[model_name].config_path
    print('model_config_path: {}'.format(model_config_path))
    model_config = load_yaml_to_dict(model_config_path)
    model_param_path = attack_config.model[model_name].save_path
    print('model_param_path: {}'.format(model_param_path))
    model = load_voxcelebtrainer_model(model_name, model_config, model_param_path)
    model.eval()
    print('model: {}'.format(model_name))
    print('load model from {}'.format(model_param_path))
    return model


