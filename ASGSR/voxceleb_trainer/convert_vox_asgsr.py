# convert vox_trainer to ASGSR format
# *** temp ***

import torch
# from models.XVector import MainModel
from models.RawNet3 import MainModel

# model_path = '/home/wangli/voxceleb_trainer/exps/XVector_AAM/model/model000000230.model'
# model = torch.load(model_path, map_location='cpu')
# asgsr_dict = {}
# for name, param in model.items():  # collections.OrderedDict
#     if '__S__.' in name:
#         asgsr_dict[name.replace('__S__.', '')] = param
#
# mainmodel = MainModel(nOut=512)
# mainmodel.load_state_dict(asgsr_dict)
# torch.save(mainmodel.state_dict(), '/home/wangli/ASGSR/voxceleb_trainer/exps/XVector_AAM/model/model000000230.pth')

asgsr_dict = {}
model_path = '/home/wangli/ASGSR/pretrained_models/RawNet3/model.pt'
model = torch.load(model_path, map_location='cpu')
vox_dict = {}
for name, param in model.items():  # collections.OrderedDict
    asgsr_dict['__S__.' + name] = param
# mainmodel = MainModel(encoder_type='ECA', nOut=256, sinc_stride=10)
# mainmodel.load_state_dict(asgsr_dict)
torch.save(asgsr_dict, '/home/wangli/ASGSR/pretrained_models/RawNet3/model.model')

