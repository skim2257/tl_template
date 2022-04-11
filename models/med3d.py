### REPURPOSED from https://github.com/Tencent/MedicalNet ###

import torch
from .resnet import resnet10, resnet18, resnet34, resnet50


def generate_model(input_W: int=128,
                   input_H: int=128,
                   input_D: int=128,
                   resnet_shortcut: str="A",
                   new_layer_names=['conv_seg'],
                   pretrained: bool=False,
                   pretrain_path: str=None,
                   phase: str="train",
                   resnet: str="resnet18"):
    if resnet == "resnet10":
        model = resnet10(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut)
    elif resnet == "resnet18":
        model = resnet18(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut)
    elif resnet == "resnet34":
        model = resnet34(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut)
    elif resnet == "resnet50":
        model = resnet50(
            sample_input_W=input_W,
            sample_input_H=input_H,
            sample_input_D=input_D,
            shortcut_type=resnet_shortcut)

    model = model.cuda() 
    net_dict = model.state_dict() 
    # print(net_dict.keys())
    # load pretrain
    if phase != 'test' and pretrained:
        print ('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path)
        # pretrain_dict = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items() if k.replace('module.', '') in net_dict.keys()}
        # print(pretrain_dict.keys())
        
        new_dict = {}
        for k, v in net_dict.items():
            k_resnet = 'module.' + k
            if k_resnet in pretrain['state_dict'].keys():
                new_dict[k] = pretrain['state_dict'][k_resnet]
            else:
                new_dict[k] = v
        model.load_state_dict(new_dict)

        # new_parameters = [] 
        # for pname, p in model.named_parameters():
        #     for layer_name in new_layer_names:
        #         if pname.find(layer_name) >= 0:
        #             new_parameters.append(p)
        #             break

        # new_parameters_id = list(map(id, new_parameters))
        # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        # parameters = {'base_parameters': base_parameters, 
        #               'new_parameters': new_parameters}

        return model#, parameters

    return model#, model.parameters()
