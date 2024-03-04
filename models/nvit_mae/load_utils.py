import torch


def load_vit_pretrain(args, model):
    v_model_path = args.v_path
    print(f'\t\t|-continue visual training from {v_model_path}...')
    checkpoint = torch.load(v_model_path, map_location="cpu")

    params = dict()

    for key, val in model.state_dict().items():
        if key in checkpoint["model"].keys():
            params.setdefault(key, checkpoint["model"][key])
            print(f'\t\t|-Load : {key}')
        else:
            params.setdefault(key, val)
            print(f'\t\t|-Scratch : {key}')

    for key, val in checkpoint["model"].items():
        if key in model.state_dict().keys():
            params[key] = val
            # print(f'\t\t|-Load : {key}')

    model.load_state_dict(params)

    return model

def freeze_model(model):
    for name, para in model.named_parameters():
        if 'nv' in name:
            print(f'\t\t|-Training : {name}')
            requires_grad = True
        else:
            # print(f'\t\t|-Freezing {name}')
            requires_grad = False

        para.requires_grad = requires_grad

    return model