import torchvision
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

pretrained_path = {
    'base_augreg_in21k_ft_in1k': '', # path_to_weights
    'large_augreg_in21k_ft_in1k': '', # path_to_weights
    'base_swag_ig_ft_plc365': '', # path_to_weights
    'base_mae_in1k_ft_in1k': '', # path_to_weights
    'large_mae_in1k_ft_in1k': '', # path_to_weights
    'base_mae_in1k_ft_plc365': '', # path_to_weights
    'large_mae_in1k_ft_plc365': '', # path_to_weights
}


def get_model(args, device=None):
    if args.vit_size == 'base':
        hidden_size, num_hidden_layers, intermediate_size, num_attention_heads = 768, 12, 3072, 12
    elif args.vit_size == 'large':
        hidden_size, num_hidden_layers, intermediate_size, num_attention_heads = 1024, 24, 4096, 16
    elif args.vit_size == 'tiny':
        hidden_size, num_hidden_layers, intermediate_size, num_attention_heads = 192, 12, 768, 3
    else:
        raise ValueError(f'Invalid ViT size! {args.vit_size}')


    if args.method.lower() == 'vanilla':
        model = None
        pass

    elif args.method.lower() == 'timm_augreg_in21k_ft_in1k':
        pretrained_cfg = 'augreg_in21k_ft_in1k'
        print(f'\t|-Encoder: Timm vit_base_patch16_224 - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.vit_size.lower() == 'base':
            from .nvit_vit.modeling_nvit_vit import nvit_base_patch16_224 as modeling
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size.lower() == 'large':
            from .nvit_vit.modeling_nvit_vit import nvit_large_patch16_224 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
        else:
            raise ValueError(f'Invalid vit size! {pretrained_cfg} do not have {args.vit_size}')

        model = modeling(
            pretrained=False,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            nvit_depth=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_vit.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_vit.load_utils import freeze_model as freeze
            model = freeze(model)


    elif args.method.lower() == 'swag_ig_ft_plc365':
        pretrained_cfg = 'swag_ig_ft_plc365'
        if args.rep:
            pretrained_cfg += '_ft'
        print(f'\t|-Encoder: Timm vit_base_patch16_224 - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.vit_size.lower() == 'base':
            from .nvit_swag.modeling_nvit_swag import ViTB16 as modeling
            image_size = 384
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size.lower() == 'large':
            from .nvit_swag.modeling_nvit_swag import ViTL16 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
            image_size = 512
        else:
            raise ValueError(f'Invalid vit size! {pretrained_cfg} do not have {args.vit_size}')

        model = modeling(
            image_size=image_size,
            num_classes=args.class_num,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            num_nvlayers=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_swag.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_swag.load_utils import freeze_model as freeze
            model = freeze(model)

    elif args.method.lower() == 'swag_ig_ft_in1k':
        pretrained_cfg = 'swag_ig_ft_in1k'
        if args.rep:
            pretrained_cfg += '_ft'
        print(f'\t|-Encoder: Timm vit_base_patch16_224 - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.vit_size.lower() == 'base':
            from .nvit_swag.modeling_nvit_swag import ViTB16 as modeling
            image_size = 384
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size.lower() == 'large':
            from .nvit_swag.modeling_nvit_swag import ViTL16 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
            image_size = 512
        else:
            raise ValueError(f'Invalid vit size! {pretrained_cfg} do not have {args.vit_size}')

        model = modeling(
            image_size=image_size,
            num_classes=args.class_num,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            num_nvlayers=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_swag.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_swag.load_utils import freeze_model as freeze
            model = freeze(model)

    elif args.method.lower() == 'swag_ig_ft_inat18':
        pretrained_cfg = 'swag_ig_ft_inat18'
        if args.rep:
            pretrained_cfg += '_ft'
        print(f'\t|-Encoder: Timm vit_base_patch16_224 - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.vit_size.lower() == 'base':
            from .nvit_swag.modeling_nvit_swag import ViTB16 as modeling
            image_size = 384
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size.lower() == 'large':
            from .nvit_swag.modeling_nvit_swag import ViTL16 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
            image_size = 512
        else:
            raise ValueError(f'Invalid vit size! {pretrained_cfg} do not have {args.vit_size}')

        model = modeling(
            image_size=image_size,
            num_classes=args.class_num,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            num_nvlayers=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_swag.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_swag.load_utils import freeze_model as freeze
            model = freeze(model)

    elif args.method.lower() == 'mae_in1k_ft_in1k':
        pretrained_cfg = 'mae_in1k_ft_in1k'
        if args.vit_size =='base':
            from .nvit_mae.modeling_nvit_mae import vit_base_patch16 as modeling
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size == 'large':
            from .nvit_mae.modeling_nvit_mae import vit_large_patch16 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
        else:
            raise ValueError(f'Invalid pretrained weight: ViT {args.vit_size} do not have {pretrained_cfg} pretrained weights...')

        model = modeling(
            num_classes=args.class_num,
            global_pool=True,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            nvit_depth=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        print(f'\t|-Encoder: MAE nvit_{args.vit_size} - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_mae.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_mae.load_utils import freeze_model as freeze
            model = freeze(model)

    elif args.method.lower() == 'mae_in1k_ft_plc365':
        pretrained_cfg = 'mae_in1k_ft_plc365'
        if args.vit_size =='base':
            from .nvit_mae.modeling_nvit_mae import vit_base_patch16 as modeling
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size == 'large':
            from .nvit_mae.modeling_nvit_mae import vit_large_patch16 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
        else:
            raise ValueError(f'Invalid pretrained weight: ViT {args.vit_size} do not have {pretrained_cfg} pretrained weights...')

        model = modeling(
            num_classes=args.class_num,
            global_pool=True,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            nvit_depth=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        print(f'\t|-Encoder: MAE nvit_{args.vit_size} - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_mae.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_mae.load_utils import freeze_model as freeze
            model = freeze(model)

    elif args.method.lower() == 'mae_in1k_ft_inat18':
        pretrained_cfg = 'mae_in1k_ft_inat18'
        if args.vit_size =='base':
            from .nvit_mae.modeling_nvit_mae import vit_base_patch16 as modeling
            pretrained_cfg = 'base_' + pretrained_cfg
        elif args.vit_size == 'large':
            from .nvit_mae.modeling_nvit_mae import vit_large_patch16 as modeling
            pretrained_cfg = 'large_' + pretrained_cfg
        else:
            raise ValueError(f'Invalid pretrained weight: ViT {args.vit_size} do not have {pretrained_cfg} pretrained weights...')

        model = modeling(
            num_classes=args.class_num,
            global_pool=True,
            object_size=args.object_size,
            cluster_layer=args.target_layer,
            nvit_depth=args.num_nvit_layers,
            temperature=0.1,
            alpha=args.alpha,
        )

        print(f'\t|-Encoder: MAE nvit_{args.vit_size} - {pretrained_cfg}')
        print(f'\t\t|-Size: {args.vit_size}({num_hidden_layers})')

        if args.transfer:
            args.v_path = pretrained_path[pretrained_cfg]
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_mae.load_utils import load_vit_pretrain as load
            model = load(args, model)

        if args.freeze:
            if args.nv_weights:
                raise ValueError('No NViT pretrained weights!')
            else:
                from .nvit_mae.load_utils import freeze_model as freeze
            model = freeze(model)

    else:
        model = None
        raise TypeError(f'Invalid method type!: {args.method}')



    return model