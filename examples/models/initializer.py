import torch
import torch.nn as nn
import os
import traceback
import numpy as np
from models.layers import Identity
from utils import load

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)

        Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained (for bert-based models)
        or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    """
    # If load_featurizer_only is True,
    # then split into (featurizer, classifier) for the purposes of loading only the featurizer,
    # before recombining them at the end

    featurize = is_featurizer or config.load_featurizer_only
    if config.pretrained_model_path is not None:
        model_urls = {config.model: config.pretrained_model_path}

    #model_urls = { 'resnet50_gn': '/home/katyag/OHSU/Detection/wilds/resnet50_gn-9186a21c.pth',
    #               'BiT-M-R50x1': '/home/katyag/OHSU/Detection/wilds/BiT-M-R50x1.npz',
    #               'BiT-M-R50x3': '/home/katyag/OHSU/Detection/wilds/BiT-M-R50x3.npz',
    #               'resnet50_gn_ws': '/home/katyag/OHSU/Detection/wilds/resnet50_gn_ws-15beedd8.pth',
    #             #  'resnet50_gn_silu': '/home/katyag/OHSU/Detection/wilds/resnet50_gn_ws-15beedd8.pth',
    #               #'resnet50_gnws_silu': '/home/katyag/OHSU/Detection/wilds/resnet50_GNws_8xb32_silu_in1k.pth',
    #               'resnet50_gn_silu': '/home/katyag/OHSU/Detection/wilds/resnet50_GNws_silu.pth',
    #               'resnet50_gnws_silu': '/home/katyag/OHSU/Detection/wilds/resnet50_GNws_silu.pth',
    #               'resnet50_gnws_gelu': '/home/katyag/OHSU/Detection/wilds/resnet50_GNws_GELU_8x32_ep34_in1k.pth',
    #                'resnet50_gnws_LeakyRelu':
    #               '/home/katyag/OHSU/Detection/wilds/resnet50_GNws_LeakyReLU_8x32_ep69_in1k.pth', #resnet50_GNws_Leaky2_8x32_in1k.pth',
    #            }

    if config.model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50', 'densenet121'):
        if featurize:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)

    elif 'bert' in config.model:
        if featurize:
            featurizer = initialize_bert_based_model(config, d_out, featurize)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_bert_based_model(config, d_out)

    elif config.model == 'resnet18_ms':  # multispectral resnet 18
        from models.resnet_multispectral import ResNet18
        if featurize:
            featurizer = ResNet18(num_classes=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet18(num_classes=d_out, **config.model_kwargs)
    elif config.model == 'resnet18_gnws_silu':  # multispectral resnet 18
        from models.resnet_GNws_SiLu import ResNet18

        if featurize:
            featurizer = ResNet18(num_classes=None) #, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet18(num_classes=d_out,num_channels=config.model_kwargs['num_channels'])

    elif config.model in ('resnet50_gnws_gelu','resnet50_gn_silu','resnet50_gnws_silu','resnet50_gnws_LeakyRelu'):  # multispectral resnet 18
        if config.model in ('resnet50_gn_silu','resnet50_gnws_silu'):
            from models.resnet_GNws_SiLu import ResNet50
        elif config.model == 'resnet50_gnws_gelu':
            from models.resnet_GNws_GeLu import ResNet50
        else:
            from models.resnet_GNws_LeakyReLu import ResNet50

        if featurize:
            featurizer = ResNet50(num_classes=None) #, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet50(num_classes=d_out,num_channels=config.model_kwargs['num_channels'])
            if config.model_kwargs['pretrained']==True:
                state_dict = torch.load(model_urls[config.model])
                state_dict['state_dict'] = dict({k.replace("backbone.",""):v for k,v in state_dict['state_dict'].items() })
                state_dict['state_dict'] = dict({k.replace("gn1","bn1"):v for k,v in state_dict['state_dict'].items() })
                state_dict['state_dict'] = dict({k.replace(".gn",".bn"):v for k,v in state_dict['state_dict'].items() })
                state_dict['state_dict'] = dict({k.replace("fc.","Fc."):v for k,v in state_dict['state_dict'].items() })
                model.load_state_dict(state_dict['state_dict'] ,strict=False)
                # freeze the 1 layer
                layer_to_freeze = config.model_kwargs['frozen_layers'] #0 # 6
                if layer_to_freeze>0:
                    ct = 0
                    for ch_name, child in model.named_children():
                    #for child in model.children():
                        ct += 1
                        if ct < layer_to_freeze:
                            for name, param in child.named_parameters():
                            #for param in child.parameters():
                                print("Freeze:--->",ch_name,name)
                                param.requires_grad = False
    elif config.model in ('BiT-M-R50x1','BiT-M-R18x1','BiT-M-R50x3'):  # multispectral resnet 18
        import models.bit_models as bit_models
        #BiT-M-R50x1
        model = bit_models.KNOWN_MODELS[config.model](head_size=d_out,
                zero_head=True,num_channels=config.model_kwargs['num_channels'])
        if config.model_kwargs['pretrained']==True:
            model.load_from(np.load(model_urls[config.model]))
            layer_to_freeze = config.model_kwargs['frozen_layers'] #0 # 6
            if layer_to_freeze>0:
                ct = 0
                for  m_name, m in model.named_children():
                    ct += 1
                    if ct < layer_to_freeze:
                        for name,param in m.named_parameters():
                            if 'block1.' in name or 'root' in m_name:
                                param.requires_grad = False
                                print(m_name,name, "param.requires_grad=",param.requires_grad)
    elif config.model == 'resnet50_gn_ws':
        from models.resnet_GN_ws import ResNet50

        if featurize:
            featurizer = ResNet50(num_classes=None) #, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
            print(model)
        else:
            model = ResNet50(num_classes=d_out,num_channels=config.model_kwargs['num_channels'])
        if config.model_kwargs['pretrained']==True:
            state_dict = torch.load(model_urls[config.model])
            state_dict['state_dict'] = dict({k.replace("gn1","bn1"):v for k,v in state_dict['state_dict'].items() })
            state_dict['state_dict'] = dict({k.replace(".gn",".bn"):v for k,v in state_dict['state_dict'].items() })
            state_dict['state_dict'] = dict({k.replace("fc.","FC."):v for k,v in state_dict['state_dict'].items() })
            model.load_state_dict(state_dict['state_dict'] ,strict=False)
        # freze the 1 layer
            layer_to_freeze = config.model_kwargs['frozen_layers'] #0 # 6
            if layer_to_freeze>0:
                ct = 0
                for child in model.children():
                    ct += 1
                    if ct < layer_to_freeze:
                        for param in child.parameters():
                            param.requires_grad = False
    elif config.model == 'resnet50_gn':  # multispectral resnet 18
        from models.resnet_GN import ResNet50

        if featurize:
            featurizer = ResNet50(num_classes=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet50(num_classes=d_out,num_channels=config.model_kwargs['num_channels'])
            if config.model_kwargs['pretrained']==True:
                state_dict = torch.load(model_urls[config.model])
                state_dict['state_dict'] = dict({k.replace("gn1","bn1"):v for k,v in state_dict['state_dict'].items() })
                state_dict['state_dict'] = dict({k.replace(".gn",".bn"):v for k,v in state_dict['state_dict'].items() })
                state_dict['state_dict'] = dict({k.replace("fc.","FC."):v for k,v in state_dict['state_dict'].items() })
                model.load_state_dict(state_dict['state_dict'] )#,strict=False)
            # freze the 1 layer
            layer_to_freeze = config.model_kwargs['frozen_layers'] #0 # 6
            if layer_to_freeze>0:
                ct = 0
                for child in model.children():
                    ct += 1
                    if ct < layer_to_freeze:
                        for param in child.parameters():
                            param.requires_grad = False

    elif config.model == 'gin-virtual':
        from models.gnn import GINVirtual
        if featurize:
            featurizer = GINVirtual(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'code-gpt-py':
        from models.code_gpt import GPT2LMHeadLogit, GPT2FeaturizerLMHeadLogit
        from transformers import GPT2Tokenizer
        name = 'microsoft/CodeGPT-small-py'
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        if featurize:
            model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))
            featurizer = model.transformer
            classifier = model.lm_head
            model = (featurizer, classifier)
        else:
            model = GPT2LMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))

    elif config.model == 'logistic_regression':
        assert not featurize, "Featurizer not supported for logistic regression"
        model = nn.Linear(out_features=d_out, **config.model_kwargs)
    elif config.model == 'unet-seq':
        from models.CNN_genome import UNet
        if featurize:
            featurizer = UNet(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = UNet(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'fasterrcnn':
        if featurize:
            raise NotImplementedError('Featurizer not implemented for detection yet')
        else:
            model = initialize_fasterrcnn_model(config, d_out)
        model.needs_y = True


    # Load pretrained weights from disk using our utils.load function
    if config.pretrained_model_path is not None:
        if config.model in ('code-gpt-py', 'logistic_regression', 'unet-seq'):
            # This has only been tested on some models (mostly vision), so run this code iff we're sure it works
            raise NotImplementedError(f"Model loading not yet tested for {config.model}.")

        if 'bert' not in config.model and 'resnet' not in config.model and 'BiT' not in config.model:  # We've already loaded pretrained weights for bert-based models using the transformers library
            try:
                if featurize:
                    if config.load_featurizer_only:
                        model_to_load = model[0]
                    else:
                        model_to_load = nn.Sequential(*model)
                else:
                    model_to_load = model

                prev_epoch, best_val_metric = load(
                    model_to_load,
                    config.pretrained_model_path,
                    device=config.device)

                print(
                    (f'Initialized model with pretrained weights from {config.pretrained_model_path} ')
                    + (f'previously trained for {prev_epoch} epochs ' if prev_epoch else '')
                    + (f'with previous val metric {best_val_metric} ' if best_val_metric else '')
                )
            except Exception as e:
                print('Something went wrong loading the pretrained model:')
                traceback.print_exc()
                raise

    # Recombine model if we originally split it up just for loading
    if featurize and not is_featurizer:
        model = nn.Sequential(*model)

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if is_featurizer:
            for submodel in model:
                submodel.needs_y = False
        else:
            model.needs_y = False

    return model


def initialize_bert_based_model(config, d_out, featurize=False):
    from models.bert.bert import BertClassifier, BertFeaturizer
    from models.bert.distilbert import DistilBertClassifier, DistilBertFeaturizer

    if config.pretrained_model_path:
        print(f'Initialized model with pretrained weights from {config.pretrained_model_path}')
        config.model_kwargs['state_dict'] = torch.load(config.pretrained_model_path, map_location=config.device)

    if config.model == 'bert-base-uncased':
        if featurize:
            model = BertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    elif config.model == 'distilbert-base-uncased':
        if featurize:
            model = DistilBertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    else:
        raise ValueError(f'Model: {config.model} not recognized.')
    return model

def initialize_torchvision_model(name, d_out, **kwargs):
    import torchvision

    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model

def initialize_fasterrcnn_model(config, d_out):
    if 'BiT-M-R50' in config.model_kwargs["backbone_model"]:
        from models.detection.fasterrcnn_bit import fasterrcnn_resnet50_fpn
    elif 'resnet50_gn' in config.model_kwargs["backbone_model"]:
        from models.detection.fasterrcnn_gn import fasterrcnn_resnet50_fpn
    else:
        from models.detection.fasterrcnn import fasterrcnn_resnet50_fpn

    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(
        pretrained=config.model_kwargs["pretrained_model"],
        pretrained_backbone=config.model_kwargs["pretrained_backbone"],
        num_classes=d_out,
        min_size=config.model_kwargs["min_size"],
        max_size=config.model_kwargs["max_size"],
        backbone_name =config.model_kwargs["backbone_model"],
        backbone_path =config.model_kwargs["pretrained_backbone_path"],
        )

    return model
