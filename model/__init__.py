import os
from importlib import import_module

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, ckp):
        super(Model, self).__init__()
        print('Model: Begin making model')
        self.args = args
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs

        # default import model.cdvd_tsp
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args).to(self.device)
        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        # load model checkpoint
        self.load(
            ckp.dir,
            pre_train=args.pre_train,
            resume=args.resume,
            cpu=args.cpu
        )

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        if not self.cpu and self.n_GPUs > 1:
            return self.model.module
        else:
            return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, is_best=False, epoch=None):
        # Save latest model
        if epoch is not None:
            target = self.get_model()
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_{}.pt'.format(epoch))
            )
        else:
            target = self.get_model()
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_latest.pt')
            )

        # Save best model
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model', 'model_best.pt')
            )

    def load(self, apath, pre_train='.', resume=False, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        # Load pretrained model
        if pre_train != '.':
            if isinstance(pre_train, str):
                print('Model: loading model from {}'.format(pre_train))

                pretrained_dict = torch.load(pre_train, **kwargs)
                model_dict = self.get_model().state_dict()

                initialized = []
                for k, v in pretrained_dict.items():
                    if k in model_dict and v.size() == model_dict[k].size():
                        model_dict[k] = v
                        initialized.append(k)

                not_initialized = []
                for k, v in model_dict.items():
                    if k not in initialized:
                        not_initialized.append(k)

                print('The following items are not initialized:')
                print(sorted(not_initialized))
                self.get_model().load_state_dict(model_dict)
            elif isinstance(pre_train, list):
                initialized = []
                model_dict = self.get_model().state_dict()
                for save_path in pre_train:
                    print('Model: loading model from {}'.format(save_path))
                    pretrained_dict = torch.load(save_path, **kwargs)

                    for k, v in pretrained_dict.items():
                        if k in model_dict and v.size() == model_dict[k].size():
                            model_dict[k] = v
                            initialized.append(k)

                not_initialized = []
                for k, v in model_dict.items():
                    if k not in initialized:
                        not_initialized.append(k)

                print('The following items are not initialized:')
                print(sorted(not_initialized))
                self.get_model().load_state_dict(model_dict)

        # Resume model
        elif resume and not self.args.test_only:
            print('Model: loading model from {}'.format(os.path.join(apath, 'model', 'model_latest.pt')))
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_latest.pt'), **kwargs),
                strict=False
            )

        # Load best model when testing
        elif self.args.test_only:
            print('Model: loading model from {}'.format(os.path.join(apath, 'model', 'model_best.pt')))
            self.get_model().load_state_dict(
                torch.load(os.path.join(apath, 'model', 'model_best.pt'), **kwargs),
                strict=False
            )
        else:
            pass
