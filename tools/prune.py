# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from copy import deepcopy
import cv2

import mmengine
import torch
import torch.nn as nn
import numpy as np

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.device import get_device

from mmengine.analysis import get_model_complexity_info
from prune_tools.finetuner import get_modules
from prune_tools.bottleneck import BottleneckReader
from prune_tools.common import CrossEntropyLabelSmooth
from prune_tools.utils import pruning
from mmengine.model import is_model_wrapper

from mmengine.dist import master_only
from typing import Optional
import warnings
from mmengine.fileio import FileClient, join_path
import time
from mmengine.utils import apply_to, get_git_hash
from mmengine.runner.checkpoint import save_checkpoint, weights_to_cpu
from mmengine.optim import OptimWrapper
from mmengine.model import ModuleList


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='the file to output results.')
    parser.add_argument(
        '--out-item',
        choices=['metrics', 'pred'],
        help='To output whether metrics or predictions. '
        'Defaults to output predictions.')
    parser.add_argument(
        '--pr-ratio',
        type=float,
        default=0.3,
        help='Define pruning ratio.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision test')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to enable the Test-Time-Aug (TTA). If the config file '
        'has `tta_pipeline` and `tta_model` fields, use them to determine the '
        'TTA transforms and how to merge the TTA results. Otherwise, use flip '
        'TTA by averaging classification score.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # enable automatic-mixed-precision test
    if args.amp:
        cfg.test_cfg.fp16 = True

    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- TTA related args --------------------
    if args.tta:
        if 'tta_model' not in cfg:
            cfg.tta_model = dict(type='mmpretrain.AverageClsScoreTTA')
        if 'tta_pipeline' not in cfg:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            cfg.tta_pipeline = deepcopy(test_pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [test_pipeline[-1]],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # ----------------- Default dataloader args -----------------
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
    )

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False

    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def select_index_flops(attribution_score, target_flops, r):
    """
        Args:
            - attribution_score: attribution score for each filter in each layer (list of list)
            - target_flops: target flops for the pruned model 
            - r: BottleneckReader
    """
    with torch.no_grad():
        # 1. we find a threshold to have the right number of flops, using dychotomy
        print(f'Looking for optimal threshold...')
        
        thres = 0.5
        delta = 0.25

        attrib = [[1 if e>thres else 0 for e in l] for l in attribution_score]
        base_flops = r.base_flops
        flops = base_flops
        iteration = 0
        # while abs(flops-target_flops)>50000 and iteration<50: #이거 삭제해도 되나? (즉, 0.5 기준으로 quantification 가능?)
        #     print(f'Testing threshold {thres}')
        #     attrib = [[1 if e>thres else 0 for e in l] for l in attribution_score]
        #     # make sure that nothing is 100% pruned
        #     for i in range(len(attrib)):
        #         if sum(attrib[i])==0:
        #             attrib[i][np.argmax(attribution_score[i])] = 1

        #     # pseudo-prune model with attrib
        #     r.update_alpha_with(attrib)
        #     flops = base_flops + r.compute_flops()

        #     print(f'Distance to target: {int(abs(flops-target_flops)):,}')
        #     if flops > target_flops: thres += delta
        #     else: thres -= delta
        #     delta /= 2
        #     iteration +=1
        # 2. once we found the right threshold, we select the indexes to prune
        from itertools import groupby
        preserved_indexes_all = [[bool(e) for e in l] for l in attrib]
        preserved_indexes_all = [[j,i] for j in range(len(preserved_indexes_all)) for i in range(len(preserved_indexes_all[j])) if preserved_indexes_all[j][i]]
        preserved_indexes_all = [[i[1] for i in e] for _,e in groupby(preserved_indexes_all, lambda x: x[0])]

        return preserved_indexes_all

class CustomRunner(Runner):
    def prune_model(self, pr_ratio): #현재 pr_ratio 사용하지 않고 있음 나중에 수정 필요
        #hyper-parameters
        nb_batches = 200
        beta = 6
        CLASSES = 1000 #label_smooth: 0.1
        criterion = CrossEntropyLabelSmooth(CLASSES, 0.1).cuda()

        input_shape = (3, 224, 224)
        device = get_device()
        analysis_results = get_model_complexity_info(self.model, input_shape=input_shape)
        maxflops = analysis_results['flops']
        targetflops = 165.65 * 10**6 #이거 나중에 수정 필요
        print(f"Pruning ratio: {(1-(targetflops/maxflops)):.2f}: Current FLOPs: {maxflops:.0f}, Target FLOPs: {targetflops:.0f}")

        modules, init_lamb = get_modules(self.model, device)
        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model
        reader = BottleneckReader(model, criterion, self.train_dataloader, modules.modules(), init_lamb, device, maxflops, targetflops, steps = nb_batches, beta=beta) 
        attribution_score = reader.get_attribution_score()

        # import pickle
        # # with open('outfile', 'wb') as fp: #when saving
        # #     pickle.dump(self.best_attribution_score, fp)
        # with open ('outfile', 'rb') as fp: #when loading
        #     attribution_score = pickle.load(fp)

        # select the indexes to preserve
        preserved_indexes_all = select_index_flops(attribution_score, targetflops, reader)
        
        attrib_list_str = "attribution_score[0:12]: \n"
        for j in range(reader.unique_alphas.len()):
            tmp = reader.unique_alphas.get_lambda(j).detach().clone().cpu().numpy()[0:12]
            attrib_list_str += ('[ ' + ' '.join("{:.2f} ".format(lmbd) for lmbd in tmp) + ']\n')
        print(f'{attrib_list_str}')

        reader.remove_layer()

        #pruning
        pruning(model, preserved_indexes_all)
        print("Model pruned...")

    @master_only
    def save_checkpoint(
        self,
        out_dir: str,
        filename: str,
        file_client_args: Optional[dict] = None,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        meta: Optional[dict] = None,
        by_epoch: bool = True,
        backend_args: Optional[dict] = None,
    ):
        """Save checkpoints.

        ``CheckpointHook`` invokes this method to save checkpoints
        periodically.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename (str): The checkpoint filename.
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. See :class:`mmengine.fileio.FileClient` for
                details. Defaults to None. It will be deprecated in future.
                Please use `backend_args` instead.
            save_optimizer (bool): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            save_param_scheduler (bool): Whether to save the param_scheduler
                to the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            by_epoch (bool): Decide the number of epoch or iteration saved in
                checkpoint. Defaults to True.
            backend_args (dict, optional): Arguments to instantiate the
                prefix of uri corresponding backend. Defaults to None.
                New in v0.2.0.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')

        if by_epoch:
            # self.epoch increments 1 after
            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
            # called by `after_train_epoch`` method of `CheckpointHook` so
            # `epoch` should be `self.epoch + 1`
            meta.setdefault('epoch', self.epoch + 1)
            meta.setdefault('iter', self.iter)
        else:
            meta.setdefault('epoch', self.epoch)
            meta.setdefault('iter', self.iter + 1)

        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set at '
                    'the same time.')

            file_client = FileClient.infer_client(file_client_args, out_dir)
            filepath = file_client.join_path(out_dir, filename)
        else:
            filepath = join_path(  # type: ignore
                out_dir, filename, backend_args=backend_args)

        meta.update(
            cfg=self.cfg.pretty_text,
            seed=self.seed,
            experiment_name=self.experiment_name,
            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            mmengine_version=mmengine.__version__ + get_git_hash())

        if hasattr(self.train_dataloader.dataset, 'metainfo'):
            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = {
            'meta':
            meta,
            'state_dict':
            weights_to_cpu(model.state_dict()),
            'message_hub':
            apply_to(self.message_hub.state_dict(),
                     lambda x: hasattr(x, 'cpu'), lambda x: x.cpu()),
        }
        # save optimizer state dict to checkpoint
        if save_optimizer:
            if isinstance(self.optim_wrapper, OptimWrapper):
                checkpoint['optimizer'] = apply_to(
                    self.optim_wrapper.state_dict(),
                    lambda x: hasattr(x, 'cpu'), lambda x: x.cpu())
            else:
                raise TypeError(
                    'self.optim_wrapper should be an `OptimWrapper` '
                    'or `OptimWrapperDict` instance, but got '
                    f'{self.optim_wrapper}')

        # save param scheduler state dict
        if save_param_scheduler and self.param_schedulers is None:
            self.logger.warning(
                '`save_param_scheduler` is True but `self.param_schedulers` '
                'is None, so skip saving parameter schedulers')
            save_param_scheduler = False
        if save_param_scheduler:
            if isinstance(self.param_schedulers, dict):
                checkpoint['param_schedulers'] = dict()
                for name, schedulers in self.param_schedulers.items():
                    checkpoint['param_schedulers'][name] = []
                    for scheduler in schedulers:
                        state_dict = scheduler.state_dict()
                        checkpoint['param_schedulers'][name].append(state_dict)
            else:
                checkpoint['param_schedulers'] = []
                for scheduler in self.param_schedulers:  # type: ignore
                    state_dict = scheduler.state_dict()  # type: ignore
                    checkpoint['param_schedulers'].append(state_dict)
        
        # get unpruned indice
        checkpoint['unpruned_indices'] = dict()
        for name, module in model.named_modules():
            if 'mixer.m.attn.mix' in name and not isinstance(module, ModuleList):
                checkpoint['unpruned_indices'][name] = module.in_index

            # elif 'proj' in name and isinstance(module, nn.Conv2d):
            #     print(name)

        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
        save_checkpoint(
            checkpoint,
            filepath,
            file_client_args=file_client_args,
            backend_args=backend_args)


def main():
    args = parse_args()

    if args.out is None and args.out_item is not None:
        raise ValueError('Please use `--out` argument to specify the '
                         'path of the output file before using `--out-item`.')

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = CustomRunner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.out and args.out_item in ['pred', None]:
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))

    # # runner 인스턴스에 prune 메서드를 동적으로 추가
    # runner.prune_model = prune_model.__get__(runner, Runner)

    # testing before pruning
    metrics = runner.test()

    # start pruning
    runner.prune_model(pr_ratio = args.pr_ratio)

    # testing after pruning
    runner.train()
    metrics = runner.test()



    if args.out and args.out_item == 'metrics':
        mmengine.dump(metrics, args.out)


if __name__ == '__main__':
    main()
