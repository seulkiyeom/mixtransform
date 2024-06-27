import re
import torch.nn as nn
from typing import Callable, Union
from collections import OrderedDict

from tools.prune_tools.arch_modif import prune_layer
from tools.prune_tools.utils import set_in_index_attr_test

from mmengine.model import ModuleList, is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model

class CustomRunner(Runner):
    # def load_pruned_model(self, arch_info=None):
    def load_pruned_model(self):
        checkpoint = _load_checkpoint(self._load_from)['state_dict']
        index_stack = []
        for name_module, module in self.model.named_modules():
            if 'mixer.m.attn.mix' in name_module and not isinstance(module, ModuleList):
                index_stack += range(checkpoint[name_module + '.weight'].size(0))
                prune_layer(self.model, name_module, range(checkpoint[name_module + '.weight'].size(0)), both_prune=True)
                prune_FFN = True
                # checkpoint[name_module + '.weight'].size(0)

            elif 'proj' in name_module and isinstance(module, nn.Conv2d) and prune_FFN is True:
                prune_layer(self.model, name_module, range(len(index_stack)), both_prune=False)
                prune_FFN = False
                index_stack = []

    def load_checkpoint(self,
                    filename: str,
                    map_location: Union[str, Callable] = 'cpu',
                    strict: bool = False,
                    revise_keys: list = [(r'^module.', '')]):
        """Load checkpoint from given ``filename``.

        Args:
            filename (str): Accept local filepath, URL, ``torchvision://xxx``,
                ``open-mmlab://xxx``.
            map_location (str or callable): A string or a callable function to
                specifying how to remap storage locations.
                Defaults to 'cpu'.
            strict (bool): strict (bool): Whether to allow different params for
                the model and checkpoint.
            revise_keys (list): A list of customized keywords to modify the
                state_dict in checkpoint. Each item is a (pattern, replacement)
                pair of the regular expression operations. Defaults to strip
                the prefix 'module.' by [(r'^module\\.', '')].
        """
        checkpoint = _load_checkpoint(filename, map_location=map_location)

        # Add comments to describe the usage of `after_load_ckpt`
        self.call_hook('after_load_checkpoint', checkpoint=checkpoint)

        if is_model_wrapper(self.model):
            model = self.model.module
        else:
            model = self.model

        checkpoint = _load_checkpoint_to_model(
            model, checkpoint, strict, revise_keys=revise_keys)

        unpruned_indices = checkpoint['unpruned_indices'].copy()
        for p, r in revise_keys:
            unpruned_indices = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in unpruned_indices.items()})
            
        for k, v in unpruned_indices.items():       
            set_in_index_attr_test(model, k, v)

        self._has_loaded = True

        self.logger.info(f'Load checkpoint from {filename}')

        return checkpoint
