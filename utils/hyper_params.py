# !/usr/bin/env python
# coding=utf-8
# @Time    : 2019-07-04 11:12
# @Author  : yingyuankai@aliyun.com
# @File    : hparams.py

import os
import ast
import re
import yaml
import json
import logging
import collections
import argparse
import traceback

from ast import literal_eval


__all__ = [
    "HyperParams",
]

logger = logging.getLogger(__name__)

class HyperParams(collections.OrderedDict):
    def __init__(self, init_dict={}):
        self.init(init_dict)
    def init(self, init_dict):
        super().__init__(init_dict)
        # 可以像访问属性一样，访问key-value
        for key in self:
            if isinstance(self[key], collections.abc.Mapping):
                self[key] = HyperParams(self[key])
            elif isinstance(self[key], list):
                for idx, val in enumerate(self[key]):
                    if isinstance(val, collections.abc.Mapping):
                        self[key][idx] = HyperParams(val)
    
    def _decode_val(self, value):
        if not isinstance(value, str):
            return value
        if value == 'None':
            value = None
        try:
            print(value)
            value = literal_eval(value) 
            print(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value
    
    def __setattr__(self, key, value):
        self[key] = value
        if isinstance(self[key], collections.abc.Mapping):
                self[key] = HyperParams(self[key])
        elif isinstance(self[key], list):
                for idx, val in enumerate(self[key]):
                    if isinstance(val, collections.abc.Mapping):
                        self[key][idx] = HyperParams(val)

    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]

    def load_config(self, config_path='configs/base_config.yaml'):
        import logging.config
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                log_config = yaml.load(f)
                self.init(log_config)
                merge_to = {}
                for k, v in log_config.items():
                    merge_to[k] = self._decode_val(v) 
                
                self.init(merge_to)
                logging.config.dictConfig(self.logging)


