# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import logging

__all__ = [logger]

class logger_info():
  kk=logging

class logger:

    def get_logger():

logging.basicConfig(
    filename='output.log',
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
)

logging.getLogger(__name__)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('output.log')
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',datafmt="%Y%m%d")

handler.setFormatter(formatter)

logger.addHandler(handler)


