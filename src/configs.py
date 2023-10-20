from pathlib import Path
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os

config_path = Path('.') / 'configs/development_config.yml'

class Config(object):
    def __init__(self):
        # Config Flask
        configs = load(open(config_path, 'r'), Loader=Loader)
        self.DEBUG = configs['DEBUG']
        self.TESTING = configs['TESTING']
        
        self.USE_CUDA = configs['USE_CUDA']
        self.DEVICE_ID = configs['DEVICE_ID']
    