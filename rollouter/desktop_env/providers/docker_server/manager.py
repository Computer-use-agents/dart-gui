import os
import platform
import zipfile
import subprocess

from time import sleep
import requests
from tqdm import tqdm

import logging

from desktop_env.providers.base import VMManager

logger = logging.getLogger("desktopenv.providers.docker.RemoteDockerVMManager")
logger.setLevel(logging.INFO)


class DockerVMManager(VMManager):
    def __init__(self, registry_path=""):
        pass

    def add_vm(self, vm_path):
        pass

    def check_and_clean(self):
        pass

    def delete_vm(self, vm_path):
        pass

    def initialize_registry(self):
        pass

    def list_free_vms(self):
        return ""

    def occupy_vm(self, vm_path):
        pass

    def get_vm_path(self, os_type, region,screen_size):
        return ""
