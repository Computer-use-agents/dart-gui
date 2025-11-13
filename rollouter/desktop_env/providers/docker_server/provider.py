import logging
import os
import platform
import time
import docker
import psutil
import requests
from filelock import FileLock
from pathlib import Path
from urllib.parse import urlparse

from desktop_env.providers.base import Provider
from desktop_env.utils import load_config

config = load_config()
logger = logging.getLogger("desktopenv.providers.docker.DockerProvider")
logger.setLevel(logging.INFO)

WAIT_TIME = 3
RETRY_INTERVAL = 1
LOCK_TIMEOUT = 10


class PortAllocationError(Exception):
    pass


class RemoteDockerProvider(Provider):
    def __init__(self, region: str, remote_docker_server_ip: str = config.remote_docker_server.ip, remote_docker_server_port: int = config.remote_docker_server.port):
        # self.client = docker.from_env()
        # remote_docker_server_ip = '10.1.110.48'
        print('remote docker server ip',remote_docker_server_ip)
        self.server_port = None
        self.vnc_port = None
        self.chromium_port = None
        self.vlc_port = None
        self.container = None
        self.emulator_id = None
        self.environment = {"DISK_SIZE": "2G", "RAM_SIZE": "2G", "CPU_CORES": "2"}  # Modify if needed
        self.remote_docker_server_ip = remote_docker_server_ip
        self.remote_docker_server_port = remote_docker_server_port
        # Allow override via environment variables (preferred when runner passes --base-url)
        base_url = os.getenv("OSWORLD_BASE_URL")
        if base_url:
            try:
                u = urlparse(base_url)
                if not u.scheme:
                    u = urlparse("http://" + base_url)
                host = u.hostname or self.remote_docker_server_ip
                port = u.port or self.remote_docker_server_port
                self.remote_docker_server_ip = host
                self.remote_docker_server_port = int(port)
            except Exception:
                pass
        else:
            ip_env = os.getenv("OSWORLD_REMOTE_DOCKER_IP")
            port_env = os.getenv("OSWORLD_REMOTE_DOCKER_PORT")
            if ip_env:
                self.remote_docker_server_ip = ip_env
            if port_env:
                try:
                    self.remote_docker_server_port = int(port_env)
                except Exception:
                    pass
        # token for quota/auth; prefer env OSWORLD_TOKEN, fallback to config.client_token if present
        self.token = os.getenv("OSWORLD_TOKEN")
        temp_dir = Path(os.getenv('TEMP') if platform.system() == 'Windows' else '/tmp')
        self.lock_file = temp_dir / "docker_port_allocation.lck"
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _get_used_ports(self):
        """Get all currently used ports (both system and Docker)."""
        # Get system ports
        system_ports = set(conn.laddr.port for conn in psutil.net_connections())
        
        # Get Docker container ports
        docker_ports = set()
        for container in self.client.containers.list():
            ports = container.attrs['NetworkSettings']['Ports']
            if ports:
                for port_mappings in ports.values():
                    if port_mappings:
                        docker_ports.update(int(p['HostPort']) for p in port_mappings)
        
        return system_ports | docker_ports

    def _get_available_port(self, start_port: int) -> int:
        """Find next available port starting from start_port."""
        used_ports = self._get_used_ports()
        port = start_port
        while port < 65354:
            if port not in used_ports:
                return port
            port += 1
        raise PortAllocationError(f"No available ports found starting from {start_port}")

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for VM to be ready by checking screenshot endpoint."""
        start_time = time.time()
        
        def check_screenshot():
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10)
                )
                return response.status_code == 200
            except Exception:
                return False

        while time.time() - start_time < timeout:
            if check_screenshot():
                return True
            logger.info("Checking if virtual machine is ready...")
            time.sleep(RETRY_INTERVAL)
        
        raise TimeoutError("VM failed to become ready within timeout period")

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str):
        """
        Start emulator via remote docker server. Send token for per-token quota control if available.
        """
        url = f"http://{self.remote_docker_server_ip}:{self.remote_docker_server_port}/start_emulator"
        # Determine token: instance attr, env, or config.client_token
        token = self.token or os.getenv("OSWORLD_TOKEN")
        try:
            if token:
                headers = {"Authorization": f"Bearer {token}"}
                resp = requests.post(url, json={"token": str(token)}, headers=headers)
            else:
                resp = requests.get(url)
            data = resp.json()
            # Validate response
            if resp.status_code != 200 or data.get("code") != 0 or "data" not in data:
                raise RuntimeError(f"Failed to start emulator: status={resp.status_code}, payload={data}")
            self.emulator_id = data["data"]["emulator_id"]
            self.server_port = data["data"]["server_port"]
            self.vnc_port = data["data"]["vnc_port"]
            self.chromium_port = data["data"]["chromium_port"]
            self.vlc_port = data["data"]["vlc_port"]
        except Exception as e:
            raise e

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("VM not started - ports not allocated")
        return f"{self.remote_docker_server_ip}:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for Docker provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def stop_emulator(self, path_to_vm: str):
        if self.emulator_id:
            logger.info("Stopping VM...")
            try:
                response = requests.post(f"http://{self.remote_docker_server_ip}:{self.remote_docker_server_port}/stop_emulator", json={"emulator_id": self.emulator_id})
            except Exception as e:
                logger.error(f"Error stopping container: {e}")
            finally:
                self.container = None
                self.server_port = None
                self.vnc_port = None
                self.chromium_port = None
                self.vlc_port = None
