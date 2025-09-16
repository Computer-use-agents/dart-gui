
wget  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb   

dpkg -i cuda-keyring_1.1-1_all.deb  

apt-get update 

apt-get install -y libnccl2=2.25.1-1+cuda12.4 libnccl-dev=2.25.1-1+cuda12.4

apt-get install -y librdmacm1 libibverbs1 ibverbs-providers

apt install infiniband-diags