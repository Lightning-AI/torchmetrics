# Docker images

## Build images from Dockerfiles

You can build it on your own, note it takes lots of time, be prepared.

```bash
git clone https://github.com/Lightning-AI/torchmetrics.git

# build with the default arguments
docker image build -t torchmetrics:latest -f dockers/ubuntu-cuda/Dockerfile .

# build with specific arguments
docker image build -t torchmetrics:ubuntu-cuda11.7.1-py3.9-torch1.13 \
  -f dockers/base-cuda/Dockerfile \
  --build-arg PYTHON_VERSION=3.9 \
  --build-arg PYTORCH_VERSION=1.13 \
  --build-arg CUDA_VERSION=11.7.1 \
  .
```

To run your docker use

```bash
docker image list
docker run --rm -it torchmetrics:latest bash
```

and if you do not need it anymore, just clean it:

```bash
docker image list
docker image rm torchmetrics:latest
```

## Run docker image with GPUs

To run docker image with access to your GPUs, you need to install

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

and later run the docker image with `--gpus all`. For example,

```bash
docker run --rm -it --gpus all torchmetrics:ubuntu-cuda11.7.1-py3.9-torch1.12
```
