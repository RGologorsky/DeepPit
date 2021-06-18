# Developed downstream of parent CUDA containers.
# Can run from cmd line like so: "docker build -t nvcr.io/nvidia/pytorch:21.03-py3 -f Dockerfile ."
# To run: docker run --gpus all -it --rm -v /home/gologr01/Desktop/DeepPit/:/home -p 6006:6006 nvcr.io/nvidia/pytorch:21.03-py3 

ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:21.02-py3
FROM $BASE_CONTAINER

# To enable Jupyterlab
# Always to pip last

EXPOSE 6006
EXPOSE 8888 
RUN pip install --no-cache-dir \
    'simpleitk' \
    'meshio' \
    'gputil' \
    'pandas' \
    'fastai' \

WORKDIR "../home"

# docker run --gpus all -it --rm -v /home/gologr01/Desktop/DeepPit/:/home -p 8888:8888 nvcr.io/nvidia/pytorch:21.03-py3

# Developed downstream of parent CUDA containers.
# Can run from cmd line like so: "docker build -t nec4brain/nec4brain:v1 -f Dockerfile.dockerfile ."
# To run: docker run --gpus all -it --rm -v /home/olab-harvey2/:/home -p 6006:6006 nec4brain/nec4brain:v1ARG BASE_CONTAINER=nvcr.io/nvidia/pytorch:21.02-py3

    
# name: py36_grace
# channels:
#  - defaults
# dependencies:
#  - cudatoolkit=10.1
#  - torchvision==0.8.2
#  - torchaudio==0.7.2
#  - pytorch==1.7.1
#  - simpleitk
#  - meshio
#  - gputil
#  - pandas
#  - fastai
#  - pyzmq
#  - notebook
#  - jupyter_contrib_nbextensions

