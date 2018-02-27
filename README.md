# Demo of using CUDA to accelerate parallelizable Python functions

The following has been cobbled together and streamlined from the following resources:

[GPU Accelerated Computing with Python (from NVIDIA website)](https://developer.nvidia.com/how-to-cuda-python)

[Installing GPU drivers using scripts (from Google Cloud Platform website)](https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script)

[NVIDIA developer help forum](https://devtalk.nvidia.com/default/topic/995277/cuda-8-0-toolkit-install-nvcc-not-found-ubuntu-16-04/)

## Getting started: running a minimal CUDA program

These instructions assume you're starting from a fresh installation of Ubuntu with CUDA-enabled GPUs, as you might be after spinning up a virtual machine on Google Cloud Platform. Everything that follows uses Python 3.

### Install Anaconda

Download:

```wget http://repo.continuum.io/archive/Anaconda3-4.0.0-Linux-x86_64.sh```

Install:

```bash Anaconda3-4.0.0-Linux-x86_64.sh```

Note: make sure to enter 'yes' for final prompt asking to Anaconda location to PATH.

Run commands added to .bashrc file to start using right away:

```source ~/.bashrc```

Make sure conda is up-to-date:

```conda upgrade conda```

### Install cudatoolkit for Python

```conda install cudatoolkit```

### Install CUDA:

Go to [https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script](https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script) and scroll down to "Installing GPU drivers using scripts"

Find the correct CUDA-installation script for your operating system. E.g., for Ubuntu 16.04 LTS, this is

```
#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-8-0; then
  # The 16.04 installer works with 16.10.
  curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda-8-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1
```

Copy and paste this script into a file called `install_cuda.sh`. Run:

```sudo bash install_cuda.sh```

which will probably take a few minutes. This installs CUDA.

Verify the installation by running:

```nvidia-smi```

which should list the CUDA-enabled GPUs on your machine.

### Test

Clone this repository onto your machine:

```git clone https://github.com/rkp8000/python_cuda_demo```

To make sure the CUDA drivers and Python interface were installed correctly, and to run the same vector addition function parallelized on a GPU, run

```python vector_add_gpu.py```

If this runs with no errors it means that everything is installed correctly.

To get a benchmark for adding two vectors using the CPU, run 

```python vector_add_cpu.py```

which should take much longer.

### Test inside a Jupyter notebook

Start a Jupyter notebook server:

```jupyter notebook --port=9999```

Navigate to the notebook URL, e.g.: 'localhost:9999`.

Run the notebook `demo.ipynb`.

The notebook will run first run a vector-addition function using the machine's CPUs, and then will run it using the GPUs. If the notebook runs successfully and prints out time measurements for each function, then congratulations, you have successfully accelerated your first parallelizable Python function with CUDA!

For more details on how the Python code works, see the NVIDIA-published video: [Your First CUDA Python Program](https://www.youtube.com/watch?v=dPQnFXD7DxM).

