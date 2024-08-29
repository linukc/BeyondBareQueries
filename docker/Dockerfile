FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

SHELL [ "/bin/bash", "-c" ]
ENV DEBIAN_FRONTEND noninteractive
ENV USER=docker_user
ARG UID=1000
ARG GID=1000

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo git wget nano unzip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN conda create -n bbq_env python=3.10
RUN conda install -y -n bbq_env pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda install -y -n bbq_env -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
RUN conda install -y -n bbq_env \
    https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
RUN conda run -n bbq_env python -m pip install h5py hydra-core open_clip_torch supervision loguru wget iopath \
    torchpack pyyaml tqdm opencv-python natsort imageio onnxruntime open3d==0.17.0 fast-pytorch-kmeans
RUN conda run -n bbq_env python -m pip install --extra-index-url=https://pypi.nvidia.com cuml-cu11==24.8.*

WORKDIR /home/${USER}

RUN git clone https://github.com/krrish94/chamferdist.git && cd chamferdist && sed -i 's/c++14/c++17/' setup.py &&\
    git checkout ee753899b583bdc06bd2505dc162fd54ba0c6c5b && conda run -n bbq_env python -m pip install .

RUN git clone https://github.com/gradslam/gradslam.git && cd gradslam && git checkout conceptfusion && \
    git checkout 59ca872e3d265ad09f63c4793d011fad67064452 && conda run -n bbq_env python -m pip install .

RUN git clone https://github.com/IDEA-Research/Grounded-Segment-Anything && cd Grounded-Segment-Anything && \
    git checkout 2b1b72eb1799facb0cbdc7daf3af133922b1d51b && cd segment_anything && \
    conda run -n bbq_env python -m pip install -e .

RUN git clone https://github.com/ChaoningZhang/MobileSAM && cd MobileSAM && \
    git checkout c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed && conda run -n bbq_env python -m pip install -e .
ENV PYTHONPATH=/home/${USER}/MobileSAM/MobileSAMv2:$PATH
RUN conda run -n bbq_env python -m pip install onnx onnxsim gdown
RUN gdown 1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE -O /home/docker_user/bbq_weights/mobilesamv2/ && \
    unzip /home/docker_user/bbq_weights/mobilesamv2/weight.zip -d /home/docker_user/bbq_weights/mobilesamv2/ && \
    cp /home/docker_user/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt \
    /home/docker_user/bbq_weights/mobilesamv2/weight/

RUN git clone https://github.com/haotian-liu/LLaVA.git && cd LLaVA && \
    git checkout c121f0432da27facab705978f83c4ada465e46fd && conda run -n bbq_env python -m pip install -e .

RUN useradd -m ${USER} --uid=${UID} \
    && usermod -s /bin/bash ${USER} \
    && usermod -a -G sudo ${USER} \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
WORKDIR /home/${USER}/BeyondBareQueries
RUN chown -R $USER /home/${USER}
USER ${USER}

RUN echo 'source activate bbq_env' >> /home/${USER}/.bashrc
