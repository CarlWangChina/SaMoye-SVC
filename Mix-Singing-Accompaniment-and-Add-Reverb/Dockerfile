ARG CUDA_VERSION=12.1
# ARG from=pytorch/pytorch:2.1.2-cuda${CUDA_VERSION}-cudnn8-runtime
ARG from=pytorch/pytorch:2.1.2-cuda${CUDA_VERSION}-cudnn8-devel
FROM ${from} as base

ARG from
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
# 设置工作目录
WORKDIR /app

# 复制项目文件到镜像中的工作目录
COPY . /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt &&\
    apt-get update &&\
    apt-get install -y libasound2-dev libfreetype6-dev libgl1-mesa-dev libcurl4-openssl-dev libwebkit2gtk-4.0-dev libgtk-3-dev &&\
    cd effector && cmake -S . -B build && cmake --build build



# 启动应用程序
CMD ["python", "app.py"]

