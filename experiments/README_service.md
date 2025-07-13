# Service for sovits svc 5.0

How to pull docker image


## pull pytorch image
```
docker pull docker.m.daocloud.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

```

## build image
```
docker build -f Dockerfile_gpu  -t sovits5_panxin:0.1 .
```

## debug container
```
docker run -it --gpus all --rm --name tmprm sovits5_panxin:0.1 bash
```