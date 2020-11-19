Build docker image for NARS
================================

Build image
---------------
```bash
docker build -t nars-cuda -f Dockerfile .
```

Run container
--------------
```bash
nvidia-docker run -it --name nars nars-cuda
```
