# test_cg

Code for [question](https://stackoverflow.com/questions/53492528/cooperative-groupsthis-grid-causes-any-cuda-api-call-to-return-unknown-erro)

Using CUDA 10.0 (or 9.1) on NVIDIA TITAN V GPU (Driver Version 410.73) attached to a server running Ubuntu 16.04.5 LTS, the error can be reproduce using the following commands: 

```
git clone https://github.com/Ahdhn/test_cg.git
cd test_cg
mkdir build
cd build
cmake ../ 
make
```
