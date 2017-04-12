# hopfield
Implementation of Hopfield networks utilizing the GPU

## Setup info
Due to the use of C++11 features, gcc 5+ and CUDA 8.0+ must be used.  This requires setting the  `LD_LIBRARY_PATH` environment variable so that the correct shared libraries can be dynamicly linked.
On ECE GPU lab machines, gcc 5.2 can be found in `/opt/rh/devtoolset-4/root/usr/bin/`, which should be added to your PATH so that the correct version is used by make.  
This can be done automaticly by adding the lines
```
set path=(/opt/rh/devtoolset-4/root/usr/bin $path)

if ($?LD_LIBRARY_PATH) then
 setenv LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
else
 setenv LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64
endif
```
to .cshrc or
```
export PATH=/opt/rh/devtoolset-4/root/usr/bin:$PATH

if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
fi
```
to .bashrc

