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

## Compilation
To compile the code, run `make` at the top-level of the repository (or `make -j` for a faster build).  This will compile everything and place the executables in the `bin/release` folder.  

## Execution
The following executables that have been built may be run:
* `mincut_driver`: Runs the mincut algorithm on random inputs.  Usage: `<num vertices> <evaluation method>`
* `queens_driver`: Runs n-queens for a given size.  Usage: `./bin/release/queens_driver N gamma threshold`
* `test_driver`: Runs a test of associative memory using random data.  Usage: `./bin/release/test_driver <evaluation algorithm> <data size>(=10000) <# of data vectors>(=100) <fraction of vector included in key>(=0.25)`
Several other execuatables are also built, used for generating test data.  These may be ignored.  
