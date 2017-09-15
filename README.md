# CUDA

# Lesson 1 (Technology trends, throughput vs latency, GPU design goals, GPU programming model, MAP)

## Core GPU Design Tenets:
1. lots of simple compute units trade simple control for more compute
2. explicitly parallel programming model
3. optimize for throughput not latency (prioritize throughout)

## importance of programming in parallel

## CUDA program
- CPU(host)
- GPU(device)
1. data cpu -> GPU
2. data gpu -> cpu
1,2 : cuda Memcpy
3. allocate GPC memory
3: cudaMelloc
4. launch kernel on GPU (HOST launches kernels on device)

## A typical GPU programming (GOOD FOR LOTS OF COMPUTATION BUT NOT FOR SMALL JOBS DUE TO LOTS OF DATA TRANSFER)
1. CPU allocates storage on GPU (cuda Melloc)
2. CPU copies input DATA from CPU -> GPU (cuda Memcpy)
3. CPU launches kernels on GPU to process the data (kernel launch)
4. CPU copies results back to CPU from GPU (cuda Memcpy)

## Define the GPU computation
- Kernels look like serial programs. Write your program as if it will run on *one* thread
- GPU will run that program on *many* threads. We can launch the program on any number of threads
- Each thread knows its own index in the block and the grid

## What is GPU good at?
1. efficiently launching lots of threads
2. running lots of threads in parallel


## A high level view of GPU
CPU : allocates memory, copy data to/from GPU, launch kernel(open how many threads for GPU)
GPU : express out = in * in (square function example)

## Code Convention
data on CPU, starts with h_
data on GPU, starts with d_
cudaMalloc : allocate data on GPU
Malloc: allocate data on CPU
```C
//transfer the array to GPU
cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

//launch the kernel (indicates by <<< >>>) call only call kernel on GPU data
cube<<<1, ARRAY_BYTES>>>(d_out, d_in);

//copy back the result array to GPU
cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

//free GPU memeory allocaton
cudaFree(d_in);
cudaFree(d_out);
```

```C
//kernel
_global_ void square(float *d_out, float *d_in){
  int idx = threadIdx.x;
  float f = d_in[idx];
  d_out[idx] = f*f;
}
```

## Configuring the kernel launch
cube<<<1, 64>>>(d_out, d_in)
1: number of blocks, 64: threads for block

1. can run many blocks at  once
2.  maximum number of threads/block (512 older gpu, 1024 newer gpu)

- Therefore, 1280 threads? -> cube<<<10, 128>>>(..., ...)
- We can launch 1,2,3D blocks and threads.

*kernel<<<Grid of blocks, Block of threads>>>(....)*
*kernel<<<dim3(tx,ty,tz), dim3(tx,ty,tz)>>>(....)*
cube<<<1, 64>>>(d_out, d_in) === cube<<<dim3(1,1,1), dim3(64,1,1)>>>(d_out, d_in)

## Map (Key building block of GPU programming)
- set of elements to process [64 floats]
- function to run on each element ["square"]
*MAP(ELEMENTS, FUNCTION)*

## GPU are good at map for 2 reasons
- GPUs have many parallel processors
- GPUs optimize for throughput

## Project 1 : color image -> grayscale (by using MAP)
