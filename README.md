# Awesome-CUDA-and-HPC
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

🚀🚀🚀 This repository lists some awesome public [CUDA](https://developer.nvidia.com/cuda-zone), [cuda-python](https://github.com/NVIDIA/cuda-python), [cuBLAS](https://developer.nvidia.com/cublas), [CUTLASS](https://github.com/NVIDIA/cutlass), [cuDNN](https://developer.nvidia.com/cudnn), [NCCL](https://github.com/NVIDIA/nccl), [TensorRT](https://developer.nvidia.com/tensorrt), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [Triton](https://github.com/triton-lang/triton), [TileLang](https://github.com/tile-ai/tilelang), [TVM](https://tvm.apache.org/), [MLIR](https://mlir.llvm.org/), [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and High Performance Computing (HPC) projects.

## Contents
- [Awesome-CUDA-and-HPC](#awesome-cuda-and-hpc)
  - [Official Version](#official-version)
  - [Awesome List](#awesome-list)
  - [Learning Resources](#learning-resources)
    - [CUDA Learning](#cuda-learning)
    - [TensorRT Learning](#tensorrt-learning)
    - [Triton Learning](#triton-learning)
    - [TileLang Learning](#tileLang-learning)
    - [TVM Learning](#tvm-learning)
    - [MLIR Learning](#mlir-learning)
    - [HPC Learning](#hpc-learning)
  - [Frameworks](#frameworks)
    - [CUDA Frameworks](#cuda-frameworks)
        - [GPU Interface](#gpu-interface)
            - [CPP Version](#cpp-version)
            - [Python version](#python-version)
            - [Rust Version](#rust-version)
            - [Julia Version](#julia-version)
        - [Performance Benchmark](#performance-benchmark)
        - [Scientific Computing Framework](#scientific-computing-framework)
        - [Attention and Transformer Framework](#attention-and-transformer-framework)
        - [Machine Learning Framework](#machine-learning-framework)
        - [AI Inference Framework](#ai-inference-framework)
            - [LLM Inference and Serving Engine](#llm-inference-and-serving-engine)
            - [High Performance Kernel Library](#high-performance-kernel-library)
            - [C Implementation](#c-implementation)
            - [CPP Implementation](#cpp-implementation)
            - [Mojo Implementation](#mojo-implementation)
            - [Rust Implementation](#rust-implementation)
            - [zig Implementation](#zig-implementation)
            - [Go Implementation](#go-implementation)
        - [Distributed and Multi-GPU Framework](#distributed-and-multi-gpu-framework)
        - [Robotics Framework](#robotics-framework)
        - [ZKP and Web3 Framework](#zkp-and-web3-framework)
    - [Triton Frameworks](#triton-frameworks)
        - [Triton Machine Learning Framework](#triton-machine-learning-framework)
        - [Triton High Performance Kernel Library](#triton-high-performance-kernel-library)
    - [MLIR Frameworks](#mlir-frameworks)
        - [MLIR GPU Programming](#mlir-gpu-programming)
        - [MLIR FFI Bindings](#mlir-ffi-bindings)
        - [MLIR Machine learning Framework](#mlir-machine-learning-framework)
    - [HPC Frameworks](#hpc-frameworks)
  - [Applications](#applications)
    - [CUDA Applications](#cuda-applications)
        - [Image Preprocess](#image-preprocess)
        - [Object Detection](#object-detection)
        - [Signal Processing](#signal-processing)
        - [Mesh Processing](#mesh-processing)
        - [Graph Analytics](#graph-analytics)
  - [Blogs](#blogs)
    - [CUDA and TensorRT Blogs](#cuda-and-tensorrt-blogs)
    - [Triton Blogs](#triton-blogs)
    - [TVM Blogs](#tvm-blogs)
    - [MLIR Blogs](#mlir-blogs)
    - [HPC Blogs](#hpc-blogs)
  - [Videos](#videos)
  - [Interview](#interview)

## Official Version

  - [CUDA](https://developer.nvidia.com/cuda-zone) : CUDA is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs).

  - [NVIDIA/cuda-python](https://github.com/NVIDIA/cuda-python) <img src="https://img.shields.io/github/stars/NVIDIA/cuda-python?style=social"/> : CUDA Python: Performance meets Productivity. [nvidia.github.io/cuda-python/](https://nvidia.github.io/cuda-python/)

  - [cuBLAS](https://developer.nvidia.com/cublas) : Basic Linear Algebra on NVIDIA GPUs. NVIDIA cuBLAS is a GPU-accelerated library for accelerating AI and HPC applications. It includes several API extensions for providing drop-in industry standard BLAS APIs and GEMM APIs with support for fusions that are highly optimized for NVIDIA GPUs. The cuBLAS library also contains extensions for batched operations, execution across multiple GPUs, and mixed- and low-precision execution with additional tuning for the best performance.

  - [cuDNN](https://developer.nvidia.com/cudnn) : The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, attention, matmul, pooling, and normalization.

  - [CUTLASS](https://github.com/NVIDIA/cutlass) <img src="https://img.shields.io/github/stars/NVIDIA/cutlass?style=social"/> : CUDA Templates for Linear Algebra Subroutines. CUTLASS is a collection of CUDA C++ template abstractions for implementing high-performance matrix-matrix multiplication (GEMM) and related computations at all levels and scales within CUDA. It incorporates strategies for hierarchical decomposition and data movement similar to those used to implement [cuBLAS](https://developer.nvidia.com/cublas) and [cuDNN](https://developer.nvidia.com/cudnn).

  - [TensorRT](https://github.com/NVIDIA/TensorRT) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT?style=social"/> : NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. This repository contains the open source components of TensorRT. [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

  - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social"/> : TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines. [nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM)

  - [Triton](https://github.com/triton-lang/triton) <img src="https://img.shields.io/github/stars/triton-lang/triton?style=social"/> : Triton is a language and compiler for parallel programming. It aims to provide a Python-based programming environment for productively writing custom DNN compute kernels capable of running at maximal throughput on modern GPU hardware. [triton-lang.org/](https://triton-lang.org/)

  - [TileLang](https://github.com/tile-ai/tilelang) <img src="https://img.shields.io/github/stars/tile-ai/tilelang?style=social"/> : Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels. [tilelang.com/](https://tilelang.com/)

  - [TVM](https://github.com/apache/tvm) <img src="https://img.shields.io/github/stars/apache/tvm?style=social"/> : Open deep learning compiler stack for cpu, gpu and specialized accelerators. [tvm.apache.org/](https://tvm.apache.org/)

  - [TileLang](https://github.com/tile-ai/tilelang) <img src="https://img.shields.io/github/stars/tile-ai/tilelang?style=social"/> : Domain-specific language designed to streamline the development of high-performance GPU/CPU kernels.

  - [MLIR](https://mlir.llvm.org/) : Multi-Level Intermediate Representation Compiler Framework. The MLIR project is a novel approach to building reusable and extensible compiler infrastructure. MLIR aims to address software fragmentation, improve compilation for heterogeneous hardware, significantly reduce the cost of building domain specific compilers, and aid in connecting existing compilers together.

  - [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) : PTX, a low-level parallel thread execution virtual machine and instruction set architecture (ISA).



## Awesome List

  - [awesome-cuda-and-hpc](https://github.com/coderonion/awesome-cuda-and-hpc) <img src="https://img.shields.io/github/stars/coderonion/awesome-cuda-and-hpc?style=social"/> : some awesome public [CUDA](https://developer.nvidia.com/cuda-zone), [cuda-python](https://github.com/NVIDIA/cuda-python), [cuBLAS](https://developer.nvidia.com/cublas), [cuDNN](https://developer.nvidia.com/cudnn), [CUTLASS](https://github.com/NVIDIA/cutlass), [TensorRT](https://developer.nvidia.com/tensorrt), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [Triton](https://github.com/triton-lang/triton), [TVM](https://tvm.apache.org/), [MLIR](https://mlir.llvm.org/), [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) and High Performance Computing (HPC) projects.

  - [Erkaman/Awesome-CUDA](https://github.com/Erkaman/Awesome-CUDA) <img src="https://img.shields.io/github/stars/Erkaman/Awesome-CUDA?style=social"/> : This is a list of useful libraries and resources for CUDA development.

  - [jslee02/awesome-gpgpu](https://github.com/jslee02/awesome-gpgpu) <img src="https://img.shields.io/github/stars/jslee02/awesome-gpgpu?style=social"/> : 😎 A curated list of awesome GPGPU (CUDA/OpenCL/Vulkan) resources.

  - [mikeroyal/CUDA-Guide](https://github.com/mikeroyal/CUDA-Guide) <img src="https://img.shields.io/github/stars/mikeroyal/CUDA-Guide?style=social"/> : A guide covering CUDA including the applications and tools that will make you a better and more efficient CUDA developer.

  - [rkinas/triton-resources](https://github.com/rkinas/triton-resources) <img src="https://img.shields.io/github/stars/rkinas/triton-resources?style=social"/> : A curated list of resources for learning and exploring Triton, OpenAI's programming language for writing efficient GPU code.




## Learning Resources


  - [chenzomi12/AISystem](https://github.com/chenzomi12/AISystem) <img src="https://img.shields.io/github/stars/chenzomi12/AISystem?style=social"/> : AISystem 主要是指AI系统，包括AI芯片、AI编译器、AI推理和训练框架等AI全栈底层技术。

  - [chenzomi12/AIFoundation](https://github.com/chenzomi12/AIFoundation) <img src="https://img.shields.io/github/stars/chenzomi12/AIFoundation?style=social"/> : AIFoundation 主要是指AI系统遇到大模型，从底层到上层如何系统级地支持大模型训练和推理，全栈的核心技术。




  - ### CUDA Learning

    - [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/) : CUDA Toolkit Documentation.

    - [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) : CUDA C++ Programming Guide.

    - [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) : CUDA C++ Best Practices Guide.

    - [NVIDIA cuBLAS Programming Guide](https://docs.nvidia.com/cuda/cublas/index.html) : NVIDIA cuBLAS Documentation.

    - [NVIDIA CUTLASS Programming Guide](https://docs.nvidia.com/cutlass/index.html) : NVIDIA CUTLASS Documentation.

    - [NVIDIA NCCL Programming Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) : NVIDIA Collective Communication Library (NCCL) Documentation.

    - [NVIDIA PTX(Parallel Thread Execution) Programming Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) : NVIDIA PTX (Parallel Thread Execution) Programming Guide.

    - [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) <img src="https://img.shields.io/github/stars/NVIDIA/cuda-samples?style=social"/> : Samples for CUDA Developers which demonstrates features in CUDA Toolkit.

    - [NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples) <img src="https://img.shields.io/github/stars/NVIDIA/CUDALibrarySamples?style=social"/> : CUDA Library Samples.

    - [NVIDIA/cuda-python](https://github.com/NVIDIA/cuda-python) <img src="https://img.shields.io/github/stars/NVIDIA/cuda-python?style=social"/> : CUDA Python: Performance meets Productivity. [nvidia.github.io/cuda-python/](https://nvidia.github.io/cuda-python/)

    - [CuPy](https://github.com/cupy/cupy) <img src="https://img.shields.io/github/stars/cupy/cupy?style=social"/> : CuPy : NumPy & SciPy for GPU. [cupy.dev](https://cupy.dev/). [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/)

    - [NVIDIA-developer-blog/code-samples](https://github.com/NVIDIA-developer-blog/code-samples) <img src="https://img.shields.io/github/stars/NVIDIA-developer-blog/code-samples?style=social"/> : Source code examples from the [Parallel Forall Blog](http://developer.nvidia.com/parallel-forall).

    - [HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese) <img src="https://img.shields.io/github/stars/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese?style=social"/> : This is a Chinese translation of the CUDA programming guide. 本项目为 CUDA C Programming Guide 的中文翻译版。

    - [brucefan1983/CUDA-Programming](https://github.com/brucefan1983/CUDA-Programming) <img src="https://img.shields.io/github/stars/brucefan1983/CUDA-Programming?style=social"/> : Sample codes for my CUDA programming book.

    - [YouQixiaowu/CUDA-Programming-with-Python](https://github.com/YouQixiaowu/CUDA-Programming-with-Python) <img src="https://img.shields.io/github/stars/YouQixiaowu/CUDA-Programming-with-Python?style=social"/> :  关于书籍CUDA Programming使用了pycuda模块的Python版本的示例代码。

    - [QINZHAOYU/CudaSteps](https://github.com/QINZHAOYU/CudaSteps) <img src="https://img.shields.io/github/stars/QINZHAOYU/CudaSteps?style=social"/> : 基于《cuda编程-基础与实践》（樊哲勇 著）的cuda学习之路。

    - [MAhaitao999/CUDA_Programming](https://github.com/MAhaitao999/CUDA_Programming) <img src="https://img.shields.io/github/stars/MAhaitao999/CUDA_Programming?style=social"/> : 《CUDA编程基础与实践》一书的代码。

    - [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) <img src="https://img.shields.io/github/stars/xlite-dev/LeetCUDA?style=social"/> : 📚LeetCUDA: Modern CUDA Learn Notes with PyTorch for Beginners🐑, 200+ CUDA/Tensor Cores Kernels, HGEMM, FA-2 MMA etc.🔥 [https://github.com/xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA)

    - [BBuf/how-to-optim-algorithm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda) <img src="https://img.shields.io/github/stars/BBuf/how-to-optim-algorithm-in-cuda?style=social"/> : how to optimize some algorithm in cuda.

    - [PaddleJitLab/CUDATutorial](https://github.com/PaddleJitLab/CUDATutorial) <img src="https://img.shields.io/github/stars/PaddleJitLab/CUDATutorial?style=social"/> : A self-learning tutorail for CUDA High Performance Programing. 从零开始学习 CUDA 高性能编程。

    - [RussWong/CUDATutorial](https://github.com/RussWong/CUDATutorial) <img src="https://img.shields.io/github/stars/RussWong/CUDATutorial?style=social"/> : A CUDA tutorial to make people learn CUDA program from 0.

    - [RichardAns/CUDA-Programs](https://github.com/RichardAns/CUDA-Programs) <img src="https://img.shields.io/github/stars/RichardAns/CUDA-Programs?style=social"/> : Examples from Programming in Parallel with CUDA.

    - [bertmaher/simplegemm](https://github.com/bertmaher/simplegemm) <img src="https://img.shields.io/github/stars/bertmaher/simplegemm?style=social"/> : Pingpong GEMM from scratch. I've been really excited to learn the lowest-level details of GPU matrix multiplication recently, so I was really inspired to read Pranjal Shankhdhar's fantastic blog post [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), which implements a fast gemm from first principles in CUDA, and actually outperforms cuBLAS. In a similar vein, I wanted to understand the [pingpong](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#hopper-warp-specialization) gemm algorithm in detail. So, I used [https://github.com/pranjalssh/fast.cu](https://github.com/pranjalssh/fast.cu) as a starting point, and wrote this kernel to see if I could match CUTLASS's pingpong implementation myself, using hand-written CUDA.

    - [pranjalssh/fast.cu](https://github.com/pranjalssh/fast.cu) <img src="https://img.shields.io/github/stars/pranjalssh/fast.cu?style=social"/> : Fastest GPU kernels, written from scratch. Matrix multiplication of square bf16 matrices, accumulated in fp32. Explanation in [https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)

    - [gpu-mode/lectures](https://github.com/gpu-mode/lectures) <img src="https://img.shields.io/github/stars/gpu-mode/lectures?style=social"/> : Material for gpu-mode lectures. [www.youtube.com/@GPUMODE](https://www.youtube.com/@GPUMODE)

    - [gpu-mode/resource-stream](https://github.com/gpu-mode/resource-stream) <img src="https://img.shields.io/github/stars/cuda-mode/resource-stream?style=social"/> :GPU programming related news and material links. [discord.gg/gpumode](https://discord.gg/gpumode)

    - [ifromeast/cuda_learning](https://github.com/ifromeast/cuda_learning) <img src="https://img.shields.io/github/stars/ifromeast/cuda_learning?style=social"/> : learning how CUDA works.

    - [a-hamdi/cuda](https://github.com/a-hamdi/cuda) <img src="https://img.shields.io/github/stars/a-hamdi/cuda?style=social"/> : 100 days of building Cuda kernels! This document serves as a log of the progress and knowledge I gained while working on CUDA programming and studying the PMPP (Parallel Programming and Optimization) book. Mentor: [https://github.com/hkproj/](https://github.com/hkproj/). Bro in the 100 days challenge: [https://github.com/1y33/100Days](https://github.com/1y33/100Days).

    - [SwekeR-463/100kernels](https://github.com/SwekeR-463/100kernels) <img src="https://img.shields.io/github/stars/SwekeR-463/100kernels?style=social"/> : 100 days of learning & making kernels in cuda / triton.

    - [Tongkaio/CUDA_Kernel_Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples) <img src="https://img.shields.io/github/stars/Tongkaio/CUDA_Kernel_Samples?style=social"/> : CUDA 算子手撕与面试指南。

    - [leimao/CUDA-GEMM-Optimization](https://github.com/leimao/CUDA-GEMM-Optimization) <img src="https://img.shields.io/github/stars/leimao/CUDA-GEMM-Optimization?style=social"/> : [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/). This repository contains the CUDA kernels for general matrix-matrix multiplication (GEMM) and the corresponding performance analysis.

    - [interestingLSY/CUDA-From-Correctness-To-Performance-Code](https://github.com/interestingLSY/CUDA-From-Correctness-To-Performance-Code) <img src="https://img.shields.io/github/stars/interestingLSY/CUDA-From-Correctness-To-Performance-Code?style=social"/> : Codes & examples for "CUDA - From Correctness to Performance". The lecture can be found at [https://wiki.lcpu.dev/zh/hpc/from-scratch/cuda](https://wiki.lcpu.dev/zh/hpc/from-scratch/cuda).

    - [Liu-xiandong/How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU) <img src="https://img.shields.io/github/stars/Liu-xiandong/How_to_optimize_in_GPU?style=social"/> : This is a series of GPU optimization topics. Here we will introduce how to optimize the CUDA kernel in detail. I will introduce several basic kernel optimizations, including: elementwise, reduce, sgemv, sgemm, etc. The performance of these kernels is basically at or near the theoretical limit.

    - [tpoisonooo/how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm) <img src="https://img.shields.io/github/stars/tpoisonooo/how-to-optimize-gemm?style=social"/> : row-major matmul optimization. [zhuanlan.zhihu.com/p/65436463](https://zhuanlan.zhihu.com/p/65436463).

    - [Bruce-Lee-LY/matrix_multiply](https://github.com/Bruce-Lee-LY/matrix_multiply) <img src="https://img.shields.io/github/stars/Bruce-Lee-LY/matrix_multiply?style=social"/> : Several common methods of matrix multiplication are implemented on CPU and Nvidia GPU using C++11 and CUDA.

    - [Bruce-Lee-LY/cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm) <img src="https://img.shields.io/github/stars/Bruce-Lee-LY/cuda_hgemm?style=social"/> : Several optimization methods of half-precision general matrix multiplication (HGEMM) using tensor core with WMMA API and MMA PTX instruction.

    - [Bruce-Lee-LY/cuda_hgemv](https://github.com/Bruce-Lee-LY/cuda_hgemv) <img src="https://img.shields.io/github/stars/Bruce-Lee-LY/cuda_hgemv?style=social"/> : Several optimization methods of half-precision general matrix vector multiplication (HGEMV) using CUDA core.

    - [enp1s0/ozIMMU](https://github.com/enp1s0/ozIMMU) <img src="https://img.shields.io/github/stars/enp1s0/ozIMMU?style=social"/> : FP64 equivalent GEMM via Int8 Tensor Cores using the Ozaki scheme. [arxiv.org/abs/2306.11975](https://arxiv.org/abs/2306.11975)

    - [Cjkkkk/CUDA_gemm](https://github.com/Cjkkkk/CUDA_gemm) <img src="https://img.shields.io/github/stars/Cjkkkk/CUDA_gemm?style=social"/> : A simple high performance CUDA GEMM implementation.

    - [AyakaGEMM/Hands-on-GEMM](https://github.com/AyakaGEMM/Hands-on-GEMM) <img src="https://img.shields.io/github/stars/AyakaGEMM/Hands-on-GEMM?style=social"/> : A GEMM tutorial.

    - [zpzim/MSplitGEMM](https://github.com/zpzim/MSplitGEMM) <img src="https://img.shields.io/github/stars/zpzim/MSplitGEMM?style=social"/> : Large matrix multiplication in CUDA.

    - [jundaf2/CUDA-INT8-GEMM](https://github.com/jundaf2/CUDA-INT8-GEMM) <img src="https://img.shields.io/github/stars/jundaf2/CUDA-INT8-GEMM?style=social"/> : CUDA 8-bit Tensor Core Matrix Multiplication based on m16n16k16 WMMA API.

    - [chanzhennan/cuda_gemm_benchmark](https://github.com/chanzhennan/cuda_gemm_benchmark) <img src="https://img.shields.io/github/stars/chanzhennan/cuda_gemm_benchmark?style=social"/> : Base on gtest/benchmark, refer to [https://github.com/Liu-xiandong/How_to_optimize_in_GPU](https://github.com/Liu-xiandong/How_to_optimize_in_GPU).

    - [YuxueYang1204/CudaDemo](https://github.com/YuxueYang1204/CudaDemo) <img src="https://img.shields.io/github/stars/YuxueYang1204/CudaDemo?style=social"/> : Implement custom operators in PyTorch with cuda/c++.

    - [CoffeeBeforeArch/cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming) <img src="https://img.shields.io/github/stars/CoffeeBeforeArch/cuda_programming?style=social"/> : Code from the "CUDA Crash Course" YouTube series by CoffeeBeforeArch.

    - [rbaygildin/learn-gpgpu](https://github.com/rbaygildin/learn-gpgpu) <img src="https://img.shields.io/github/stars/rbaygildin/learn-gpgpu?style=social"/> : Algorithms implemented in CUDA + resources about GPGPU.

    - [godweiyang/NN-CUDA-Example](https://github.com/godweiyang/NN-CUDA-Example) <img src="https://img.shields.io/github/stars/godweiyang/NN-CUDA-Example?style=social"/> : Several simple examples for popular neural network toolkits calling custom CUDA operators.

    - [yhwang-hub/Matrix_Multiplication_Performance_Optimization](https://github.com/yhwang-hub/Matrix_Multiplication_Performance_Optimization) <img src="https://img.shields.io/github/stars/yhwang-hub/Matrix_Multiplication_Performance_Optimization?style=social"/> : Matrix Multiplication Performance Optimization.

    - [caiwanxianhust/ClusteringByCUDA](https://github.com/caiwanxianhust/ClusteringByCUDA) <img src="https://img.shields.io/github/stars/caiwanxianhust/ClusteringByCUDA?style=social"/> : 使用 CUDA C++ 实现的一系列聚类算法。

    - [ulrichstern/cuda-convnet](https://github.com/ulrichstern/cuda-convnet) <img src="https://img.shields.io/github/stars/ulrichstern/cuda-convnet?style=social"/> : Alex Krizhevsky's original code from Google Code. "微信公众号「人工智能大讲堂」《[找到了AlexNet当年的源代码，没用框架，从零手撸CUDA/C++](https://mp.weixin.qq.com/s/plxXG8y5QlxSionyjyPXqw)》"。

    - [PacktPublishing/Learn-CUDA-Programming](https://github.com/PacktPublishing/Learn-CUDA-Programming) <img src="https://img.shields.io/github/stars/PacktPublishing/Learn-CUDA-Programming?style=social"/> : Learn CUDA Programming, published by Packt.

    - [PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA) <img src="https://img.shields.io/github/stars/PacktPublishing/Hands-On-GPU-Programming-with-Python-and-CUDA?style=social"/> : Hands-On GPU Programming with Python and CUDA, published by Packt.

    - [PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA](https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA) <img src="https://img.shields.io/github/stars/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA?style=social"/> : Hands-On GPU Accelerated Computer Vision with OpenCV and CUDA, published by Packt.

    - [BobMcDear/neural-network-cuda](https://github.com/BobMcDear/neural-network-cuda) <img src="https://img.shields.io/github/stars/BobMcDear/neural-network-cuda?style=social"/> : Neural network from scratch in CUDA/C++.

    - [zjhellofss/KuiperLLama](https://github.com/zjhellofss/KuiperLLama) <img src="https://img.shields.io/github/stars/zjhellofss/KuiperLLama?style=social"/> : 《动手自制大模型推理框架》。KuiperLLama 动手自制大模型推理框架，支持LLama2/3和Qwen2.5。校招、秋招、春招、实习好项目，带你从零动手实现支持LLama2/3和Qwen2.5的大模型推理框架。

    - [zjhellofss/KuiperInfer](https://github.com/zjhellofss/KuiperInfer) <img src="https://img.shields.io/github/stars/zjhellofss/KuiperInfer?style=social"/> :  校招、秋招、春招、实习好项目！带你从零实现一个高性能的深度学习推理库，支持大模型 llama2 、Unet、Yolov5、Resnet等模型的推理。Implement a high-performance deep learning inference library step by step。

    - [zjhellofss/kuiperdatawhale](https://github.com/zjhellofss/kuiperdatawhale) <img src="https://img.shields.io/github/stars/zjhellofss/kuiperdatawhale?style=social"/> :  从零自制深度学习推理框架。

    - [MarioSieg/magnetron](https://github.com/MarioSieg/magnetron) <img src="https://img.shields.io/github/stars/MarioSieg/magnetron?style=social"/> :  (WIP) A small but powerful, homemade PyTorch from scratch. Minimalistic homemade PyTorch alternative, written in C99 and Python.

    - [lucasdelimanogueira/PyNorch](https://github.com/lucasdelimanogueira/PyNorch) <img src="https://img.shields.io/github/stars/lucasdelimanogueira/PyNorch?style=social"/> :  Recreating PyTorch from scratch (C/C++, CUDA, NCCL and Python, with multi-GPU support and automatic differentiation!)

    - [xgqdut2016/cuda_code](https://github.com/xgqdut2016/cuda_code) <img src="https://img.shields.io/github/stars/xgqdut2016/cuda_code?style=social"/> : easy cuda code. CUDA代码简单入门。

    - [xgqdut2016/hpc_project](https://github.com/xgqdut2016/hpc_project) <img src="https://img.shields.io/github/stars/xgqdut2016/hpc_project?style=social"/> : some hpc project for learning.

    - [xgqdut2016/hpc2torch](https://github.com/xgqdut2016/hpc2torch) <img src="https://img.shields.io/github/stars/xgqdut2016/hpc2torch?style=social"/> : 这个仓库打算搭建一个高性能底层库的测试框架，将会针对onnx的算子编写相关的高性能kernel，作为pytorch的补充，从python端对比手写kernel和pytorch库函数的性能以及精度对比。

    - [zoheth/yan](https://github.com/zoheth/yan) <img src="https://img.shields.io/github/stars/zoheth/yan?style=social"/> : Yan (炎) is a high-performance CUDA operator library designed for learning purposes while emphasizing clean code and maximum performance.




  - ### TensorRT Learning

    - [NVIDIA TensorRT Docs](https://docs.nvidia.com/deeplearning/tensorrt/) : NVIDIA Deep Learning TensorRT Documentation.

    - [TensorRT](https://github.com/NVIDIA/TensorRT) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT?style=social"/> : NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. This repository contains the open source components of TensorRT. [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

    - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social"/> : TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines. [nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM)

    - [HeKun-NVIDIA/TensorRT-Developer_Guide_in_Chinese](https://github.com/HeKun-NVIDIA/TensorRT-Developer_Guide_in_Chinese) <img src="https://img.shields.io/github/stars/HeKun-NVIDIA/TensorRT-Developer_Guide_in_Chinese?style=social"/> : 本项目是NVIDIA TensorRT的中文版开发手册， 有个人翻译并添加自己的理解。

    - [kalfazed/tensorrt_starter](https://github.com/kalfazed/tensorrt_starter) <img src="https://img.shields.io/github/stars/kalfazed/tensorrt_starter?style=social"/> : This repository give a guidline to learn CUDA and TensorRT from the beginning.

    - [LitLeo/TensorRT_Tutorial](https://github.com/LitLeo/TensorRT_Tutorial) <img src="https://img.shields.io/github/stars/LitLeo/TensorRT_Tutorial?style=social"/> : TensorRT_Tutorial.





  - ### Triton Learning

    - [Triton](https://github.com/triton-lang/triton) <img src="https://img.shields.io/github/stars/triton-lang/triton?style=social"/> : Development repository for the Triton language and compiler. [triton-lang.org/](https://triton-lang.org/)

    - [Triton Docs](https://triton-lang.org/main/index.html) : Triton Documentation.

    - [hyperai/triton-cn](https://github.com/hyperai/triton-cn) <img src="https://img.shields.io/github/stars/hyperai/triton-cn?style=social"/> : Triton Documentation in Chinese Simplified / Triton 中文文档. [triton.hyper.ai](https://triton.hyper.ai/)



  - ### TileLang Learning

    - [TileLang](https://github.com/tile-ai/tilelang) <img src="https://img.shields.io/github/stars/tile-ai/tilelang?style=social"/> : Domain-specific language designed to streamline the development of high-performance GPU/CPU/Accelerators kernels. [tilelang.com/](https://tilelang.com/)

    - [TileLang Docs](https://tilelang.com/) : TileLang Documentation.




  - ### TVM Learning

    - [Apache TVM 中文站](https://tvm.hyper.ai/) : Apache TVM 中文文档！



  - ### MLIR Learning

    - [LLVM Docs](https://llvm.org/docs/) : LLVM Documentation.

    - [MLIR Docs](https://mlir.llvm.org/docs/) : MLIR Code Documentation.

    - [BBuf/tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn) <img src="https://img.shields.io/github/stars/BBuf/tvm_mlir_learn?style=social"/> : compiler learning resources collect.

    - [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial) <img src="https://img.shields.io/github/stars/j2kun/mlir-tutorial?style=social"/> : This is the code repository for a series of articles on the [MLIR framework](https://mlir.llvm.org/) for building compilers.

    - [KEKE046/mlir-tutorial](https://github.com/KEKE046/mlir-tutorial) <img src="https://img.shields.io/github/stars/KEKE046/mlir-tutorial?style=social"/> : Hands-On Practical MLIR Tutorial.

    - [AyakaGEMM/Hands-on-MLIR](https://github.com/AyakaGEMM/Hands-on-MLIR) <img src="https://img.shields.io/github/stars/AyakaGEMM/Hands-on-MLIR?style=social"/> : Hands-on-MLIR.

    - [yao-jiashu/KernelCodeGen](https://github.com/yao-jiashu/KernelCodeGen) <img src="https://img.shields.io/github/stars/yao-jiashu/KernelCodeGen?style=social"/> : GEMM/Conv2d CUDA/HIP kernel code generation using MLIR.



  - ### HPC Learning

    - [LAFF-On-PfHP](https://www.cs.utexas.edu/~flame/laff/pfhp/LAFF-On-PfHP.html) : LAFF-On Programming for High Performance.

    - [flame/how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm) <img src="https://img.shields.io/github/stars/flame/how-to-optimize-gemm?style=social"/> : How To Optimize Gemm wiki pages. [https://github.com/flame/how-to-optimize-gemm/wiki](https://github.com/flame/how-to-optimize-gemm/wiki)

    - [flame/blislab](https://github.com/flame/blislab) <img src="https://img.shields.io/github/stars/flame/blislab?style=social"/> : BLISlab: A Sandbox for Optimizing GEMM. Check the [tutorial](https://github.com/flame/blislab/blob/master/tutorial.pdf) for more details.

    - [tpoisonooo/how-to-optimize-gemm](https://github.com/tpoisonooo/how-to-optimize-gemm) <img src="https://img.shields.io/github/stars/tpoisonooo/how-to-optimize-gemm?style=social"/> : row-major matmul optimization. [zhuanlan.zhihu.com/p/65436463](https://zhuanlan.zhihu.com/p/65436463).

    - [YichengDWu/matmul.mojo](https://github.com/YichengDWu/matmul.mojo) <img src="https://img.shields.io/github/stars/YichengDWu/matmul.mojo?style=social"/> : High Performance Matrix Multiplication in Pure Mojo 🔥





## Frameworks

  - ### CUDA Frameworks

    - #### GPU Interface
      ##### GPU接口

        - ##### CPP Version

            - [CCCL](https://github.com/NVIDIA/cccl) <img src="https://img.shields.io/github/stars/NVIDIA/cccl?style=social"/> : CUDA C++ Core Libraries. The concept for the CUDA C++ Core Libraries (CCCL) grew organically out of the Thrust, CUB, and libcudacxx projects that were developed independently over the years with a similar goal: to provide high-quality, high-performance, and easy-to-use C++ abstractions for CUDA developers.

            - [HIP](https://github.com/ROCm/HIP) <img src="https://img.shields.io/github/stars/ROCm/HIP?style=social"/> : HIP: C++ Heterogeneous-Compute Interface for Portability. HIP is a C++ Runtime API and Kernel Language that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code. [rocmdocs.amd.com/projects/HIP/](https://rocmdocs.amd.com/projects/HIP/)


        - ##### Python Version

            - [NVIDIA/cuda-python](https://github.com/NVIDIA/cuda-python) <img src="https://img.shields.io/github/stars/NVIDIA/cuda-python?style=social"/> : CUDA Python is the home for accessing NVIDIA’s CUDA platform from Python. CUDA Python Low-level Bindings. [nvidia.github.io/cuda-python/](https://nvidia.github.io/cuda-python/latest/)

            - [CuPy](https://github.com/cupy/cupy) <img src="https://img.shields.io/github/stars/cupy/cupy?style=social"/> : CuPy : NumPy & SciPy for GPU. [cupy.dev](https://cupy.dev/)

            - [PyCUDA](https://github.com/inducer/pycuda) <img src="https://img.shields.io/github/stars/inducer/pycuda?style=social"/> : PyCUDA: Pythonic Access to CUDA, with Arrays and Algorithms. [mathema.tician.de/software/pycuda](http://mathema.tician.de/software/pycuda)



        - ##### Rust Version

            - [jessfraz/advent-of-cuda](https://github.com/jessfraz/advent-of-cuda) <img src="https://img.shields.io/github/stars/jessfraz/advent-of-cuda?style=social"/> : Doing advent of code with CUDA and rust.

            - [Bend](https://github.com/HigherOrderCO/Bend) <img src="https://img.shields.io/github/stars/HigherOrderCO/Bend?style=social"/> : A massively parallel, high-level programming language.[higherorderco.com](https://higherorderco.com/)

            - [HVM](https://github.com/HigherOrderCO/HVM) <img src="https://img.shields.io/github/stars/HigherOrderCO/HVM?style=social"/> : A massively parallel, optimal functional runtime in Rust.[higherorderco.com](https://higherorderco.com/)

            - [ZLUDA](https://github.com/vosen/ZLUDA) <img src="https://img.shields.io/github/stars/vosen/ZLUDA?style=social"/> : CUDA on AMD GPUs.

            - [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA) <img src="https://img.shields.io/github/stars/Rust-GPU/Rust-CUDA?style=social"/> : Ecosystem of libraries and tools for writing and executing fast GPU code fully in Rust.

            - [cudarc](https://github.com/coreylowman/cudarc) <img src="https://img.shields.io/github/stars/coreylowman/cudarc?style=social"/> : cudarc: minimal and safe api over the cuda toolkit.

            - [bindgen_cuda](https://github.com/Narsil/bindgen_cuda) <img src="https://img.shields.io/github/stars/Narsil/bindgen_cuda?style=social"/> : Similar crate than [bindgen](https://github.com/rust-lang/rust-bindgen) in philosophy. It will help create automatic bindgen to cuda kernels source files and make them easier to use directly from Rust.

            - [cuda-driver](https://github.com/YdrMaster/cuda-driver) <img src="https://img.shields.io/github/stars/YdrMaster/cuda-driver?style=social"/> : 基于 CUDA Driver API 的 cuda 运行时环境。

            - [async-cuda](https://github.com/oddity-ai/async-cuda) <img src="https://img.shields.io/github/stars/oddity-ai/async-cuda?style=social"/> : Asynchronous CUDA for Rust.

            - [async-tensorrt](https://github.com/oddity-ai/async-tensorrt) <img src="https://img.shields.io/github/stars/oddity-ai/async-tensorrt?style=social"/> : Asynchronous TensorRT for Rust.

            - [krnl](https://github.com/charles-r-earp/krnl) <img src="https://img.shields.io/github/stars/charles-r-earp/krnl?style=social"/> : Safe, portable, high performance compute (GPGPU) kernels.

            - [custos](https://github.com/elftausend/custos) <img src="https://img.shields.io/github/stars/elftausend/custos?style=social"/> : A minimal OpenCL, CUDA, WGPU and host CPU array manipulation engine / framework.

            - [spinorml/nvlib](https://github.com/spinorml/nvlib) <img src="https://img.shields.io/github/stars/spinorml/nvlib?style=social"/> : Rust interoperability with NVIDIA CUDA NVRTC and Driver.

            - [DoeringChristian/cuda-rs](https://github.com/DoeringChristian/cuda-rs) <img src="https://img.shields.io/github/stars/DoeringChristian/cuda-rs?style=social"/> : Cuda Bindings for rust generated with bindgen-cli (similar to cust_raw).

            - [romankoblov/rust-nvrtc](https://github.com/romankoblov/rust-nvrtc) <img src="https://img.shields.io/github/stars/romankoblov/rust-nvrtc?style=social"/> : NVRTC bindings for RUST.

            - [solkitten/astro-cuda](https://github.com/solkitten/astro-cuda) <img src="https://img.shields.io/github/stars/solkitten/astro-cuda?style=social"/> : CUDA Driver API bindings for Rust.

            - [bokutotu/curs](https://github.com/bokutotu/curs) <img src="https://img.shields.io/github/stars/bokutotu/curs?style=social"/> : cuda&cublas&cudnn wrapper for Rust.

            - [rust-cuda/cuda-sys](https://github.com/rust-cuda/cuda-sys) <img src="https://img.shields.io/github/stars/rust-cuda/cuda-sys?style=social"/> : Rust binding to CUDA APIs.

            - [bheisler/RustaCUDA](https://github.com/bheisler/RustaCUDA) <img src="https://img.shields.io/github/stars/bheisler/RustaCUDA?style=social"/> : Rusty wrapper for the CUDA Driver API.

            - [tmrob2/cuda2rust_sandpit](https://github.com/tmrob2/cuda2rust_sandpit) <img src="https://img.shields.io/github/stars/tmrob2/cuda2rust_sandpit?style=social"/> : Minimal examples to get CUDA linear algebra programs working with Rust using CC & FFI.

            - [PhDP/rust-cuda-template](https://github.com/PhDP/rust-cuda-template) <img src="https://img.shields.io/github/stars/PhDP/rust-cuda-template?style=social"/> : Simple template for Rust + CUDA.

            - [neka-nat/cuimage](https://github.com/neka-nat/cuimage) <img src="https://img.shields.io/github/stars/neka-nat/cuimage?style=social"/> : Rust implementation of image processing library with CUDA.

            - [yanghaku/cuda-driver-sys](https://github.com/yanghaku/cuda-driver-sys) <img src="https://img.shields.io/github/stars/yanghaku/cuda-driver-sys?style=social"/> : Rust binding to CUDA Driver APIs.

            - [Canyon-ml/canyon-sys](https://github.com/Canyon-ml/canyon-sys) <img src="https://img.shields.io/github/stars/Canyon-ml/canyon-sys?style=social"/> : Rust Bindings for Cuda, CuDNN.

            - [cea-hpc/HARP](https://github.com/cea-hpc/HARP) <img src="https://img.shields.io/github/stars/cea-hpc/HARP?style=social"/> : Small tool for profiling the performance of hardware-accelerated Rust code using OpenCL and CUDA.

            - [Conqueror712/CUDA-Simulator](https://github.com/Conqueror712/CUDA-Simulator) <img src="https://img.shields.io/github/stars/Conqueror712/CUDA-Simulator?style=social"/> : A self-developed version of the user-mode CUDA emulator project and a learning repository for Rust.

            - [cszach/rust-cuda-template](https://github.com/cszach/rust-cuda-template) <img src="https://img.shields.io/github/stars/cszach/rust-cuda-template?style=social"/> : A Rust CUDA template with detailed instructions.

            - [exor2008/fluid-simulator](https://github.com/exor2008/fluid-simulator) <img src="https://img.shields.io/github/stars/exor2008/fluid-simulator?style=social"/> : Rust CUDA fluid simulator.

            - [chichieinstein/rustycuda](https://github.com/chichieinstein/rustycuda) <img src="https://img.shields.io/github/stars/chichieinstein/rustycuda?style=social"/> : Convenience functions for generic handling of CUDA resources on the Rust side.

            - [Jafagervik/cruda](https://github.com/Jafagervik/cruda) <img src="https://img.shields.io/github/stars/Jafagervik/cruda?style=social"/> : CRUDA - Writing rust with cuda.

            - [lennyerik/cutransform](https://github.com/lennyerik/cutransform) <img src="https://img.shields.io/github/stars/lennyerik/cutransform?style=social"/> : CUDA kernels in any language supported by LLVM.

           - [cjordan/hip-sys](https://github.com/cjordan/hip-sys) <img src="https://img.shields.io/github/stars/cjordan/hip-sys?style=social"/> : Rust bindings for HIP.

            - [rust-gpu](https://github.com/EmbarkStudios/rust-gpu) <img src="https://img.shields.io/github/stars/EmbarkStudios/rust-gpu?style=social"/> : 🐉 Making Rust a first-class language and ecosystem for GPU shaders 🚧 [shader.rs](https://shader.rs/)

            - [wgpu](https://github.com/gfx-rs/wgpu) <img src="https://img.shields.io/github/stars/gfx-rs/wgpu?style=social"/> : Safe and portable GPU abstraction in Rust, implementing WebGPU API. [wgpu.rs](https://wgpu.rs/)

            - [Vulkano](https://github.com/vulkano-rs/vulkano) <img src="https://img.shields.io/github/stars/vulkano-rs/vulkano?style=social"/> : Safe and rich Rust wrapper around the Vulkan API. Vulkano is a Rust wrapper around [the Vulkan graphics API](https://www.vulkan.org/). It follows the Rust philosophy, which is that as long as you don't use unsafe code you shouldn't be able to trigger any undefined behavior. In the case of Vulkan, this means that non-unsafe code should always conform to valid API usage.

            - [Ash](https://github.com/ash-rs/ash) <img src="https://img.shields.io/github/stars/ash-rs/ash?style=social"/> : Vulkan bindings for Rust.

            - [ocl](https://github.com/cogciprocate/ocl) <img src="https://img.shields.io/github/stars/cogciprocate/ocl?style=social"/> : OpenCL for Rust.

            - [opencl3](https://github.com/kenba/opencl3) <img src="https://img.shields.io/github/stars/kenba/opencl3?style=social"/> : A Rust implementation of the Khronos [OpenCL 3.0](https://registry.khronos.org/OpenCL/) API.





        - ##### Julia Version

            - [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) <img src="https://img.shields.io/github/stars/JuliaGPU/CUDA.jl?style=social"/> : CUDA programming in Julia. [juliagpu.org/](https://juliagpu.org/)

            - [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) <img src="https://img.shields.io/github/stars/JuliaGPU/AMDGPU.jl?style=social"/> : AMD GPU (ROCm) programming in Julia.


    - #### Performance Benchmark

        - [NVIDIA/nvbench](https://github.com/NVIDIA/nvbench) <img src="https://img.shields.io/github/stars/NVIDIA/nvbench?style=social"/> : CUDA Kernel Benchmarking Library.

        - [FlagPerf](https://github.com/FlagOpen/FlagPerf) <img src="https://img.shields.io/github/stars/FlagOpen/FlagPerf?style=social"/> : FlagPerf is an open-source software platform for benchmarking AI chips. FlagPerf是智源研究院联合AI硬件厂商共建的一体化AI硬件评测引擎，旨在建立以产业实践为导向的指标体系，评测AI硬件在软件栈组合（模型+框架+编译器）下的实际能力。

        - [te42kyfo/gpu-benches](https://github.com/te42kyfo/gpu-benches) <img src="https://img.shields.io/github/stars/te42kyfo/gpu-benches?style=social"/> : collection of benchmarks to measure basic GPU capabilities.



    - #### Scientific Computing Framework
      ##### 科学计算框架

        - [cuBLAS](https://developer.nvidia.com/cublas) : Basic Linear Algebra on NVIDIA GPUs. NVIDIA cuBLAS is a GPU-accelerated library for accelerating AI and HPC applications. It includes several API extensions for providing drop-in industry standard BLAS APIs and GEMM APIs with support for fusions that are highly optimized for NVIDIA GPUs. The cuBLAS library also contains extensions for batched operations, execution across multiple GPUs, and mixed- and low-precision execution with additional tuning for the best performance.

        - [CUTLASS](https://github.com/NVIDIA/cutlass) <img src="https://img.shields.io/github/stars/NVIDIA/cutlass?style=social"/> : CUDA Templates for Linear Algebra Subroutines.

        - [MatX](https://github.com/NVIDIA/MatX) <img src="https://img.shields.io/github/stars/NVIDIA/MatX?style=social"/> : MatX - GPU-Accelerated Numerical Computing in Modern C++. An efficient C++17 GPU numerical computing library with Python-like syntax. [nvidia.github.io/MatX](https://nvidia.github.io/MatX)

        - [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) <img src="https://img.shields.io/github/stars/deepseek-ai/DeepGEMM?style=social"/> : DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling.

        - [MUTLASS](https://github.com/MooreThreads/mutlass) <img src="https://img.shields.io/github/stars/MooreThreads/mutlass?style=social"/> : MUSA Templates for Linear Algebra Subroutines.

        - [CuPy](https://github.com/cupy/cupy) <img src="https://img.shields.io/github/stars/cupy/cupy?style=social"/> : CuPy : NumPy & SciPy for GPU. [cupy.dev](https://cupy.dev/)

        - [GenericLinearAlgebra.jl](https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl) <img src="https://img.shields.io/github/stars/JuliaLinearAlgebra/GenericLinearAlgebra.jl?style=social"/> : Generic numerical linear algebra in Julia.

        - [custos-math](https://github.com/elftausend/custos-math) <img src="https://img.shields.io/github/stars/elftausend/custos-math?style=social"/> : This crate provides CUDA, OpenCL, CPU (and Stack) based matrix operations using [custos](https://github.com/elftausend/custos).


    - #### Attention and Transformer Framework

        - [FlashAttention](https://github.com/Dao-AILab/flash-attention) <img src="https://img.shields.io/github/stars/Dao-AILab/flash-attention?style=social"/> : Fast and memory-efficient exact attention. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". (**[arXiv 2022](https://arxiv.org/abs/2205.14135)**).

        - [SageAttention](https://github.com/thu-ml/SageAttention) <img src="https://img.shields.io/github/stars/thu-ml/SageAttention?style=social"/> : Quantized Attention that achieves speedups of 2.1-3.1x and 2.7-5.1x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models. "SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration". (**[arXiv 2024](https://arxiv.org/abs/2410.02367)**). "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization". (**[arXiv 2024](https://arxiv.org/abs/2411.10958)**).

        - [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) <img src="https://img.shields.io/github/stars/fla-org/flash-linear-attention?style=social"/> : 🚀 Efficient implementations of state-of-the-art linear attention models in Pytorch and Triton.

        - [66RING/tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) <img src="https://img.shields.io/github/stars/66RING/tiny-flash-attention?style=social"/> : [flash attention](https://github.com/Dao-AILab/flash-attention) tutorial written in python, triton, cuda, cutlass.

        - [weishengying/tiny-flash-attention](https://github.com/weishengying/tiny-flash-attention) <img src="https://img.shields.io/github/stars/weishengying/tiny-flash-attention?style=social"/> : 使用 cutlass 实现 flash-attention 精简版，具有教学意义。

        - [jepeake/tiny-flash-attention](https://github.com/jepeake/tiny-flash-attention) <img src="https://img.shields.io/github/stars/jepeake/tiny-flash-attention?style=social"/> : flash attention in ~20 lines.




    - #### Machine Learning Framework

        - [cuDNN](https://developer.nvidia.com/cudnn) : The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for [deep neural networks](https://developer.nvidia.com/deep-learning). cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, attention, matmul, pooling, and normalization.

        - [PyTorch](https://github.com/pytorch/pytorch) <img src="https://img.shields.io/github/stars/pytorch/pytorch?style=social"/> : Tensors and Dynamic neural networks in Python with strong GPU acceleration. [pytorch.org](https://pytorch.org/)

        - [MooreThreads/torch_musa](https://github.com/MooreThreads/torch_musa) <img src="https://img.shields.io/github/stars/MooreThreads/torch_musa?style=social"/> : torch_musa is an open source repository based on PyTorch, which can make full use of the super computing power of MooreThreads graphics cards.

        - [PaddlePaddle](https://github.com/PaddlePaddle/Paddle) <img src="https://img.shields.io/github/stars/PaddlePaddle/Paddle?style=social"/> : PArallel Distributed Deep LEarning: Machine Learning Framework from Industrial Practice （『飞桨』核心框架，深度学习&机器学习高性能单机、分布式训练和跨平台部署）. [www.paddlepaddle.org/](http://www.paddlepaddle.org/)

        - [flashlight/flashlight](https://github.com/flashlight/flashlight) <img src="https://img.shields.io/github/stars/flashlight/flashlight?style=social"/> : A C++ standalone library for machine learning. [fl.readthedocs.io/en/latest/](https://fl.readthedocs.io/en/latest/)

        - [yhwang-hub/dl_model_infer](https://github.com/yhwang-hub/dl_model_infer) <img src="https://img.shields.io/github/stars/yhwang-hub/dl_model_infer?style=social"/> : his is a c++ version of the AI reasoning library. Currently, it only supports the reasoning of the tensorrt model. The follow-up plan supports the c++ reasoning of frameworks such as Openvino, NCNN, and MNN. There are two versions for pre- and post-processing, c++ version and cuda version. It is recommended to use the cuda version., This repository provides accelerated deployment cases of deep learning CV popular models, and cuda c supports dynamic-batch image process, infer, decode, NMS.

        - [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) <img src="https://img.shields.io/github/stars/NVlabs/tiny-cuda-nn?style=social"/> : Lightning fast C++/CUDA neural network framework.

        - [zjhellofss/KuiperLLama](https://github.com/zjhellofss/KuiperLLama) <img src="https://img.shields.io/github/stars/zjhellofss/KuiperLLama?style=social"/> : 《动手自制大模型推理框架》。KuiperLLama 动手自制大模型推理框架，支持LLama2/3和Qwen2.5。校招、秋招、春招、实习好项目，带你从零动手实现支持LLama2/3和Qwen2.5的大模型推理框架。

        - [zjhellofss/KuiperInfer](https://github.com/zjhellofss/KuiperInfer) <img src="https://img.shields.io/github/stars/zjhellofss/KuiperInfer?style=social"/> :  校招、秋招、春招、实习好项目！带你从零实现一个高性能的深度学习推理库，支持大模型 llama2 、Unet、Yolov5、Resnet等模型的推理。Implement a high-performance deep learning inference library step by step。

        - [zjhellofss/kuiperdatawhale](https://github.com/zjhellofss/kuiperdatawhale) <img src="https://img.shields.io/github/stars/zjhellofss/kuiperdatawhale?style=social"/> :  从零自制深度学习推理框架。

        - [MarioSieg/magnetron](https://github.com/MarioSieg/magnetron) <img src="https://img.shields.io/github/stars/MarioSieg/magnetron?style=social"/> :  (WIP) A small but powerful, homemade PyTorch from scratch. Minimalistic homemade PyTorch alternative, written in C99 and Python.

        - [lucasdelimanogueira/PyNorch](https://github.com/lucasdelimanogueira/PyNorch) <img src="https://img.shields.io/github/stars/lucasdelimanogueira/PyNorch?style=social"/> :  Recreating PyTorch from scratch (C/C++, CUDA, NCCL and Python, with multi-GPU support and automatic differentiation!)


    - #### AI Inference Framework
      ##### AI推理框架





        - ##### LLM Inference and Serving Engine

            - [TensorRT](https://github.com/NVIDIA/TensorRT) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT?style=social"/> : NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. This repository contains the open source components of TensorRT. [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

            - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social"/> : TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines. [nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM)

            - [NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT-Model-Optimizer?style=social"/> : TensorRT Model Optimizer is a unified library of state-of-the-art model optimization techniques such as quantization, pruning, distillation, etc. It compresses deep learning models for downstream deployment frameworks like TensorRT-LLM or TensorRT to optimize inference speed on NVIDIA GPUs. [nvidia.github.io/TensorRT-Model-Optimizer](https://nvidia.github.io/TensorRT-Model-Optimizer/)

            - [Ollama](https://github.com/ollama/ollama) <img src="https://img.shields.io/github/stars/ollama/ollama?style=social"/> : Get up and running with Llama 3.3, DeepSeek-R1, Phi-4, Gemma 2, and other large language models. [ollama.com](https://ollama.com/)

            - [vLLM](https://github.com/vllm-project/vllm) <img src="https://img.shields.io/github/stars/vllm-project/vllm?style=social"/> : A high-throughput and memory-efficient inference and serving engine for LLMs. [docs.vllm.ai](https://docs.vllm.ai/)

            - [SGLang](https://github.com/sgl-project/sglang) <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social"/> : SGLang is a fast serving framework for large language models and vision language models. [docs.sglang.ai/](https://docs.sglang.ai/)

            - [MLC LLM](https://github.com/mlc-ai/mlc-llm) <img src="https://img.shields.io/github/stars/mlc-ai/mlc-llm?style=social"/> : Universal LLM Deployment Engine with ML Compilation. [llm.mlc.ai/](https://llm.mlc.ai/)

            - [KTransformers](https://github.com/kvcache-ai/ktransformers) <img src="https://img.shields.io/github/stars/kvcache-ai/ktransformers?style=social"/> : A Flexible Framework for Experiencing Cutting-edge LLM Inference Optimizations. [kvcache-ai.github.io/ktransformers/](https://kvcache-ai.github.io/ktransformers/)

            - [Chitu（赤兔）](https://github.com/thu-pacman/chitu) <img src="https://img.shields.io/github/stars/thu-pacman/chitu?style=social"/> : High-performance inference framework for large language models, focusing on efficiency, flexibility, and availability.

            - [Aphrodite](https://github.com/aphrodite-engine/aphrodite-engine) <img src="https://img.shields.io/github/stars/aphrodite-engine/aphrodite-engine?style=social"/> : Large-scale LLM inference engine. [aphrodite.pygmalion.chat](https://aphrodite.pygmalion.chat/)

            - [GPUStack](https://github.com/gpustack/gpustack) <img src="https://img.shields.io/github/stars/gpustack/gpustack?style=social"/> : GPUStack is an open-source GPU cluster manager for running AI models. Manage GPU clusters for running AI models. [gpustack.ai](https://gpustack.ai/)

            - [Lamini](https://github.com/lamini-ai/lamini) <img src="https://img.shields.io/github/stars/lamini-ai/lamini?style=social"/> : The Official Python Client for Lamini's API. [lamini.ai/](https://lamini.ai/)

            - [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) <img src="https://img.shields.io/github/stars/datawhalechina/self-llm?style=social"/> :  《开源大模型食用指南》基于Linux环境快速部署开源大模型，更适合中国宝宝的部署教程。

            - [ninehills/llm-inference-benchmark](https://github.com/ninehills/llm-inference-benchmark) <img src="https://img.shields.io/github/stars/ninehills/llm-inference-benchmark?style=social"/> : LLM Inference benchmark.

            - [csbench/csbench](https://github.com/csbench/csbench) <img src="https://img.shields.io/github/stars/csbench/csbench?style=social"/> : "CS-Bench: A Comprehensive Benchmark for Large Language Models towards Computer Science Mastery". (**[arXiv 2024](https://arxiv.org/abs/2406.08587)**).

            - [MooreThreads/vllm_musa](https://github.com/MooreThreads/vllm_musa) <img src="https://img.shields.io/github/stars/MooreThreads/vllm_musa?style=social"/> : A high-throughput and memory-efficient inference and serving engine for LLMs. [docs.vllm.ai](https://docs.vllm.ai/)

            - [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference) <img src="https://img.shields.io/github/stars/dusty-nv/jetson-inference?style=social"/> : Hello AI World guide to deploying deep-learning inference networks and deep vision primitives with TensorRT and NVIDIA Jetson.

            - [m0dulo/InferSpore](https://github.com/m0dulo/InferSpore) <img src="https://img.shields.io/github/stars/m0dulo/InferSpore?style=social"/> : 🌱 A fully independent Large Language Model (LLM) inference engine, built leveraging cuBLAS and cub. 🧩




        - ##### High Performance Kernel Library

            - [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) <img src="https://img.shields.io/github/stars/deepseek-ai/DeepGEMM?style=social"/> : DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling.

            - [FlashInfer](https://github.com/flashinfer-ai/flashinfer) <img src="https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social"/> : FlashInfer: Kernel Library for LLM Serving . [flashinfer.ai](flashinfer.ai)

            - [FlashMLA](https://github.com/deepseek-ai/FlashMLA) <img src="https://img.shields.io/github/stars/deepseek-ai/FlashMLA?style=social"/> : FlashMLA: Efficient MLA Decoding Kernel for Hopper GPUs.

            - [DeepEP](https://github.com/deepseek-ai/DeepEP) <img src="https://img.shields.io/github/stars/deepseek-ai/DeepEP?style=social"/> : DeepEP: an efficient expert-parallel communication library.




        - ##### C Implementation

            - [llm.c](https://github.com/karpathy/llm.c) <img src="https://img.shields.io/github/stars/karpathy/llm.c?style=social"/> : LLM training in simple, pure C/CUDA. There is no need for 245MB of PyTorch or 107MB of cPython. For example, training GPT-2 (CPU, fp32) is ~1,000 lines of clean code in a single file. It compiles and runs instantly, and exactly matches the PyTorch reference implementation.

            - [llama2.c](https://github.com/karpathy/llama2.c) <img src="https://img.shields.io/github/stars/karpathy/llama2.c?style=social"/> : Inference Llama 2 in one file of pure C. Train the Llama 2 LLM architecture in PyTorch then inference it with one simple 700-line C file (run.c).


        - ##### CPP Implementation

            - [gemma.cpp](https://github.com/google/gemma.cpp) <img src="https://img.shields.io/github/stars/google/gemma.cpp?style=social"/> :  gemma.cpp is a lightweight, standalone C++ inference engine for the Gemma foundation models from Google.

            - [llama.cpp](https://github.com/ggerganov/llama.cpp) <img src="https://img.shields.io/github/stars/ggerganov/llama.cpp?style=social"/> : Inference of [LLaMA](https://github.com/facebookresearch/llama) model in pure C/C++.

            - [whisper.cpp](https://github.com/ggerganov/whisper.cpp) <img src="https://img.shields.io/github/stars/ggerganov/whisper.cpp?style=social"/> : High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisper) automatic speech recognition (ASR) model.

            - [ChatGLM.cpp](https://github.com/li-plus/chatglm.cpp) <img src="https://img.shields.io/github/stars/li-plus/chatglm.cpp?style=social"/> : C++ implementation of [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) and [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B).

            - [MegEngine/InferLLM](https://github.com/MegEngine/InferLLM) <img src="https://img.shields.io/github/stars/MegEngine/InferLLM?style=social"/> : InferLLM is a lightweight LLM model inference framework that mainly references and borrows from the llama.cpp project.

            - [DeployAI/nndeploy](https://github.com/DeployAI/nndeploy) <img src="https://img.shields.io/github/stars/DeployAI/nndeploy?style=social"/> : nndeploy是一款模型端到端部署框架。以多端推理以及基于有向无环图模型部署为内核，致力为用户提供跨平台、简单易用、高性能的模型部署体验。[nndeploy-zh.readthedocs.io/zh/latest/](https://nndeploy-zh.readthedocs.io/zh/latest/)

            - [zjhellofss/KuiperInfer (自制深度学习推理框架)](https://github.com/zjhellofss/KuiperInfer) <img src="https://img.shields.io/github/stars/zjhellofss/KuiperInfer?style=social"/> :  带你从零实现一个高性能的深度学习推理库，支持llama 、Unet、Yolov5、Resnet等模型的推理。Implement a high-performance deep learning inference library step by step.

            - [skeskinen/llama-lite](https://github.com/skeskinen/llama-lite) <img src="https://img.shields.io/github/stars/skeskinen/llama-lite?style=social"/> : Embeddings focused small version of Llama NLP model.

            - [Const-me/Whisper](https://github.com/Const-me/Whisper) <img src="https://img.shields.io/github/stars/Const-me/Whisper?style=social"/> : High-performance GPGPU inference of OpenAI's Whisper automatic speech recognition (ASR) model.

            - [wangzhaode/ChatGLM-MNN](https://github.com/wangzhaode/ChatGLM-MNN) <img src="https://img.shields.io/github/stars/wangzhaode/ChatGLM-MNN?style=social"/> : Pure C++, Easy Deploy ChatGLM-6B.

            - [ztxz16/fastllm](https://github.com/ztxz16/fastllm) <img src="https://img.shields.io/github/stars/ztxz16/fastllm?style=social"/> : 纯c++实现，无第三方依赖的大模型库，支持CUDA加速，目前支持国产大模型ChatGLM-6B，MOSS; 可以在安卓设备上流畅运行ChatGLM-6B。

            - [davidar/eigenGPT](https://github.com/davidar/eigenGPT) <img src="https://img.shields.io/github/stars/davidar/eigenGPT?style=social"/> : Minimal C++ implementation of GPT2.

            - [Tlntin/Qwen-TensorRT-LLM](https://github.com/Tlntin/Qwen-TensorRT-LLM) <img src="https://img.shields.io/github/stars/Tlntin/Qwen-TensorRT-LLM?style=social"/> : 使用TRT-LLM完成对Qwen-7B-Chat实现推理加速。

            - [FeiGeChuanShu/trt2023](https://github.com/FeiGeChuanShu/trt2023) <img src="https://img.shields.io/github/stars/FeiGeChuanShu/trt2023?style=social"/> : NVIDIA TensorRT Hackathon 2023复赛选题：通义千问Qwen-7B用TensorRT-LLM模型搭建及优化。

            - [TRT2022/trtllm-llama](https://github.com/TRT2022/trtllm-llama) <img src="https://img.shields.io/github/stars/TRT2022/trtllm-llama?style=social"/> : ☢️ TensorRT 2023复赛——基于TensorRT-LLM的Llama模型推断加速优化。



        - ##### Mojo Implementation

            - [llama2.mojo](https://github.com/tairov/llama2.mojo) <img src="https://img.shields.io/github/stars/tairov/llama2.mojo?style=social"/> : Inference Llama 2 in one file of pure 🔥

            - [dorjeduck/llm.mojo](https://github.com/dorjeduck/llm.mojo) <img src="https://img.shields.io/github/stars/dorjeduck/llm.mojo?style=social"/> : port of Andrjey Karpathy's llm.c to Mojo.


        - ##### Rust Implementation

            - [Candle](https://github.com/huggingface/candle) <img src="https://img.shields.io/github/stars/huggingface/candle?style=social"/> : Minimalist ML framework for Rust.

            - [Safetensors](https://github.com/huggingface/safetensors) <img src="https://img.shields.io/github/stars/huggingface/safetensors?style=social"/> : Simple, safe way to store and distribute tensors. [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors/index)

            - [Tokenizers](https://github.com/huggingface/tokenizers) <img src="https://img.shields.io/github/stars/huggingface/tokenizers?style=social"/> : 💥 Fast State-of-the-Art Tokenizers optimized for Research and Production. [huggingface.co/docs/tokenizers](https://huggingface.co/docs/tokenizers/index)

            - [Burn](https://github.com/burn-rs/burn) <img src="https://img.shields.io/github/stars/burn-rs/burn?style=social"/> : Burn - A Flexible and Comprehensive Deep Learning Framework in Rust. [burn-rs.github.io/](https://burn-rs.github.io/)

            - [dfdx](https://github.com/coreylowman/dfdx) <img src="https://img.shields.io/github/stars/coreylowman/dfdx?style=social"/> : Deep learning in Rust, with shape checked tensors and neural networks.

            - [luminal](https://github.com/jafioti/luminal) <img src="https://img.shields.io/github/stars/jafioti/luminal?style=social"/> : Deep learning at the speed of light. [www.luminalai.com/](https://www.luminalai.com/)

            - [crabml](https://github.com/crabml/crabml) <img src="https://img.shields.io/github/stars/crabml/crabml?style=social"/> : crabml is focusing on the reimplementation of GGML using the Rust programming language.

            - [TensorFlow Rust](https://github.com/tensorflow/rust) <img src="https://img.shields.io/github/stars/tensorflow/rust?style=social"/> : Rust language bindings for TensorFlow.

            - [tch-rs](https://github.com/LaurentMazare/tch-rs) <img src="https://img.shields.io/github/stars/LaurentMazare/tch-rs?style=social"/> : Rust bindings for the C++ api of PyTorch.

            - [rustai-solutions/candle_demo_openchat_35](https://github.com/rustai-solutions/candle_demo_openchat_35) <img src="https://img.shields.io/github/stars/rustai-solutions/candle_demo_openchat_35?style=social"/> : candle_demo_openchat_35.

            - [llama2.rs](https://github.com/srush/llama2.rs) <img src="https://img.shields.io/github/stars/srush/llama2.rs?style=social"/> : A fast llama2 decoder in pure Rust.

            - [Llama2-burn](https://github.com/Gadersd/llama2-burn) <img src="https://img.shields.io/github/stars/Gadersd/llama2-burn?style=social"/> : Llama2 LLM ported to Rust burn.

            - [gaxler/llama2.rs](https://github.com/gaxler/llama2.rs) <img src="https://img.shields.io/github/stars/gaxler/llama2.rs?style=social"/> : Inference Llama 2 in one file of pure Rust 🦀

            - [whisper-burn](https://github.com/Gadersd/whisper-burn) <img src="https://img.shields.io/github/stars/Gadersd/whisper-burn?style=social"/> : A Rust implementation of OpenAI's Whisper model using the burn framework.

            - [stable-diffusion-burn](https://github.com/Gadersd/stable-diffusion-burn) <img src="https://img.shields.io/github/stars/Gadersd/stable-diffusion-burn?style=social"/> : Stable Diffusion v1.4 ported to Rust's burn framework.

            - [coreylowman/llama-dfdx](https://github.com/coreylowman/llama-dfdx) <img src="https://img.shields.io/github/stars/coreylowman/llama-dfdx?style=social"/> : [LLaMa 7b](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) with CUDA acceleration implemented in rust. Minimal GPU memory needed!

            - [tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs) <img src="https://img.shields.io/github/stars/tazz4843/whisper-rs?style=social"/> : Rust bindings to [whisper.cpp](https://github.com/ggerganov/whisper.cpp).

            - [rustformers/llm](https://github.com/rustformers/llm) <img src="https://img.shields.io/github/stars/rustformers/llm?style=social"/> : Run inference for Large Language Models on CPU, with Rust 🦀🚀🦙.

            - [Chidori](https://github.com/ThousandBirdsInc/chidori) <img src="https://img.shields.io/github/stars/ThousandBirdsInc/chidori?style=social"/> : A reactive runtime for building durable AI agents. [docs.thousandbirds.ai](https://docs.thousandbirds.ai/).

            - [llm-chain](https://github.com/sobelio/llm-chain) <img src="https://img.shields.io/github/stars/sobelio/llm-chain?style=social"/> : llm-chain is a collection of Rust crates designed to help you work with Large Language Models (LLMs) more effectively. [llm-chain.xyz](https://llm-chain.xyz/)

            - [Atome-FE/llama-node](https://github.com/Atome-FE/llama-node) <img src="https://img.shields.io/github/stars/Atome-FE/llama-node?style=social"/> : Believe in AI democratization. llama for nodejs backed by llama-rs and llama.cpp, work locally on your laptop CPU. support llama/alpaca/gpt4all/vicuna model. [www.npmjs.com/package/llama-node](https://www.npmjs.com/package/llama-node)

            - [Noeda/rllama](https://github.com/Noeda/rllama) <img src="https://img.shields.io/github/stars/Noeda/rllama?style=social"/> : Rust+OpenCL+AVX2 implementation of LLaMA inference code.

            - [lencx/ChatGPT](https://github.com/lencx/ChatGPT) <img src="https://img.shields.io/github/stars/lencx/ChatGPT?style=social"/> : 🔮 ChatGPT Desktop Application (Mac, Windows and Linux). [NoFWL](https://app.nofwl.com/).

            - [Synaptrix/ChatGPT-Desktop](https://github.com/Synaptrix/ChatGPT-Desktop) <img src="https://img.shields.io/github/stars/Synaptrix/ChatGPT-Desktop?style=social"/> : Fuel your productivity with ChatGPT-Desktop - Blazingly fast and supercharged!

            - [Poordeveloper/chatgpt-app](https://github.com/Poordeveloper/chatgpt-app) <img src="https://img.shields.io/github/stars/Poordeveloper/chatgpt-app?style=social"/> : A ChatGPT App for all platforms. Built with Rust + Tauri + Vue + Axum.

            - [mxismean/chatgpt-app](https://github.com/mxismean/chatgpt-app) <img src="https://img.shields.io/github/stars/mxismean/chatgpt-app?style=social"/> : Tauri 项目：ChatGPT App.

            - [sonnylazuardi/chat-ai-desktop](https://github.com/sonnylazuardi/chat-ai-desktop) <img src="https://img.shields.io/github/stars/sonnylazuardi/chat-ai-desktop?style=social"/> : Chat AI Desktop App. Unofficial ChatGPT desktop app for Mac & Windows menubar using Tauri & Rust.

            - [yetone/openai-translator](https://github.com/yetone/openai-translator) <img src="https://img.shields.io/github/stars/yetone/openai-translator?style=social"/> : The translator that does more than just translation - powered by OpenAI.

            - [m1guelpf/browser-agent](https://github.com/m1guelpf/browser-agent) <img src="https://img.shields.io/github/stars/m1guelpf/browser-agent?style=social"/> : A browser AI agent, using GPT-4. [docs.rs/browser-agent](https://docs.rs/browser-agent/latest/browser_agent/)

            - [sigoden/aichat](https://github.com/sigoden/aichat) <img src="https://img.shields.io/github/stars/sigoden/aichat?style=social"/> : Using ChatGPT/GPT-3.5/GPT-4 in the terminal.

            - [uiuifree/rust-openai-chatgpt-api](https://github.com/uiuifree/rust-openai-chatgpt-api) <img src="https://img.shields.io/github/stars/uiuifree/rust-openai-chatgpt-api?style=social"/> : "rust-openai-chatgpt-api" is a Rust library for accessing the ChatGPT API, a powerful NLP platform by OpenAI. The library provides a simple and efficient interface for sending requests and receiving responses, including chat. It uses reqwest and serde for HTTP requests and JSON serialization.

            - [1595901624/gpt-aggregated-edition](https://github.com/1595901624/gpt-aggregated-edition) <img src="https://img.shields.io/github/stars/1595901624/gpt-aggregated-edition?style=social"/> : 聚合ChatGPT官方版、ChatGPT免费版、文心一言、Poe、chatchat等多平台，支持自定义导入平台。

            - [Cormanz/smartgpt](https://github.com/Cormanz/smartgpt) <img src="https://img.shields.io/github/stars/Cormanz/smartgpt?style=social"/> : A program that provides LLMs with the ability to complete complex tasks using plugins.

            - [femtoGPT](https://github.com/keyvank/femtoGPT) <img src="https://img.shields.io/github/stars/keyvank/femtoGPT?style=social"/> : femtoGPT is a pure Rust implementation of a minimal Generative Pretrained Transformer. [discord.gg/wTJFaDVn45](https://github.com/keyvank/femtoGPT)

            - [shafishlabs/llmchain-rs](https://github.com/shafishlabs/llmchain-rs) <img src="https://img.shields.io/github/stars/shafishlabs/llmchain-rs?style=social"/> : 🦀Rust + Large Language Models - Make AI Services Freely and Easily. Inspired by LangChain.

            - [flaneur2020/llama2.rs](https://github.com/flaneur2020/llama2.rs) <img src="https://img.shields.io/github/stars/flaneur2020/llama2.rs?style=social"/> : An rust reimplementatin of [https://github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c).

            - [Heng30/chatbox](https://github.com/Heng30/chatbox) <img src="https://img.shields.io/github/stars/Heng30/chatbox?style=social"/> : A Chatbot for OpenAI ChatGPT. Based on Slint-ui and Rust.

            - [fairjm/dioxus-openai-qa-gui](https://github.com/fairjm/dioxus-openai-qa-gui) <img src="https://img.shields.io/github/stars/fairjm/dioxus-openai-qa-gui?style=social"/> : a simple openai qa desktop app built with dioxus.

            - [purton-tech/bionicgpt](https://github.com/purton-tech/bionicgpt) <img src="https://img.shields.io/github/stars/purton-tech/bionicgpt?style=social"/> : Accelerate LLM adoption in your organisation. Chat with your confidential data safely and securely. [bionic-gpt.com](https://bionic-gpt.com/)




        - #### Zig Implementation

            - [llama2.zig](https://github.com/cgbur/llama2.zig) <img src="https://img.shields.io/github/stars/cgbur/llama2.zig?style=social"/> : Inference Llama 2 in one file of pure Zig.

            - [renerocksai/gpt4all.zig](https://github.com/renerocksai/gpt4all.zig) <img src="https://img.shields.io/github/stars/renerocksai/gpt4all.zig?style=social"/> : ZIG build for a terminal-based chat client for an assistant-style large language model with ~800k GPT-3.5-Turbo Generations based on LLaMa.

            - [EugenHotaj/zig_inference](https://github.com/EugenHotaj/zig_inference) <img src="https://img.shields.io/github/stars/EugenHotaj/zig_inference?style=social"/> : Neural Network Inference Engine in Zig.


        - ##### Go Implementation

            - [Ollama](https://github.com/ollama/ollama/) <img src="https://img.shields.io/github/stars/ollama/ollama?style=social"/> : Get up and running with Llama 2, Mistral, Gemma, and other large language models. [ollama.com](https://ollama.com/)


            - [go-skynet/LocalAI](https://github.com/go-skynet/LocalAI) <img src="https://img.shields.io/github/stars/go-skynet/LocalAI?style=social"/> : 🤖 Self-hosted, community-driven, local OpenAI-compatible API. Drop-in replacement for OpenAI running LLMs on consumer-grade hardware. Free Open Source OpenAI alternative. No GPU required. LocalAI is an API to run ggml compatible models: llama, gpt4all, rwkv, whisper, vicuna, koala, gpt4all-j, cerebras, falcon, dolly, starcoder, and many other. [localai.io](https://localai.io/)



    - #### Distributed and Multi-GPU Framework
      ##### 分布式以及多GPU框架

        - [NVIDIA/nccl](https://github.com/NVIDIA/nccl) <img src="https://img.shields.io/github/stars/NVIDIA/nccl?style=social"/> : Optimized primitives for collective multi-GPU communication.

        - [NVIDIA/multi-gpu-programming-models](https://github.com/NVIDIA/multi-gpu-programming-models) <img src="https://img.shields.io/github/stars/NVIDIA/multi-gpu-programming-models?style=social"/> : Examples demonstrating available options to program multiple GPUs in a single node or a cluster.

        - [wilicc/gpu-burn](https://github.com/wilicc/gpu-burn) <img src="https://img.shields.io/github/stars/wilicc/gpu-burn?style=social"/> : Multi-GPU CUDA stress test.

        - [SCUDA](https://github.com/kevmo314/scuda) <img src="https://img.shields.io/github/stars/kevmo314/scuda?style=social"/> : SCUDA: GPU-over-IP. SCUDA is a GPU over IP bridge allowing GPUs on remote machines to be attached to CPU-only machines.




    - #### Robotics Framework
      ##### 机器人框架


        - [Cupoch](https://github.com/neka-nat/cupoch) <img src="https://img.shields.io/github/stars/neka-nat/cupoch?style=social"/> : Robotics with GPU computing.



    - #### ZKP and Web3 Framework
      ##### 零知识证明和Web3框架

        - [Tachyon](https://github.com/kroma-network/tachyon) <img src="https://img.shields.io/github/stars/kroma-network/tachyon?style=social"/> : Modular ZK(Zero Knowledge) backend accelerated by GPU.

        - [Blitzar](https://github.com/spaceandtimelabs/blitzar) <img src="https://img.shields.io/github/stars/spaceandtimelabs/blitzar?style=social"/> : Zero-knowledge proof acceleration with GPUs for C++ and Rust. [www.spaceandtime.io/](https://www.spaceandtime.io/)

        - [blitzar-rs](https://github.com/spaceandtimelabs/blitzar-rs) <img src="https://img.shields.io/github/stars/spaceandtimelabs/blitzar-rs?style=social"/> : High-Level Rust wrapper for the blitzar-sys crate. [www.spaceandtime.io/](https://www.spaceandtime.io/)

        - [ICICLE](https://github.com/ingonyama-zk/icicle) <img src="https://img.shields.io/github/stars/ingonyama-zk/icicle?style=social"/> : ICICLE is a library for ZK acceleration using CUDA-enabled GPUs.




  - ### Triton Frameworks

    - #### Triton Machine Learning Framework

        - [BobMcDear/attorch](https://github.com/BobMcDear/attorch) <img src="https://img.shields.io/github/stars/BobMcDear/attorch?style=social"/> : A subset of PyTorch's neural network modules, written in Python using OpenAI's Triton.



    - #### Triton High Performance Kernel Library

        - [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) <img src="https://img.shields.io/github/stars/linkedin/Liger-Kernel?style=social"/> : Efficient Triton Kernels for LLM Training. [arxiv.org/pdf/2410.10989](https://arxiv.org/pdf/2410.10989)

        - [FlagGems](https://github.com/FlagOpen/FlagGems) <img src="https://img.shields.io/github/stars/FlagOpen/FlagGems?style=social"/> : FlagGems is a high-performance general operator library implemented in [OpenAI Triton](https://github.com/openai/triton). It aims to provide a suite of kernel functions to accelerate LLM training and inference.

        - [linxihui/dkernel](https://github.com/linxihui/dkernel) <img src="https://img.shields.io/github/stars/linxihui/dkernel?style=social"/> : This repo contains customized CUDA kernels written in OpenAI Triton. As of now, it contains the sparse attention kernel used in [phi-3-small models](https://huggingface.co/microsoft/Phi-3-small-8k-instruct). The sparse attention is also supported in vLLM for efficient inference.



    - #### Triton Inference Framework

        - [harleyszhang/lite_llama](https://github.com/harleyszhang/lite_llama) <img src="https://img.shields.io/github/stars/harleyszhang/lite_llama?style=social"/> : The llama model inference lite framework by triton.


  - ### MLIR Frameworks

    - #### MLIR GPU Programming

        - ['gpu' Dialect](https://mlir.llvm.org/docs/Dialects/GPU/) : This dialect provides middle-level abstractions for launching GPU kernels following a programming model similar to that of CUDA or OpenCL.

        - ['amdgpu' Dialect](https://mlir.llvm.org/docs/Dialects/AMDGPU/) : The AMDGPU dialect provides wrappers around AMD-specific functionality and LLVM intrinsics.



    - #### MLIR FFI Bindings

        - [pyMLIR](https://github.com/spcl/pymlir) <img src="https://img.shields.io/github/stars/spcl/pymlir?style=social"/> : Python interface for MLIR - the Multi-Level Intermediate Representation. pyMLIR is a full Python interface to parse, process, and output [MLIR](https://mlir.llvm.org/) files according to the syntax described in the [MLIR documentation](https://github.com/llvm/llvm-project/tree/master/mlir/docs). pyMLIR supports the basic dialects and can be extended with other dialects.


    - #### MLIR Machine Learning Framework

        - [Torch-MLIR](https://github.com/llvm/torch-mlir) <img src="https://img.shields.io/github/stars/llvm/torch-mlir?style=social"/> : The Torch-MLIR project aims to provide first class support from the PyTorch ecosystem to the MLIR ecosystem.

        - [ONNX-MLIR](https://github.com/onnx/onnx-mlir) <img src="https://img.shields.io/github/stars/onnx/onnx-mlir?style=social"/> : Representation and Reference Lowering of ONNX Models in MLIR Compiler Infrastructure.

        - [TPU-MLIR](https://github.com/sophgo/tpu-mlir) <img src="https://img.shields.io/github/stars/sophgo/tpu-mlir?style=social"/> : Machine learning compiler based on MLIR for Sophgo TPU. TPU-MLIR is an open-source machine-learning compiler based on MLIR for TPU. This project provides a complete toolchain, which can convert pre-trained neural networks from different frameworks into binary files bmodel that can be efficiently operated on TPUs.

        - [IREE](https://github.com/iree-org/iree) <img src="https://img.shields.io/github/stars/iree-org/iree?style=social"/> : IREE: Intermediate Representation Execution Environment. A retargetable MLIR-based machine learning compiler and runtime toolkit. [iree.dev/](http://iree.dev/)

        - [ByteIR](https://github.com/bytedance/byteir) <img src="https://img.shields.io/github/stars/bytedance/byteir?style=social"/> : The ByteIR Project is a ByteDance model compilation solution. ByteIR includes compiler, runtime, and frontends, and provides an end-to-end model compilation solution. [byteir.ai](https://byteir.ai/)

        - [Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie) <img src="https://img.shields.io/github/stars/Xilinx/mlir-aie?style=social"/> : An MLIR-based toolchain for AMD AI Engine-enabled devices. This repository contains an MLIR-based toolchain for AI Engine-enabled devices, such as [AMD Ryzen™ AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) and [Versal™](https://www.xilinx.com/products/technology/ai-engine.html).







  - ### HPC Frameworks

    - [BLAS](https://www.netlib.org/blas/) : BLAS (Basic Linear Algebra Subprograms). The BLAS (Basic Linear Algebra Subprograms) are routines that provide standard building blocks for performing basic vector and matrix operations. The Level 1 BLAS perform scalar, vector and vector-vector operations, the Level 2 BLAS perform matrix-vector operations, and the Level 3 BLAS perform matrix-matrix operations.

    - [LAPACK](https://github.com/Reference-LAPACK/lapack) <img src="https://img.shields.io/github/stars/Reference-LAPACK/lapack?style=social"/> : LAPACK development repository. [LAPACK](https://www.netlib.org/lapack/) — Linear Algebra PACKage. LAPACK is written in Fortran 90 and provides routines for solving systems of simultaneous linear equations, least-squares solutions of linear systems of equations, eigenvalue problems, and singular value problems. The associated matrix factorizations (LU, Cholesky, QR, SVD, Schur, generalized Schur) are also provided, as are related computations such as reordering of the Schur factorizations and estimating condition numbers. Dense and banded matrices are handled, but not general sparse matrices. In all areas, similar functionality is provided for real and complex matrices, in both single and double precision.

    - [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) <img src="https://img.shields.io/github/stars/OpenMathLib/OpenBLAS?style=social"/> : OpenBLAS is an optimized BLAS library based on GotoBLAS2 1.13 BSD version. [www.openblas.net](http://www.openblas.net/)

    - [BLIS](https://github.com/flame/blis) <img src="https://img.shields.io/github/stars/flame/blis?style=social"/> : BLAS-like Library Instantiation Software Framework.

    - [NumPy](https://github.com/numpy/numpy) <img src="https://img.shields.io/github/stars/numpy/numpy?style=social"/> : The fundamental package for scientific computing with Python. [numpy.org](https://numpy.org/)

    - [SciPy](https://github.com/scipy/scipy) <img src="https://img.shields.io/github/stars/scipy/scipy?style=social"/> : SciPy library main repository. SciPy (pronounced "Sigh Pie") is an open-source software for mathematics, science, and engineering. It includes modules for statistics, optimization, integration, linear algebra, Fourier transforms, signal and image processing, ODE solvers, and more. [scipy.org](https://scipy.org/)

    - [Gonum](https://github.com/gonum/gonum) <img src="https://img.shields.io/github/stars/gonum/gonum?style=social"/> : Gonum is a set of numeric libraries for the Go programming language. It contains libraries for matrices, statistics, optimization, and more. [www.gonum.org/](https://www.gonum.org/)

    - [YichengDWu/matmul.mojo](https://github.com/YichengDWu/matmul.mojo) <img src="https://img.shields.io/github/stars/YichengDWu/matmul.mojo?style=social"/> : High Performance Matrix Multiplication in Pure Mojo 🔥. Matmul.🔥 is a high performance muilti-threaded implimentation of the [BLIS](https://en.wikipedia.org/wiki/BLIS_(software)) algorithm in pure Mojo 🔥.




## Applications

  - ### CUDA Applications


    - #### Image Preprocess

        - [dusty-nv/jetson-utils](https://github.com/dusty-nv/jetson-utils) <img src="https://img.shields.io/github/stars/dusty-nv/jetson-utils?style=social"/> : C++/CUDA/Python multimedia utilities for NVIDIA Jetson.

        - [emptysoal/cuda-image-preprocess](https://github.com/emptysoal/cuda-image-preprocess) <img src="https://img.shields.io/github/stars/emptysoal/cuda-image-preprocess?style=social"/> : Speed up image preprocess with cuda when handle image or tensorrt inference. Cuda编程加速图像预处理。



    - #### Object Detection

        - [laugh12321/TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) <img src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=social"/> : 🚀 TensorRT-YOLO: Support YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, PP-YOLOE using TensorRT acceleration with EfficientNMS! TensorRT-YOLO 是一个支持 YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、PP-YOLOE 和 PP-YOLOE+ 的推理加速项目，使用 NVIDIA TensorRT 进行优化。项目不仅集成了 EfficientNMS TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数来加速前处理过程。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供快速而优化的目标检测解决方案。

        - [l-sf/Linfer](https://github.com/l-sf/Linfer) <img src="https://img.shields.io/github/stars/l-sf/Linfer?style=social"/> : 基于TensorRT的C++高性能推理库，Yolov10, YoloPv2，Yolov5/7/X/8，RT-DETR，单目标跟踪OSTrack、LightTrack。

        - [Melody-Zhou/tensorRT_Pro-YOLOv8](https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8) <img src="https://img.shields.io/github/stars/Melody-Zhou/tensorRT_Pro-YOLOv8?style=social"/> : This repository is based on [shouxieai/tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro), with adjustments to support YOLOv8. 目前已支持 YOLOv8、YOLOv8-Cls、YOLOv8-Seg、YOLOv8-OBB、YOLOv8-Pose、RT-DETR、ByteTrack、YOLOv9、YOLOv10、RTMO 高性能推理！！！🚀🚀🚀

        - [shouxieai/tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro) <img src="https://img.shields.io/github/stars/shouxieai/tensorRT_Pro?style=social"/> : C++ library based on tensorrt integration.

        - [shouxieai/infer](https://github.com/shouxieai/infer) <img src="https://img.shields.io/github/stars/shouxieai/infer?style=social"/> : A new tensorrt integrate. Easy to integrate many tasks.

        - [kalfazed/tensorrt_starter](https://github.com/kalfazed/tensorrt_starter) <img src="https://img.shields.io/github/stars/kalfazed/tensorrt_starter?style=social"/> : This repository give a guidline to learn CUDA and TensorRT from the beginning.

        - [hamdiboukamcha/yolov10-tensorrt](https://github.com/hamdiboukamcha/yolov10-tensorrt) <img src="https://img.shields.io/github/stars/hamdiboukamcha/yolov10-tensorrt?style=social"/> : YOLOv10 C++ TensorRT : Real-Time End-to-End Object Detection.

        - [triple-Mu/YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT) <img src="https://img.shields.io/github/stars/triple-Mu/YOLOv8-TensorRT?style=social"/> : YOLOv8 using TensorRT accelerate !

        - [FeiYull/TensorRT-Alpha](https://github.com/FeiYull/TensorRT-Alpha) <img src="https://img.shields.io/github/stars/NVIDIA-AI-IOT/torch2trt?style=social"/> : 🔥🔥🔥TensorRT for YOLOv8、YOLOv8-Pose、YOLOv8-Seg、YOLOv8-Cls、YOLOv7、YOLOv6、YOLOv5、YOLONAS......🚀🚀🚀CUDA IS ALL YOU NEED.🍎🍎🍎

        - [cyrusbehr/YOLOv8-TensorRT-CPP](https://github.com/cyrusbehr/YOLOv8-TensorRT-CPP) <img src="https://img.shields.io/github/stars/cyrusbehr/YOLOv8-TensorRT-CPP?style=social"/> : YOLOv8 TensorRT C++ Implementation. A C++ Implementation of YoloV8 using TensorRT Supports object detection, semantic segmentation, and body pose estimation.

        - [VIDIA-AI-IOT/torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) <img src="https://img.shields.io/github/stars/NVIDIA-AI-IOT/torch2trt?style=social"/> : An easy to use PyTorch to TensorRT converter.

        - [zhiqwang/yolort](https://github.com/zhiqwang/yolort) <img src="https://img.shields.io/github/stars/zhiqwang/yolort?style=social"/> : yolort is a runtime stack for yolov5 on specialized accelerators such as tensorrt, libtorch, onnxruntime, tvm and ncnn. [zhiqwang.com/yolort](https://zhiqwang.com/yolort/)

        - [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) <img src="https://img.shields.io/github/stars/Linaom1214/TensorRT-For-YOLO-Series?style=social"/> : YOLO Series TensorRT Python/C++. tensorrt for yolo series (YOLOv8, YOLOv7, YOLOv6....), nms plugin support.

        - [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) <img src="https://img.shields.io/github/stars/wang-xinyu/tensorrtx?style=social"/> : TensorRTx aims to implement popular deep learning networks with tensorrt network definition APIs.


        - [DefTruth/lite.ai.toolkit](https://github.com/DefTruth/lite.ai.toolkit) <img src="https://img.shields.io/github/stars/DefTruth/lite.ai.toolkit?style=social"/> : 🛠 A lite C++ toolkit of awesome AI models with ONNXRuntime, NCNN, MNN and TNN. YOLOX, YOLOP, YOLOv6, YOLOR, MODNet, YOLOX, YOLOv7, YOLOv5. MNN, NCNN, TNN, ONNXRuntime. “🛠Lite.Ai.ToolKit: 一个轻量级的C++ AI模型工具箱，用户友好（还行吧），开箱即用。已经包括 100+ 流行的开源模型。这是一个根据个人兴趣整理的C++工具箱，, 涵盖目标检测、人脸检测、人脸识别、语义分割、抠图等领域。”

        - [PaddlePaddle/FastDeploy](https://github.com/PaddlePaddle/FastDeploy) <img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?style=social"/> : ⚡️An Easy-to-use and Fast Deep Learning Model Deployment Toolkit for ☁️Cloud 📱Mobile and 📹Edge. Including Image, Video, Text and Audio 20+ main stream scenarios and 150+ SOTA models with end-to-end optimization, multi-platform and multi-framework support.

        - [enazoe/yolo-tensorrt](https://github.com/enazoe/yolo-tensorrt) <img src="https://img.shields.io/github/stars/enazoe/yolo-tensorrt?style=social"/> : TensorRT8.Support Yolov5n,s,m,l,x .darknet -> tensorrt. Yolov4 Yolov3 use raw darknet *.weights and *.cfg fils. If the wrapper is useful to you,please Star it.

        - [guojianyang/cv-detect-robot](https://github.com/guojianyang/cv-detect-robot) <img src="https://img.shields.io/github/stars/guojianyang/cv-detect-robot?style=social"/> : 🔥🔥🔥🔥🔥🔥Docker NVIDIA Docker2 YOLOV5 YOLOX YOLO Deepsort TensorRT ROS Deepstream Jetson Nano TX2 NX for High-performance deployment(高性能部署)。

        - [BlueMirrors/Yolov5-TensorRT](https://github.com/BlueMirrors/Yolov5-TensorRT) <img src="https://img.shields.io/github/stars/BlueMirrors/Yolov5-TensorRT?style=social"/> : Yolov5 TensorRT Implementations.

        - [lewes6369/TensorRT-Yolov3](https://github.com/lewes6369/TensorRT-Yolov3) <img src="https://img.shields.io/github/stars/lewes6369/TensorRT-Yolov3?style=social"/> : TensorRT for Yolov3.

        - [CaoWGG/TensorRT-YOLOv4](https://github.com/CaoWGG/TensorRT-YOLOv4) <img src="https://img.shields.io/github/stars/CaoWGG/TensorRT-YOLOv4?style=social"/> :tensorrt5, yolov4, yolov3,yolov3-tniy,yolov3-tniy-prn.

        - [isarsoft/yolov4-triton-tensorrt](https://github.com/isarsoft/yolov4-triton-tensorrt) <img src="https://img.shields.io/github/stars/isarsoft/yolov4-triton-tensorrt?style=social"/> : YOLOv4 on Triton Inference Server with TensorRT.

        - [TrojanXu/yolov5-tensorrt](https://github.com/TrojanXu/yolov5-tensorrt) <img src="https://img.shields.io/github/stars/TrojanXu/yolov5-tensorrt?style=social"/> : A tensorrt implementation of yolov5.

        - [tjuskyzhang/Scaled-YOLOv4-TensorRT](https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT) <img src="https://img.shields.io/github/stars/tjuskyzhang/Scaled-YOLOv4-TensorRT?style=social"/> : Implement yolov4-tiny-tensorrt, yolov4-csp-tensorrt, yolov4-large-tensorrt(p5, p6, p7) layer by layer using TensorRT API.

        - [Syencil/tensorRT](https://github.com/Syencil/tensorRT) <img src="https://img.shields.io/github/stars/Syencil/tensorRT?style=social"/> : TensorRT-7 Network Lib 包括常用目标检测、关键点检测、人脸检测、OCR等 可训练自己数据。

        - [SeanAvery/yolov5-tensorrt](https://github.com/SeanAvery/yolov5-tensorrt) <img src="https://img.shields.io/github/stars/SeanAvery/yolov5-tensorrt?style=social"/> : YOLOv5 in TensorRT.

        - [Monday-Leo/YOLOv7_Tensorrt](https://github.com/Monday-Leo/YOLOv7_Tensorrt) <img src="https://img.shields.io/github/stars/Monday-Leo/YOLOv7_Tensorrt?style=social"/> : A simple implementation of Tensorrt YOLOv7.

        - [ibaiGorordo/ONNX-YOLOv6-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv6-Object-Detection) <img src="https://img.shields.io/github/stars/ibaiGorordo/ONNX-YOLOv6-Object-Detection?style=social"/> : Python scripts performing object detection using the YOLOv6 model in ONNX.

        - [ibaiGorordo/ONNX-YOLOv7-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection) <img src="https://img.shields.io/github/stars/ibaiGorordo/ONNX-YOLOv7-Object-Detection?style=social"/> : Python scripts performing object detection using the YOLOv7 model in ONNX.

        - [triple-Mu/yolov7](https://github.com/triple-Mu/yolov7) <img src="https://img.shields.io/github/stars/triple-Mu/yolov7?style=social"/> : End2end TensorRT YOLOv7.

        - [hewen0901/yolov7_trt](https://github.com/hewen0901/yolov7_trt) <img src="https://img.shields.io/github/stars/hewen0901/yolov7_trt?style=social"/> : yolov7目标检测算法的c++ tensorrt部署代码。

        - [tsutof/tiny_yolov2_onnx_cam](https://github.com/tsutof/tiny_yolov2_onnx_cam) <img src="https://img.shields.io/github/stars/tsutof/tiny_yolov2_onnx_cam?style=social"/> : Tiny YOLO v2 Inference Application with NVIDIA TensorRT.

        - [Monday-Leo/Yolov5_Tensorrt_Win10](https://github.com/Monday-Leo/Yolov5_Tensorrt_Win10) <img src="https://img.shields.io/github/stars/Monday-Leo/Yolov5_Tensorrt_Win10?style=social"/> : A simple implementation of tensorrt yolov5 python/c++🔥

        - [Wulingtian/yolov5_tensorrt_int8](https://github.com/Wulingtian/yolov5_tensorrt_int8) <img src="https://img.shields.io/github/stars/Wulingtian/yolov5_tensorrt_int8?style=social"/> : TensorRT int8 量化部署 yolov5s 模型，实测3.3ms一帧！

        - [Wulingtian/yolov5_tensorrt_int8_tools](https://github.com/Wulingtian/yolov5_tensorrt_int8_tools) <img src="https://img.shields.io/github/stars/Wulingtian/yolov5_tensorrt_int8_tools?style=social"/> : tensorrt int8 量化yolov5 onnx模型。

        - [MadaoFY/yolov5_TensorRT_inference](https://github.com/MadaoFY/yolov5_TensorRT_inference) <img src="https://img.shields.io/github/stars/MadaoFY/yolov5_TensorRT_inference?style=social"/> : 记录yolov5的TensorRT量化及推理代码，经实测可运行于Jetson平台。

        - [ibaiGorordo/ONNX-YOLOv8-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection) <img src="https://img.shields.io/github/stars/ibaiGorordo/ONNX-YOLOv8-Object-Detection?style=social"/> : Python scripts performing object detection using the YOLOv8 model in ONNX.

        - [we0091234/yolov8-tensorrt](https://github.com/we0091234/yolov8-tensorrt) <img src="https://img.shields.io/github/stars/we0091234/yolov8-tensorrt?style=social"/> : yolov8 tensorrt 加速.

        - [FeiYull/yolov8-tensorrt](https://github.com/FeiYull/yolov8-tensorrt) <img src="https://img.shields.io/github/stars/FeiYull/yolov8-tensorrt?style=social"/> : YOLOv8的TensorRT+CUDA加速部署，代码可在Win、Linux下运行。

        - [cvdong/YOLO_TRT_SIM](https://github.com/cvdong/YOLO_TRT_SIM) <img src="https://img.shields.io/github/stars/cvdong/YOLO_TRT_SIM?style=social"/> : 🐇 一套代码同时支持YOLO X, V5, V6, V7, V8 TRT推理 ™️ 🔝 ,前后处理均由CUDA核函数实现 CPP/CUDA🚀

        - [cvdong/YOLO_TRT_PY](https://github.com/cvdong/YOLO_TRT_PY) <img src="https://img.shields.io/github/stars/cvdong/YOLO_TRT_PY?style=social"/> : 🐰 一套代码同时支持YOLOV5, V6, V7, V8 TRT推理 ™️ PYTHON ✈️

        - [Psynosaur/Jetson-SecVision](https://github.com/Psynosaur/Jetson-SecVision) <img src="https://img.shields.io/github/stars/Psynosaur/Jetson-SecVision?style=social"/> : Person detection for Hikvision DVR with AlarmIO ports, uses TensorRT and yolov4.

        - [tatsuya-fukuoka/yolov7-onnx-infer](https://github.com/tatsuya-fukuoka/yolov7-onnx-infer) <img src="https://img.shields.io/github/stars/tatsuya-fukuoka/yolov7-onnx-infer?style=social"/> : Inference with yolov7's onnx model.

        - [MadaoFY/yolov5_TensorRT_inference](https://github.com/MadaoFY/yolov5_TensorRT_inference) <img src="https://img.shields.io/github/stars/MadaoFY/yolov5_TensorRT_inference?style=social"/> : 记录yolov5的TensorRT量化及推理代码，经实测可运行于Jetson平台。

        - [ervgan/yolov5_tensorrt_inference](https://github.com/ervgan/yolov5_tensorrt_inference) <img src="https://img.shields.io/github/stars/ervgan/yolov5_tensorrt_inference?style=social"/> : TensorRT cpp inference for Yolov5 model. Supports yolov5 v1.0, v2.0, v3.0, v3.1, v4.0, v5.0, v6.0, v6.2, v7.0.

        - [AlbinZhu/easy-trt](https://github.com/AlbinZhu/easy-trt) <img src="https://img.shields.io/github/stars/AlbinZhu/easy-trt?style=social"/> : TensorRT for YOLOv10 with CUDA.


    - #### Signal Processing

        - [Alisah-Ozcan/GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT) <img src="https://img.shields.io/github/stars/Alisah-Ozcan/GPU-FFT?style=social"/> : Welcome to the GPU-FFT-Optimization repository! We present cutting-edge algorithms and implementations for optimizing the Fast Fourier Transform (FFT) on Graphics Processing Units (GPUs).

    - #### Mesh Processing

        - [owensgroup/RXMesh](https://github.com/owensgroup/RXMesh) <img src="https://img.shields.io/github/stars/owensgroup/RXMesh?style=social"/> : GPU-accelerated triangle mesh processing. RXMesh is a surface triangle mesh data structure and programming model for processing static meshes on the GPU. RXMesh aims at provides a high-performance, generic, and compact data structure that can handle meshes regardless of their quality (e.g., non-manifold).

    - #### Graph Analytics

        - [rapidsai/cugraphs](https://github.com/rapidsai/cugraph) <img src="https://img.shields.io/github/stars/rapidsai/cugraph?style=social"/> : cuGraph - RAPIDS Graph Analytics Library. [docs.rapids.ai/api/cugraph/stable/](https://docs.rapids.ai/api/cugraph/stable/)





## Blogs

  - ### CUDA and TensorRT Blogs

    - 微信公众号「NVIDIA英伟达」
        - [2023-10-27，现已公开发布！欢迎使用 NVIDIA TensorRT-LLM 优化大语言模型推理](https://mp.weixin.qq.com/s/QaSbvyAmI6XXtr0y6W4LNQ)
        - [2023-11-24，使用 NVIDIA IGX Orin 开发者套件在边缘部署大语言模型](https://mp.weixin.qq.com/s/TOTVc5ntQJfH-DJ4_8uNTQ)
        - [2024-06-03，COMPUTEX 2024 | “加速一切”，NVIDIA CEO 黄仁勋在 COMPUTEX 开幕前发表主题演讲](https://mp.weixin.qq.com/s/usHo79-ssQiX0Rt5dvJ-sQ)
        - [2024-06-19，NVIDIA CEO 黄仁勋寄语毕业生：“对非常规、未经探索的东西保持信仰”](https://mp.weixin.qq.com/s/L8Lv6pz9BIgzLdm6qZm6dQ)
    - 微信公众号「NVIDIA英伟达企业解决方案」
        - [2024-04-24，NVIDIA GPU 架构下的 FP8 训练与推理](https://mp.weixin.qq.com/s/KV4XC9WT-8mfpmEzflIuvw)
        - [2024-06-14，初创加速计划 | 基于 NVIDIA Jetson 平台，国讯芯微实现大小脑端到端协同控制](https://mp.weixin.qq.com/s/R7U5JUgUCMK4rvtIpgStKQ)
        - [2024-06-20，NVIDIA Isaac Sim 4.0 和 NVIDIA Isaac Lab 为机器人工作流和仿真提供强大助力](https://mp.weixin.qq.com/s/BYqLDexhHnPMVsQMPLWpOA)
        - [2024-06-21，消除仿真与现实之间的差距：使用 NVIDIA Isaac Lab 训练 Spot 四足机器人运动](https://mp.weixin.qq.com/s/Nb4oMxijBofiidSAHkafag)
        - [2024-07-01，NVIDIA 端到端解决方案助力理想汽车打造智能驾驶体验与个性化车内空间](https://mp.weixin.qq.com/s/gmkYFj5BcJZHO4GJ_b8pyQ)
        - [2024-11-27，NVIDIA TensorRT-LLM Roadmap 现已在 GitHub 上公开发布！](https://mp.weixin.qq.com/s/zqAkxmWinwNMbcIBVA1hnA)
    - 微信公众号「AI不止算法」
        - [2024-03-20，C++模板推导再炫技：统一AI各个device各个kernel的调用和分发](https://mp.weixin.qq.com/s/r1XFocdVrfuArDWzpBYdAg)
        - [2024-04-09，全网首篇从tensorRT-LLM MoE CUDA kernel角度理解Mixtral-8x7b的推理加速及展望](https://mp.weixin.qq.com/s/3PsVUba-kTLIHK_s0RA2ow)
        - [2024-05-10，全面探究GPU SM内CUDA core-Tensor core能否同时计算？(上篇)](https://mp.weixin.qq.com/s/YASkRa12Ecr6fLtupP1WHg)
        - [2024-05-16，全面探究GPU SM内CUDA core-Tensor core能否同时计算？(下篇)](https://mp.weixin.qq.com/s/Jcu_HkAMiMXYagBjNhSCZQ)
        - [2024-10-09，深入解读tensorRT-LLM的关键技术 (未完待续)](https://mp.weixin.qq.com/s/2l5Ko2Q-iNOL3PpwpUdArw)
        - [2025-01-20，深入解析cutlass的诞生历程、特性和对友商的各大优势](https://mp.weixin.qq.com/s/-dHq2DOzsEiJQe1LvsRxrQ)
    - 微信公众号「澎峰科技PerfXLab」
        - [2022-10-18，深入浅出GPU优化系列：reduce优化](https://mp.weixin.qq.com/s/tNDRd18Ol56U-spoinzttg)
        - [2022-10-31，深入浅出GPU优化系列：spmv优化](https://mp.weixin.qq.com/s/JIqUbPFtYc3fs_cvKi1r3A)
        - [2023-05-24，深入浅出GPU优化系列：gemv优化](https://mp.weixin.qq.com/s/VCuJMrwGwyf9QCaaXcKmAg)
        - [2023-05-24，深入浅出GPU优化系列：GEMM优化（一）](https://mp.weixin.qq.com/s/4aPW_93IV54lzs5JRn0JiA)
        - [2023-06-02，深入浅出GPU优化系列：GEMM优化（二）](https://mp.weixin.qq.com/s/1q5ocZ7vDDsvew3HNo_9Vg)
        - [2023-06-16，深入浅出GPU优化系列：GEMM优化（三）](https://mp.weixin.qq.com/s/13Nw6fubNLOMFR3ROc0z0w)
        - [2023-06-26，深入浅出GPU优化系列：elementwise优化及CUDA工具链介绍](https://mp.weixin.qq.com/s/5h0lpKun0DlbefH_y-AdMg)
        - [2023-06-27，漫谈高性能计算与性能优化：访存](https://mp.weixin.qq.com/s/9BhJOqkbNbwXcSJuHUU52w)
        - [2024-07-04，澎峰科技研发的高性能计算原语库PerfIPP库技术白皮书发布（附下载）](https://mp.weixin.qq.com/s/Hd8S7bJGjvz9GUK6Q0lWSw)
    - 微信公众号「大猿搬砖简记」
        - [2024-03-11，图解Mixtral 8 * 7b推理优化原理与源码实现](https://mp.weixin.qq.com/s/jjZQ4A-rvk_e-woKLlNTVQ)
        - [2024-03-29，图解大模型计算加速系列之：vLLM核心技术PagedAttention原理](https://mp.weixin.qq.com/s/-5EniAmFf1v9RdxI5-CwiQ)
        - [2024-04-06，图解大模型计算加速系列：vLLM源码解析1，整体架构](https://mp.weixin.qq.com/s/r_t6_zMvPT7za82MZX4oRA)
        - [2024-04-12，图解大模型计算加速系列：vLLM源码解析2，调度器策略(Scheduler)](https://mp.weixin.qq.com/s/UCdqQUM_9a36uXkO36wpSg)
        - [2024-04-19，从啥也不会到Cuda GEMM优化](https://mp.weixin.qq.com/s/YLrsu1KAhzG8gFQ2L-TaMA)
    - 微信公众号「oldpan博客」
        - [2024-03-19，NVIDIA大语言模型落地的全流程解析](https://mp.weixin.qq.com/s/-sNnuDvkucUB_9K9RBfDEw)
        - [2024-03-20，TensorRT-LLM初探（二）简析了结构，用的更明白](https://mp.weixin.qq.com/s/Jk-AK84sllBbkDDpvkv62w)
        - [2024-03-21，高性能 LLM 推理框架的设计与实现](https://mp.weixin.qq.com/s/zys9KvQWbbdRHkOyhzZqUw)
        - [2024-04-15，[深入分析CUTLASS系列] 0x01 cutlass 源码分析(零) --- 软件架构(附ncu性能分析方法)](https://mp.weixin.qq.com/s/sLvZoWgILuRnvyiimMgeaQ)
        - [2024-04-21，搞懂 NVIDIA GPU 性能指标 很容易弄混的一个概念： Utilization vs Saturation](https://mp.weixin.qq.com/s/6PcF2RwGdm1G0JllGSS3jw)
        - [2024-04-22，快速提升性能，如何更好地使用GPU（上）](https://mp.weixin.qq.com/s/dUj058iBzYm-J2vlS5DfNA)
        - [2024-05-14，快速提升性能，如何更好地使用GPU（下）](https://mp.weixin.qq.com/s/NPcCHlLjBZeUiAhQOHX5qA)
        - [2024-05-22，大模型精度（FP16，FP32，BF16）详解与实践](https://mp.weixin.qq.com/s/95CUl1bGN-fSvmAbH0O-DA)
        - [2024-07-24，CUDA性能简易优化（一）背景知识](https://mp.weixin.qq.com/s/mFMlBh3zPZaCRWQH-neeDA)
        - [2024-08-06，如何把 PyTorch 的 GPU 利用率提升到 100% ?](https://mp.weixin.qq.com/s/Hwc32FDTOMaZSbbtNInCtg)
        - [2024-08-13，TensorRT-LLM初探（三）最佳部署实践](https://mp.weixin.qq.com/s/BiSvYW0-Nb6qf-bTXTRUwg)
    - 微信公众号「DeepPrompting」
        - [2024-01-09，LLM推理库TensorRT-LLM深入分析](https://mp.weixin.qq.com/s/hI6maWtVGHnTi0uGPj6tmA)
        - [2024-04-10，一文上手 Tensor Core指令级编程](https://mp.weixin.qq.com/s/Gi8ExdfErUkfWu3oRyKvBw)
        - [2024-04-23，大语言模型量化](https://mp.weixin.qq.com/s/3RUVgfrLdxyeoWX1R2Hq-Q)
        - [2024-04-25，动手实现混合精度矩阵乘CUDA内核](https://mp.weixin.qq.com/s/JGYFOsPvUSNMQWjR1gKOOg)
        - [2024-04-26，一文了解CUDA矩阵乘编程](https://mp.weixin.qq.com/s/vG7d7-tAt-mXOgRSb-jZRA)
    - 微信公众号「GiantPandaCV」
        - [2024-04-20，Tensor Cores 使用介绍](https://mp.weixin.qq.com/s/Mr-yR_YW5nNKV2dSrr5U2Q)
        - [2024-05-27，[并行训练]Context Parallelism的原理与代码浅析](https://mp.weixin.qq.com/s/vXWUUtAQNkBUpgDIJV8C0w)
        - [2024-06-20， FP8量化解读--8bit下最优方案？（一）](https://mp.weixin.qq.com/s/WcFG7mmsEwrL0g3dSJTC5A)
        - [2024-07-01，CUDA-MODE 课程笔记 第一课: 如何在 PyTorch 中 profile CUDA kernels](https://mp.weixin.qq.com/s/owF7AFR61SLrOosUPdZPQQ)
        - [2024-07-04，CUDA-MODE 第一课课后实战（上）](https://mp.weixin.qq.com/s/9XeJPWUsKTaMU2OdPkL-OQ)
        - [2024-07-06，CUDA-MODE 课程笔记 第二课: PMPP 书的第1-3章速通](https://mp.weixin.qq.com/s/y0fYn8gUqHqEoRO41ftKnA)
        - [2024-07-13，CUDA-MODE 课程笔记 第四课: PMPP 书的第4-5章笔记](https://mp.weixin.qq.com/s/P87c8LRJ1CEOOyaQw8L-cA)
        - [2024-07-18，CUDA-MODE课程笔记 第6课: 如何优化PyTorch中的优化器](https://mp.weixin.qq.com/s/qxPYdGZ71DKVLnnYxmvUVA)
        - [2024-07-19，CUDA-MODE 第一课课后实战（下）](https://mp.weixin.qq.com/s/FCqnQESCQTtlqCG_BSLulA)
        - [2024-07-23，CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记](https://mp.weixin.qq.com/s/r9b1dGyOr82ooMl4LD1n_Q)
        - [2024-07-28，CUDA-MODE课程笔记 第7课: Quantization Cuda vs Triton](https://mp.weixin.qq.com/s/1gCgpp49NF7sDw__EpO-nw)
        - [2024-08-01，TRT-LLM中的Quantization GEMM（Ampere Mixed GEMM）CUTLASS 2.x 课程学习笔记](https://mp.weixin.qq.com/s/NPytrkchX25YRBc_6Zy6nA)
        - [2024-08-05，CUDA-MODE课程笔记 第8课: CUDA性能检查清单](https://mp.weixin.qq.com/s/zJLDVF-yjuZ_lMjaCHoS5g)
        - [2024-09-12，CUDA-MODE课程笔记 第12课，Flash Attention](hhttps://mp.weixin.qq.com/s/IBeBHO5WlS5BfyL0nZaDHg)
    - 微信公众号「GPUS开发者」
        - [2023-10-30，利用NVIDIA Jetson Orin的强大能力执行本地LLM模型](https://mp.weixin.qq.com/s/6J7fEnumqpzSGrG3plcInw)
        - [2024-05-07，基于NVIDIA Jetson AGX Orin和Audio2Face做一个AI聊天数字人](https://mp.weixin.qq.com/s/7z0uU58IxwoXcI4bZ3z68g)
        - [2024-05-14，CUDA与OpenCL：并行计算革命的冲突与未来](https://mp.weixin.qq.com/s/h0nBvuV8nnfsbX1mjXAXVw)
        - [2024-10-08，NVIDIA Jetson平台助力Instacart，实现超市智能购物无缝体验](https://mp.weixin.qq.com/s/Q9x83ts0boNoQHZhSIpgUw)
        - [2024-11-28，TensorRT-LLM：开启Jetson平台上大语言模型推理的新篇章](https://mp.weixin.qq.com/s/FGg1s__LORE6SOzI_VsJYw)
        - [2024-11-29，在 Nvidia Jetson AGX Orin 上使用 TensorRT-LLM 运行 LLM](https://mp.weixin.qq.com/s/PhrusT3NisyhxtsO7G5U1g)
        - [2025-04-07，CUDA编程革命：cuTile分块计算赋能高效GPU开发](https://mp.weixin.qq.com/s/YfxJkAzr_5FRuY828Pv9kg)
    - 微信公众号「机器学习研究组订阅」
        - [2017-12-07，【推荐】CUTLASS：CUDA C++高性能线性代数运算库](https://mp.weixin.qq.com/s/EDmbQ4y3nnkYiHhl3HG_HA)
    - 微信公众号「自动驾驶之心」
        - [2024-02-28，熬了几个通宵，我写了份CUDA新手入门代码](https://mp.weixin.qq.com/s/UXIzQ9SYhtN4q8VfzNXDqA)
        - [2024-03-24，CUDA之通用矩阵乘法：从入门到熟练！](https://mp.weixin.qq.com/s/n9c-SZl5I_oj4N2l3I5kRg)
        - [2024-05-13，Shared memory！CUDA数据拷贝速度拉满~](https://mp.weixin.qq.com/s/P5CdO3QCSQKuj3nWjS_2yA)
    - 微信公众号「Meet DSA」
        - [2024-03-29，大语言模型硬件加速器综述](https://mp.weixin.qq.com/s/rtq8e_zVUWLc-vkT4V0qzQ)
    - 微信公众号「AI寒武纪」
        - [2024-04-10，【太疯狂了】用 1000 行纯 C 代码实现 GPT-2 训练：Andrej Karpathy重塑LLM训练格局](https://mp.weixin.qq.com/s/hNKWVqepbega6YPf48b8ag)
        - [2024-04-14，【全球黑客加持】Karpathy 1000行纯C训练大模型速度已追平PyTorch](https://mp.weixin.qq.com/s/VvwDhMmq80yN-Wcb8s3aiQ)
    - 微信公众号「关于NLP那些你不知道的事」
        - [2024-01-26，基于TensorRT-LLM的大模型部署(速通笔记)](https://mp.weixin.qq.com/s/2d6ihFFDTDfppYbjtBPHMw)
    - 微信公众号「InfoQ」
        - [2024-04-09，“真男人就应该用 C 编程”！用 1000 行 C 代码手搓了一个大模型，Mac 即可运行，特斯拉前AI总监爆火科普 LLM](https://mp.weixin.qq.com/s/qb0dhdFnXZS4LeW2mvG6fg)
    - 微信公众号「机器之心」
        - [2024-04-09，纯C语言手搓GPT-2，前OpenAI、特斯拉高管新项目火了](https://mp.weixin.qq.com/s/YMuq9Jo9Nibl1QFbLNxazg)
        - [2024-05-20，首个GPU高级语言，大规模并行就像写Python，已获8500 Star](https://mp.weixin.qq.com/s/dC7Z5Rk05sM7ND7bYUsrZA)
    - 微信公众号「新智元」
        - [2023-09-10，H100推理飙升8倍！英伟达官宣开源TensorRT-LLM，支持10+模型](https://mp.weixin.qq.com/s/xcNQBG69XkS6mOstzqROAw)
        - [2024-04-07，Llama提速500%！谷歌美女程序员手搓矩阵乘法内核](https://mp.weixin.qq.com/s/2ROw_Tmmh4NHf8WOiwnJLg)
        - [2024-04-09，1000行C语言搓出GPT-2！AI大神Karpathy新项目刚上线就狂揽2.5k星](https://mp.weixin.qq.com/s/_W2GlbO8nAfpLPtRtQJ-yw)
    - 微信公众号「GitHubStore」
        - [2024-04-11，llm.c：实现了大语言模型(LLM)训练的简单、纯 C/CUDA 版本，无需 PyTorch 或 cPython](https://mp.weixin.qq.com/s/7cHYDBHqs8ClkijI-Fya9A)
    - 微信公众号「云云众生s」
        - [2024-04-17，NVIDIA希望有更多支持CUDA的编程语言](https://mp.weixin.qq.com/s/jABUruiJwjhGstbPG3U2Fw)
    - 微信公众号「手写AI」
        - [2022-10-16，TensorRT/CUDA超全代码资料仓库](https://mp.weixin.qq.com/s/WXZXVlAohZn2YJ490pddpQ)
    - 微信公众号「美团技术团队」
        - [2024-04-11，美团外卖基于GPU的向量检索系统实践](https://mp.weixin.qq.com/s/pPl-anyQnFNFkmBlVsrBpA)
    - 微信公众号「GitHubFun网站」
        - [2024-04-20，英伟达开源人工智能代数库：线性代数子例程的 CUDA 模板](https://mp.weixin.qq.com/s/CwTnG89-tc1HaapvbU0D6g)
    - 微信公众号「大模型生态圈」
        - [2024-03-18，LLM百倍推理加速之量化篇](https://mp.weixin.qq.com/s/jbpVBZLZ0AkrP7bacY5mKw)
        - [2024-03-22，LLM推理：GPU资源和推理框架选择](https://mp.weixin.qq.com/s/qUaLOXZmk1xyGHGKX4ZtpQ)
        - [2024-03-27，LLM 推理加速方式汇总](https://mp.weixin.qq.com/s/IlaQw6Ut25NNoTZkxs63Vg)
        - [2024-04-26，LLM推理量化：FP8 VS INT8](https://mp.weixin.qq.com/s/e7QZC1qNkETXNXZpcD9cRg)
        - [2024-04-28，Nvidia GPU池化-远程GPU](https://mp.weixin.qq.com/s/tFdtYy5L_0V85OTvlPVK0A)
        - [2024-05-01，Nvidia Tensor Core 初探](https://mp.weixin.qq.com/s/VAuk2WdFqiW4ujV0A3-8HA)
        - [2024-05-24，Pytorch 显存管理机制与显存占用分析方法](https://mp.weixin.qq.com/s/QufR1esHGc3qkwgW6sAM-Q)
        - [2024-06-02，[LLM推理优化][万字]TensorRT-LLM部署调优-指北](https://mp.weixin.qq.com/s/PGOleShWEjHCPpw1wuV7SA)
    - 微信公众号「苏哲管理咨询」
        - [2024-02-25，英伟达（NVIDA）崛起不平凡之路--老黄全球AI芯片新帝国简史](https://mp.weixin.qq.com/s/4c8FtVeJmNlXL6akj5lj8A)
    - 微信公众号「后来遇见AI」
        - [2022-08-08，【机器学习】K均值聚类算法原理](https://mp.weixin.qq.com/s/o9bl1M9G1cOSYzzTZ3eYxw)
        - [2022-08-11，【CUDA编程】基于CUDA的Kmeans算法的简单实现](https://mp.weixin.qq.com/s/2PfocGm9l84l5Jj1vYF5bg)
        - [2024-01-23，【CUDA编程】基于 CUDA 的 Kmeans 算法的进阶实现（一）](hhttps://mp.weixin.qq.com/s/5Kr8ltlzy1nL7aeGrETYvA)
        - [2024-01-24，【CUDA编程】基于 CUDA 的 Kmeans 算法的进阶实现（二）](https://mp.weixin.qq.com/s/xPN5cupqt4B-JrX6KUNJrw)
        - [2024-04-08，【CUDA编程】CUDA 统一内存](https://mp.weixin.qq.com/s/DynVo_Mu7pUQxRLHH3ii9Q)
        - [2024-08-06，【CUDA编程】cuBLAS 库中矩阵乘法参数设置问题](https://mp.weixin.qq.com/s/MvTaIBfVW3gcwQtV2VjMTw)
        - [2025-04-08，【CUDA编程】Flash Attention CUDA 算子的实现思路](https://mp.weixin.qq.com/s/ReVeJ9fNlWeyfP8CMLolkg)
    - 微信公众号「江大白」
        - [2023-09-06，GPU底层优化，如何让Transformer在GPU上跑得更快？](https://mp.weixin.qq.com/s/Xdbkld6ZrJ7Q93PEOedBMA)
        - [2024-04-12，深入浅出，PyTorch模型int8量化原理拆解](https://mp.weixin.qq.com/s/j2QS3LdudrrlyZYQkVrl5Q)
        - [2024-04-13，CUDA模型部署实战，自己写的CUDA矩阵乘法能优化到多快？](https://mp.weixin.qq.com/s/ySfGSHyLrW5cRG17-B14rQ)
        - [2024-04-22，CUDA编程中，Tensor Cores的详细拆解](https://mp.weixin.qq.com/s/uDWOg9-pRudcvroZADsIbg)
        - [2024-06-22，FP8量化解读，8bit下部署最优方案？](https://mp.weixin.qq.com/s/5DdMXCRq7X6QkS2yXJqF7g)
        - [2024-06-26，Cuda编程实践，我的第一份Cuda代码](https://mp.weixin.qq.com/s/JxpNDmDTiS-ctCCG-RY1nw)
        - [2025-01-03，AI项目工程化，CUDA开发心得汇总！](https://mp.weixin.qq.com/s/9mSV2ZY6EdBAAbtgq4gkZg)
    - 微信公众号「Tim在路上」
        - [2024-03-25，理解NVIDIA GPU 性能：利用率与饱和度](https://mp.weixin.qq.com/s/4_An51JuRGWTU0dLgZYHpQ)
        - [2024-04-30，加速矩阵计算：英伟达TensorCore架构演进与原理最全解析](https://mp.weixin.qq.com/s/dwT1Fl6F4V1MvWGgt1ac0Q)
        - [2024-05-15，揭秘 Tensor Core 底层：如何让AI计算速度飞跃](https://mp.weixin.qq.com/s/UL7CLWp3cmdUgGILr4iVzA)
        - [2024-05-27，浅析GPU分布式通信技术-PCle、NVLink、NVSwitch](https://mp.weixin.qq.com/s/ZllBWNqBwiY-Cb0UFIkwVg)
        - [2024-09-09，使用Nsight Profiling工具对大模型进行性能调优](https://mp.weixin.qq.com/s/1t22WJCtqaWTP6UvFRnFeg)
    - 微信公众号「潮观世界」
        - [2024-04-19，AI 推理：CPU 的崛起](https://mp.weixin.qq.com/s/rpdCT1fj2E3GKknfygAWRw)
    - 微信公众号「DeepDriving」
        - [2023-07-21，AI模型部署 | TensorRT模型INT8量化的Python实现](https://mp.weixin.qq.com/s/IQTCUs8CcfgHxJCyV6cm3w)
    - 微信公众号「人工智能大讲堂」
        - [2024-05-11，我找到了AlexNet当年的源代码，没用框架，从零手撸CUDA/C++](https://mp.weixin.qq.com/s/plxXG8y5QlxSionyjyPXqw)
    - 微信公众号「未来科技潮」
        - [2024-04-10，针对大型语言模型的高效CUDA优化可实现性能翻倍提升](https://mp.weixin.qq.com/s/FtJpRrfnFACM37p9fQv3cw)
        - [2024-06-21，解密高性能计算：如何用流和Kernel触发提升GPU通信效率](https://mp.weixin.qq.com/s/X3A8Dc_48oHMo2arPFqHoQ)
    - 微信公众号「AI道上」
        - [2024-04-19，英伟达坚持了16年的CUDA，到底是什么](https://mp.weixin.qq.com/s/nsBxZe_UXdvwfQ7DmCFuQg)
    - 微信公众号「科技译览」
        - [2024-04-09，100行C代码重塑深度学习：用纯C/CUDA打造的极简LLM训练](https://mp.weixin.qq.com/s/Th3RX3_FS5git0qJEcu4ZA)
    - 微信公众号「小白学视觉」
        - [2024-03-29，图像预处理库CV-CUDA开源了，打破预处理瓶颈，提升推理吞吐量20多倍](https://mp.weixin.qq.com/s/Zn4yI1xu2TuXZkJCzQt_yA)
        - [2025-01-24，算法岗平时需要自己写cuda吗？](https://mp.weixin.qq.com/s/L_SoagU1V0lZA75-7qef3A)
    - 微信公众号「卡巴斯」
        - [2024-02-26，GPU（一）GPU简介](https://mp.weixin.qq.com/s/V4mMjzQ261kk6qmyH-STUQ)
    - 微信公众号「码砖杂役」
        - [2024-04-03，【CUDA】一文讲清流与并发，讲不清我重讲](https://mp.weixin.qq.com/s/-eJOdG7A-bvum9GFkiNIoQ)
        - [2024-05-02，【【CUDA】一文讲清共享内存和常量内存](https://mp.weixin.qq.com/s/qcynKSz2zQQQ2Ylk_sSorw)
    - 微信公众号「星想法」
        - [2022-09-19，零知识证明 - FPGA vs. GPU](https://mp.weixin.qq.com/s/SjoeQHboe2RI4EJKfpMjKw)
    - 微信公众号「太极图形」
        - [2022-06-16，减少重复造轮子，帮你解放生产力的「小矩阵功能」来啦！](https://mp.weixin.qq.com/s/5PGXUxcUMSfsbVbrennUFA)
    - 微信公众号「硅星人Pro」
        - [2024-06-03，黄仁勋：英伟达将一年推一款全新芯片，没有英伟达就没有今天AI的一切（附最新演讲全文）](https://mp.weixin.qq.com/s/Uc6heL537JNn63JXsDSVOg)
    - 微信公众号「3D视觉之心」
        - [2024-06-01，传统SLAM使用CUDA加速是否有比较大的优势呢？](https://mp.weixin.qq.com/s/5SlVcsDJd8VvABo6wCe4AQ)
    - 微信公众号「中国企业家杂志」
        - [2024-06-01，黄仁勋：不喜欢裁员，我宁愿“折磨”他们｜中企荐读](https://mp.weixin.qq.com/s/8jIgJPsWuCnj92wa61llSw)
    - 微信公众号「CSharp与边缘模型部署」
        - [2024-06-04，使用 TensorRT C++ API 调用GPU加速部署 YOLOv10 实现 500FPS 推理速度——快到飞起！！](https://mp.weixin.qq.com/s/yijeZtkRhbQxuSE1AsyUhA)
    - 微信公众号「NeuralTalk」
        - [2023-06-16，SIMD 指令集与数据并行程序](https://mp.weixin.qq.com/s/dgTtEY5NZh-npQ6KN2WoaA)
    - 微信公众号「小吴持续学习AI」
        - [2023-06-12，为CUDA Kernel选择合适的grid_size和block_size](https://mp.weixin.qq.com/s/Je0ZCPv6RKacX__TFL1y4A)
    - 微信公众号「大模型新视界」
        - [2024-06-20，大模型量化性能评价指标](https://mp.weixin.qq.com/s/S76alcWhBdM5gWJvT0udAQ)
        - [2024-06-24，FP8 量化基础 - 英伟达](https://mp.weixin.qq.com/s/MnOze4BGP-a7Un4K0sakbg)
        - [2024-07-05，聊聊大模型推理中的分离式推理](https://mp.weixin.qq.com/s/4vO3j4LXcmsZ97WfabZzfA)
        - [2024-07-11，FP8 低精度训练：Transformer Engine 简析](https://mp.weixin.qq.com/s/r836OOVNo9z_HHTX-MtO-A)
    - 微信公众号「量子位」
        - [2024-06-17，黄仁勋致毕业生：勇于进入0亿美元市场，希望你能找到自己的GPU](https://mp.weixin.qq.com/s/m7ySazb1DrsLUQHqSW37mg)
    - 微信公众号「HPC智能流体大本营」
        - [2024-03-26，GPU 上 GEMM 的性能优化指标](https://mp.weixin.qq.com/s/0sNkjkE9LJ3o6_w5uR_XgA)
    - 微信公众号「人工智能前沿讲习」
        - [2023-07-06，【他山之石】CUDA SGEMM矩阵乘法优化笔记——从入门到cublas](https://mp.weixin.qq.com/s/0iAbHZ-hN6Mj2c2j2Uw03w)
    - 微信公众号「AI让生活更美好」
        - [2024-07-06，Thrust 库：让 C++ 并行计算飞跃](https://mp.weixin.qq.com/s/GtMolxSU-VKjs0cheMOykg)
    - 微信公众号「NE时代智能车」
        - [2024-07-09，理想是如何将视觉语言大模型部署到Orin-X上的？](https://mp.weixin.qq.com/s/EBnfgXY_fxlQI-7eykwqZA)
    - 微信公众号「OpenCV与AI深度学习」
        - [2024-07-08，实战 | YOLOv8使用TensorRT加速推理教程（步骤 + 代码）](https://mp.weixin.qq.com/s/VcUifHycY9aw99d3WD1h1w)
        - [2024-07-10，OpenCV使用CUDA加速资料汇总(pdf+视频+源码)](https://mp.weixin.qq.com/s/o-AECBLDucxVLr1Q0yxZ_g)
    - 微信公众号「InfiniTensor」
        - [2024-07-19，softmax算子开发介绍](https://mp.weixin.qq.com/s/KT5rzTj3jNj_aw5hXm-kTw)
        - [2024-07-24，CUDA实现matmul的并行策略](https://mp.weixin.qq.com/s/U_-NnW2yx3jnc1vCfEi1Cg)
        - [2024-07-27，flash attention的CUDA编程](https://mp.weixin.qq.com/s/RRP45uuC-KgKZ88bzTLgUQ)
        - [2024-07-30，CUDA实现规约的并行策略](https://mp.weixin.qq.com/s/OslgzL-qXV9KNrzgb3Fsvg)
    - 微信公众号「GeekSavvy」
        - [2024-03-19，史上最强芯片推出！英伟达发布新一代BlackWell GPU](https://mp.weixin.qq.com/s/Ld2P68IBzFTDYrx3rY8rNw)
    - 微信公众号「阿木实验室」
        - [2024-07-31，搭载英伟达Jetson Orin的Allspark 2全新亮相，算力高达100TOPS！](https://mp.weixin.qq.com/s/hjU3eH1LAvxVsOHWZXFAyQ)
    - 微信公众号「吃果冻不吐果冻皮」
        - [2024-08-09，如何把 PyTorch 的 GPU 利用率提升到 100% ?](https://mp.weixin.qq.com/s/9HSZppiFjypwu-TttbxqMQ)
        - [2024-08-26，基于 NVIDIA TensorRT-LLM 的大语言模型调度方法](https://mp.weixin.qq.com/s/nDSYDsmAtozmHNzLsx4KzQ)
    - 微信公众号「AI大模型实验室」
        - [2024-03-19，Nvidia推出Blackwell B200 GPU，是目前最强的人工智能芯片](https://mp.weixin.qq.com/s/SPgz7q_Ycr2lNxeH0WoNEQ)
    - 微信公众号「科技最前线」
        - [2024-03-19，能超越英伟达的只有英伟达](https://mp.weixin.qq.com/s/ORMH0JKdeS-i_DAjdcNvXg)
    - 微信公众号「AI范儿」
        - [2024-03-20，NVIDIA 与 Blackwell 一起改写摩尔定律](https://mp.weixin.qq.com/s/3ffnbFdnXV7lQylzo_3o9w)
    - 微信公众号「DataFunTalk」
        - [2024-03-15，NVIDIA大语言模型落地的全流程解析](https://mp.weixin.qq.com/s/mhGcW8FqLigBeePlRZGBDg)
    - 微信公众号「先进编译实验室」
        - [2025-01-20，玩转GPU加速：CUDA编程优化之旅开启](https://mp.weixin.qq.com/s/tke62pNpdM1AW4VPt8MHlg)
        - [2025-02-04，CUDA优化秘籍：解锁内存性能的四大关键](https://mp.weixin.qq.com/s/fI7mBYM6SrexCgZRjoccCw)
    - 微信公众号「讳疾忌医-note」
        - [2024-12-21，国外深度学习工程师分享：从零开始重现 PyTorch（支持 GPU）超详细](https://mp.weixin.qq.com/s/FAfADW1jPt40RFxwyfCGJQ)
    - 微信公众号「DeepHub IMBA」
        - [2023-10-22，使用TensorRT-LLM进行高性能推理](https://mp.weixin.qq.com/s/pIZ9ceJzTG8kMZMn1m5oQw)
    - 微信公众号「zarbot」
        - [2024-07-25，Tensor-001 矩阵乘法分块乘法概述](https://mp.weixin.qq.com/s/61CzHIZcX6DSUBcjbgr2eQ)
        - [2024-07-25，Tensor-002 矩阵乘法优化](https://mp.weixin.qq.com/s/tpzBq_rZdZt3whqw0txCDQ)
        - [2024-08-03，Tensor-003 TensorCore架构](https://mp.weixin.qq.com/s/xdC8SlMQb7O0S3E5VXXpRQ)
        - [2024-08-10，Tensor-004 TensorCore编程及优化](https://mp.weixin.qq.com/s/eB6nbPpVwPqLyo9YMiw93w)
        - [2024-08-20，Tensor-005 CUTLASS简介](https://mp.weixin.qq.com/s/LaK8jucDgrRxPlDMq-aTvw)
        - [2024-08-22，Tensor-006 AI软硬件交互界面: 可组合的Kernel](https://mp.weixin.qq.com/s/JA8wCZFAlkTVwfW8tmke9w)
        - [2024-08-24，Tensor-007 Cute Layout简介](https://mp.weixin.qq.com/s/3uJtWOyhB9PX72ynocjVZg)
        - [2024-08-29，Tensor-008 CuTe Layout代数](https://mp.weixin.qq.com/s/CWBxDkd1mj0WPmSAUHSHhg)
        - [2024-09-09，Tensor-009 Cute Tensor](https://mp.weixin.qq.com/s/sd7HY68Z-UZiqlv_PfoayA)
        - [2024-09-14，Tensor-010 Tensor Copy](https://mp.weixin.qq.com/s/90D7Ro_kkbM1rj4szyeDRw)
    - 微信公众号「Infra研习社」
        - [2025-03-24，cuda-python 使用与cuTicle介绍](https://mp.weixin.qq.com/s/Q_fSDMy6FNPqgmAW_kaPUw)



    - [知乎「紫气东来」](https://www.zhihu.com/people/zi-qi-dong-lai-1)
        - [2023-09-02，CUDA（一）：CUDA 编程基础](https://zhuanlan.zhihu.com/p/645330027)
        - [2023-09-09，CUDA（二）：GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)
        - [2023-09-29，CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)
        - [2024-04-29，ops(1)：LayerNorm 算子的 CUDA 实现与优化](https://zhuanlan.zhihu.com/p/694974164)
        - [2024-04-30，ops(2)：SoftMax算子的 CUDA 实现](https://zhuanlan.zhihu.com/p/695307283)
        - [2024-05-01，ops(3)：Cross Entropy 的 CUDA 实现](https://zhuanlan.zhihu.com/p/695594396)
        - [2024-05-01，ops(4)：AdamW 优化器的 CUDA 实现](https://zhuanlan.zhihu.com/p/695611950)
        - [2024-05-02，ops(5)：激活函数与残差连接的 CUDA 实现](https://zhuanlan.zhihu.com/p/695703671)
        - [2024-05-03，ops(6)：embedding 层与 LM head 层的 CUDA 实现](https://zhuanlan.zhihu.com/p/695785781)
        - [2024-05-06，ops(7)：self-attention 的 CUDA 实现及优化 (上)](https://zhuanlan.zhihu.com/p/695898274)
        - [2024-05-08，ops(8)：self-attention 的 CUDA 实现及优化 (下)](https://zhuanlan.zhihu.com/p/696197013)
        - [2024-05-14，CUDA（四）：使用 CUDA 实现 Transformer 结构](https://zhuanlan.zhihu.com/p/694416583)
    - [知乎「是聪明貂吖」](https://www.zhihu.com/people/cmd23333)
        - [2024-02-18，《高性能并行编程与优化》课程笔记目录](https://zhuanlan.zhihu.com/p/671684145)


    - [Medium Blog](https://medium.com/)
        - [2024-05-15，Recreating PyTorch from Scratch (with GPU Support and Automatic Differentiation)](https://medium.com/towards-data-science/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc)



  - ### Triton Blogs

    - 微信公众号「智源研究院」
        - [2024-07-09，智源打造基于Triton的大模型算子库，助力AI芯片软硬件生态建设](https://mp.weixin.qq.com/s/03EYxAyu4uWI4tjMkFZjSQ)
    - 微信公众号「智源FlagOpen」
        - [2024-09-06，智源打造基于Triton的大模型算子库，助力AI芯片软硬件生态建设](https://mp.weixin.qq.com/s/QMzgj-h6q2mZHokS5QpXmg)
        - [2024-09-18，Triton大会@硅谷：芯片、AI大厂齐站台](https://mp.weixin.qq.com/s/euX2nxQ4lhG6yaLYMugyrw)
        - [2024-11-20，Triton活动｜Triton中国社区贡献者茶话会](https://mp.weixin.qq.com/s/ASU1Z9B21kD8WJMhZ5E_ZA)
        - [2024-12-04，Triton中国社区贡献者茶话会圆满落地](https://mp.weixin.qq.com/s/vtSMaGgp_Uxo_QvmmV7sjg)
        - [2024-12-10，Triton入门实践 | 算子性能优化：自动调优的艺术](https://mp.weixin.qq.com/s/txtyci4MUCVNpTZwF6fS4Q)
        - [2024-12-19，智源大模型通用算子库FlagGems四大能力升级，为AI系统开源生态注入新活力](https://mp.weixin.qq.com/s/2X5DyY1MbYhz6zVusPpKww)
        - [2025-01-13，Triton生态 | 突破CUDA束缚，拥抱开放与多元的未来](https://mp.weixin.qq.com/s/IX3kalsNUcXJ_JgDWKJ1iQ)
    - 微信公众号「PyTorch」
        - [2025-01-24，Triton加速2D动态块量化Float8 GEMM](https://mp.weixin.qq.com/s/jixXXEBId9PHhK40L77bEQ)
    - 微信公众号「摩尔线程」
        - [2024-07-18，摩尔线程 × 智源研究院｜完成基于Triton的大模型算子库适配](https://mp.weixin.qq.com/s/84LKQ4Xo1RSdNoFJG0tUmg)
        - [2024-11-05，开源vLLM-MUSA｜摩尔线程持续加速基于国产GPU的AI大模型推理开发](https://mp.weixin.qq.com/s/5YLCXLlkbZ9WwoaiSIFJhA)
        - [2024-11-12，开源MUTLASS｜摩尔线程加速基于国产GPU的算子开发以及算法创新](https://mp.weixin.qq.com/s/Nm8BKAJD_ibht8pG9CfEFQ)
    - 微信公众号「HyperAI超神经」
        - [2024-10-14，首个完整 Triton 中文文档上线！开启 GPU 推理加速新时代](https://mp.weixin.qq.com/s/ytuxAMlpss5Il_cWf8Y1YQ)
    - 微信公众号「InfiniTensor」
        - [2024-08-22，OpenAI Triton 简介（一）](https://mp.weixin.qq.com/s/WWPoIcqUDSYQsUI9HbN5-g)
        - [2024-10-24，OpenAI Triton 简介（二）](https://mp.weixin.qq.com/s/XVzfsVFWDsteOIXAkqTYIw)
    - 微信公众号「吃果冻不吐果冻皮」
        - [2023-05-22，模型推理服务化框架Triton保姆式教程（一）：快速入门 ](https://mp.weixin.qq.com/s/YES9OO9NX6-HnzR-pvfFyQ)
        - [2023-06-02，模型推理服务化框架Triton保姆式教程（二）：架构解析](https://mp.weixin.qq.com/s/BVKLsQ9GBN_VqStdjielrA)
        - [2023-06-03，模型推理服务化框架Triton保姆式教程（三）：开发实践](https://mp.weixin.qq.com/s/2jAw7tf4Pvd1o1AmukTiDw)
    - 微信公众号「GiantPandaCV」
        - [2024-01-22，【BBuf的CUDA笔记】十三，OpenAI Triton 入门笔记一](https://mp.weixin.qq.com/s/RMR_n1n6nBqpdMl6tdd7pQ)
        - [2024-10-08，【翻译】【PyTorch 奇技淫巧】FlexAttetion 基于Triton打造灵活度拉满的Attention](https://mp.weixin.qq.com/s/KJUk-jmwGPrJvVuLQ44DyQ)
    - 微信公众号「新智元」
        - [2024-09-06，PyTorch官宣：告别CUDA，GPU推理迎来Triton加速新时代](https://mp.weixin.qq.com/s/AiViRkOOpmIUm8DfRUu4JA)
    - 微信公众号「CV技术指南」
        - [2024-09-08，PyTorch官宣：告别CUDA，GPU推理迎来Triton加速新时代](https://mp.weixin.qq.com/s/TABVE-pvXqNmsnkdkv_9MA)
    - 微信公众号「AI时代窗口」
        - [2024-09-10，不依赖CUDA的大模型推理已经实现](https://mp.weixin.qq.com/s/TrzwSBQ301Grcpye9-pt1Q)
    - 微信公众号「先进编译实验室」
        - [2025-01-04，探索 Triton 编程密码：语法与实践指南大揭秘](https://mp.weixin.qq.com/s/e2iZNBdiBPRmF_8GrLCKfw)
        - [2025-01-06，深入Triton源码：揭开AI加速引擎的神秘面纱！](https://mp.weixin.qq.com/s/tSQecwSrpfJVVxD4_4blWQ)

    - [知乎「Soaring」](https://www.zhihu.com/people/soaring-52-57)
        - [2024-05-14，OpenAI Triton 入门](https://zhuanlan.zhihu.com/p/697626885)


  - ### TVM Blogs

    - 微信公众号「小喵学AI」
        - [2023-05-04，实战 | TVM优化Pytorch模型](https://mp.weixin.qq.com/s/I-YXPJel_g0rGS9Vy8BE3Q)



  - ### MLIR Blogs

    - 微信公众号「GiantPandaCV」
        - [2022-05-23，基于 MLIR 完成对 GEMM 的编译优化 中英视频上，中部分](https://mp.weixin.qq.com/s/9wyM3hKsJA0YxFsms1Rpuw)
        - [2023-06-25，MLIR_对自定义IR Dialect编写bufferization pass](https://mp.weixin.qq.com/s/3aHwYDkI9K3u-10v6-9iVA)




  - ### HPC Blogs

    - 微信公众号「RVBoards」
        - [2021-03-23，张先轶博士：OpenBLAS项目与矩阵乘法优化](https://mp.weixin.qq.com/s/20SX_FL4cEDUx9pDJpOxnA)
    - 微信公众号「猿禹宙」
        - [2023-11-11， 朱懿：HPC之矩阵乘法高性能实验报告](https://mp.weixin.qq.com/s/WoDacoBqAJeV4PgNGtDq_A)
    - 微信公众号「NeuralTalk」
        - [2023-06-16，SIMD 指令集与数据并行程序](https://mp.weixin.qq.com/s/dgTtEY5NZh-npQ6KN2WoaA)
    - 微信公众号「有限元语言与编程」
        - [2024-05-21，并行计算：超级大脑背后的魔术师](https://mp.weixin.qq.com/s/GnnJtXr6BZrnGsHJB-a-ag)
        - [2024-06-29，BLAS简介：基于Fortran的高性能矩阵计算基础库](https://mp.weixin.qq.com/s/FXkxeezDVEY7asjl_PWX1g)
        - [2024-07-08，LAPACK简介：基于Fortran的高性能线性代数工具箱](https://mp.weixin.qq.com/s/iAxHrRFmVtcpX8otZytHvw)
    - 微信公众号「鸟窝聊技术」
        - [2024-07-12，使用SIMD优化二叉搜索树](https://mp.weixin.qq.com/s/u8BcfQKmtWIB86B4GetULQ)
    - 微信公众号「OpenCV与AI深度学习」
        - [2024-06-21，YOLOv10在PyTorch和OpenVINO中推理对比](https://mp.weixin.qq.com/s/xZ4HlfBPXFbf8OPxmXwbrQ)

    - [知乎「白牛」](https://www.zhihu.com/people/huan-jun-81)
        - [2023-05-04，OpenBLAS gemm从零入门](https://zhuanlan.zhihu.com/p/65436463)
    - [知乎「庄碧晨」](https://www.zhihu.com/people/zhuang-chen-84-13)
        - [2021-01-22，多线程 GEMM 论文 笔记](https://zhuanlan.zhihu.com/p/346254572)
    - [知乎「OeuFcoque」](https://www.zhihu.com/people/fsybdh)
        - [2020-04-12，高性能计算简介（一）：初步分析，BLAS，BLIS简介](https://zhuanlan.zhihu.com/p/129187064)
    - [知乎「赵小明12138」](https://www.zhihu.com/people/zhao-qi-ming-67)
        - [2022-10-26，并行计算-canon算法：矩阵相乘](https://zhuanlan.zhihu.com/p/577512867)
    - [知乎「zero」](https://www.zhihu.com/people/zero-35-40)
        - [2021-12-18，稠密矩阵乘003(gemm)-OpenBLAS和BLIS分块策略](https://zhuanlan.zhihu.com/p/446908156)
    - [知乎「严忻恺」](https://www.zhihu.com/people/yan-xin-kai-38)
        - [2022-03-31，斯坦福CS217(三)GEMM计算加速](https://zhuanlan.zhihu.com/p/280771849)
    - [黎明灰烬 博客](https://zhenhuaw.me/)
        - [2019-06-12，通用矩阵乘（GEMM）优化算法](http://zhenhuaw.me/blog/2019/gemm-optimization.html)
    - [Modular Blog](https://www.modular.com/blog)
        - [2023-03-23，AI’s compute fragmentation: what matrix multiplication teaches us](https://www.modular.com/blog/ais-compute-fragmentation-what-matrix-multiplication-teaches-us)
        - [2023-04-20，The world's fastest unified matrix multiplication](https://www.modular.com/blog/the-worlds-fastest-unified-matrix-multiplication)
        - [2023-05-02，A unified, extensible platform to superpower your AI](https://www.modular.com/blog/a-unified-extensible-platform-to-superpower-your-ai)
        - [2023-08-18，How Mojo🔥 gets a 35,000x speedup over Python – Part 1](https://www.modular.com/blog/how-mojo-gets-a-35-000x-speedup-over-python-part-1)
        - [2023-08-28，How Mojo🔥 gets a 35,000x speedup over Python – Part 2](https://www.modular.com/blog/how-mojo-gets-a-35-000x-speedup-over-python-part-2)
        - [2023-09-06，Mojo🔥 - A journey to 68,000x speedup over Python - Part 3](https://www.modular.com/blog/mojo-a-journey-to-68-000x-speedup-over-python-part-3)
        - [2024-02-12，Mojo vs. Rust: is Mojo 🔥 faster than Rust 🦀 ?](https://www.modular.com/blog/mojo-vs-rust-is-mojo-faster-than-rust)
        - [2024-04-10，Row-major vs. column-major matrices: a performance analysis in Mojo and NumPy](https://www.modular.com/blog/row-major-vs-column-major-matrices-a-performance-analysis-in-mojo-and-numpy)




## Videos

  - bilibili「深圳王哥的科技频道」
    - [2022-06-24，【张先轶】BLISlab学习优化矩阵乘。第一课](https://www.bilibili.com/video/BV1c94y117Uw)
    - [2022-06-24，【张先轶】BLISlab学习优化矩阵乘。第二课](https://www.bilibili.com/video/BV1BY411N72y)
    - [2022-06-24，【张先轶】BLISlab学习优化矩阵乘。第三课](https://www.bilibili.com/video/BV1b94y117BK)
    - [2022-10-19，【张先轶】BLISlab学习矩阵乘。第四课](https://www.bilibili.com/video/BV1oe4y1v7Dm)
    - [2022-09-08，【张先轶】OpenBLAS快速入门](https://www.bilibili.com/video/BV1Ze4y1h7GF)
  - bilibili「HITsz-OSA」
    - [2022-07-07，稠密矩阵乘在单核上的优化](https://www.bilibili.com/video/BV17U4y1D7T8)
  - bilibili「权双」
    - [2023-07-14，CUDA编程基础入门系列（持续更新）](https://www.bilibili.com/video/BV1sM4y1x7of)




## Interview

  - [Tongkaio/CUDA_Kernel_Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples) <img src="https://img.shields.io/github/stars/Tongkaio/CUDA_Kernel_Samples?style=social"/> : CUDA 算子手撕与面试指南。

  - 微信公众号「大模型生态圈」
    - [2024-04-21，推理部署工程师面试题库](https://mp.weixin.qq.com/s/q46vKFPlQhcN7LyZNTRhXA)
  - 微信公众号「Cver」
    - [2024-06-01，英伟达算法岗面试，问的贼细！](https://mp.weixin.qq.com/s/dwXC572U9u5SAmJPnyjHXA)
  - 微信公众号「高通内推王」
    - [2023-12-21，[英伟达内推] 英*达面试过程全面剖析](https://mp.weixin.qq.com/s/GoZKlLfdoGN9ngbe_PzG7w)
    - [2024-04-16，一份英伟达的offer，一年能到手多少钱](https://mp.weixin.qq.com/s/dZAG-AXbZkGi9CJQZMhCNA)
  - [知乎「Tim在路上​」](https://www.zhihu.com/people/lao-zhang-cao-mei-yuan)
    - [2024-01-18，国内大厂GPU CUDA高频面试问题汇总（含部分答案）](https://zhuanlan.zhihu.com/p/678602674)