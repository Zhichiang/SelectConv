from setuptools import  setup
import torch
import os, glob
from torch.utils.cpp_extension import (CUDAExtension, CppExtension, BuildExtension)
from torch.utils.cpp_extension import include_paths, library_paths
# 这三个extension很重要！

def get_extensions():
    extensions = []
    ext_name = 'conv_extension'  # 编译后保存的文件前缀名称及其位置
    # prevent ninja from using too many resources
    # os.environ.setdefault('MAX_JOBS', '4')
    define_macros = []

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        define_macros += [('WITH_CUDA', None)]
        # 宏处理，会在每个.h/.cpp/.cu/.cuh源文件前添加 #define WITH_CUDA！！这个操作很重要
        # 这样在拓展的源文件中就可以通过#ifdef WITH_CUDA来判断是否编译代码
        op_files = []
        op_files.append("select_conv.cpp")
        op_files += glob.glob('./cuda/*.cu') + \
            glob.glob('./source/*.cpp')
        extension = CUDAExtension # 如果cuda可用，那么extension类型为CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = []
        op_files.append("select_conv.cpp")
        extension = CppExtension

    include_dirs = []
    include_dirs.append(os.path.abspath('./include'))
    include_dirs.append(os.path.abspath('./cuda'))
    include_dirs += include_paths(cuda=True)

    library_dirs = []
    library_dirs += library_paths(cuda=True)

    extra_compile_args = {'cxx': []}
    cuda_args = os.getenv('MMCV_CUDA_ARGS')
    extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
    extra_compile_args['cxx'] = ['-std=c++14']
    # prevent cub/thrust conflict with other python library
    # More context See issues #1454
    # extra_compile_args['nvcc'] += ['-Xcompiler=-fno-gnu-unique']

    ext_ops = extension( # 返回setuptools.Extension类
        name=ext_name,
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        language='c++',
        library_dirs=library_dirs)
    extensions.append(ext_ops)
    return extensions # 由setuptools.Extension组成的list

setup(
    name='select_conv',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension}, # BuildExtension代替setuptools.command.build_ext
    )

