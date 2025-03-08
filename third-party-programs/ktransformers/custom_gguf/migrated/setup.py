from setuptools import setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

setup(
    name='dequantize_extension',
    ext_modules=[
        SyclExtension(
            name='dequantize_extension',
            sources=[
                'src/bindings.cpp',
                'src/dequant.dp.cpp'
            ],
            # library_dirs=[
            #     '/home/chengxiw/hackathon/workspace/xputorch/lib/python3.10/site-packages/torch/lib'
            # ],
            libraries=[
                'torch_xpu', 'torch_cpu', 'c10_xpu', 'c10'
            ],
            extra_compile_args=['-fsycl'],
            extra_link_args=['-fsycl']
         ),
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    }
)