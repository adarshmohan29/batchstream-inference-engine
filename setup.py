from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="batchstream-inference-engine",
    package_dir={"": "src"},
    ext_modules=[
        CppExtension(
            "server.paged_attention_cpp",
            ["csrc/paged_attention.cpp"],
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
