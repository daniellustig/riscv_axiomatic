#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="rubigi",
    version="2021.1.29",
    author="Daniel Lustig",
    author_email="dlustig@nvidia.com",
    description="An axiomatic model of RISC-V concurrency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daniellustig/riscv_axiomatic",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
