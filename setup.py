import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GOCPT",
    version="0.1.1",
    author="Chaoqi Yang",
    author_email="chaoqiy2@illinois.edu",
    description="A package for generalized online tensor decomposition and completion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ycq091044/GOCPT",
    packages=setuptools.find_packages(),
    install_requires=['urllib==3.7', 'numpy==1.14.4'],
    entry_points={
        'console_scripts': [
            'GOCPT=GOCPT:GOCPT'
        ],
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)