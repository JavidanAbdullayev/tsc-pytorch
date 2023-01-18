from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.22'
DESCRIPTION = 'Time series classification using Deep Learning'


# Setting up
setup(
    name="test-tscai-1",
    version=VERSION,
    author="JavidanAbdullayev (Javidan Abdullayev)",
    author_email="<javidan.abdullayev@uha.fr>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'sklearn', 'importlib', 'torch', 'torchsummary',],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    include_package_data=True,
    package_data={'': ['data/*.tsv']},

)
