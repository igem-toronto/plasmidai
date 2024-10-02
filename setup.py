import os
from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, "README.md")

    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="plasmidai",
    version="1.2.0",
    packages=["plasmidai"],
    author="iGEM Toronto",
    author_email="Adibvafa.fallahpour@mail.utoronto.ca",
    description="The largest open-source library to develop plasmid foundation models and generate novel plasmids using machine learning.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/igem-toronto/plasmidai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
)