from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="plasmidai",
    version="1.0.0",
    author="iGEM Toronto",
    author_email="Adibvafa.fallahpour@mail.utoronto.ca",
    description="A package for plasmid analysis using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igem-toronto/plasmidai",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'plasmidai=plasmidai.experimental.train:main',
        ],
    },
)