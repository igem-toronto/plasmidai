# Plasmid.ai

Plasmid.ai is the largest open-source toolkit for developing plasmid foundation models. Created by the iGEM Toronto team, this project aims to revolutionize the field of synthetic biology by leveraging machine learning to generate novel plasmids.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Plasmid.ai provides a comprehensive set of tools and models for the analysis, design, and generation of plasmids. By utilizing state-of-the-art machine learning techniques, this project enables researchers and synthetic biologists to explore new possibilities in plasmid engineering and design.

## Installation

### Using pip

To install the Plasmid.ai package, run the following command:

```bash
pip install plasmidai
```

### Using git

For development or to access the latest features, you can clone the repository:

```bash
git clone https://github.com/igem-toronto/plasmid-ai.git
cd plasmid-ai
pip install -e .
```

## Usage

Here's a basic example of how to use Plasmid.ai:

```python
import plasmidai as pai
```

## Project Structure

The Plasmid.ai project is organized into several key components:

- `data/`: Contains datasets and scripts for data processing.
  - `scripts/`: Helper scripts for data manipulation.
  - `tokenizers/`: Custom tokenizers for plasmid sequences.
- `datasets/`: Modules for loading and preprocessing plasmid datasets.
- `experimental/`: Cutting-edge features and models in development.
  - `callbacks.py`: Custom callbacks for model training.
  - `lit.py`: Lightning modules for PyTorch Lightning integration.
  - `optimizers.py`: Custom optimizers for training plasmid models.
  - `sample.py`: Functions for sampling from trained models.
  - `train.py`: Training pipelines for plasmid models.
- `utils.py`: Utility functions used across the project.
- `paths.py`: Path configurations for the project.

## Contributing

We welcome contributions from the community!
