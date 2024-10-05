<p align="center">
  <img src="https://github.com/user-attachments/assets/7c6114cf-f795-45d5-9c11-a31b3cf48f34" alt="Plasmid.ai Logo" width="30%" height="30%" style="vertical-align: middle;"/>
</p>

<p align="center">
  <a href="https://2024.igem.wiki/utoronto/"><img src="https://img.shields.io/badge/iGEM-2024%20Wiki-00B89E?style=for-the-badge&logo=wikipedia&logoColor=white" alt="iGEM 2024 Wiki"></a>
  <a href="https://github.com/igem-toronto/plasmidai"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://pypi.org/project/plasmidai/"><img src="https://img.shields.io/badge/PyPI-Package-FFBF00?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI Package"></a>
</p>

Plasmid.ai is the largest open-source toolkit for developing plasmid foundation models. Created by the iGEM Toronto team, this project aims to revolutionize the field of synthetic biology by leveraging machine learning to generate novel plasmids.
<br><br>

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Authors and acknowledgment](#authors-and-acknowledgment)
- [License](#license)
<br><br>

## Overview

Plasmid.ai provides a comprehensive set of tools and models for the analysis, design, and generation of plasmids. By utilizing state-of-the-art machine learning techniques, this project enables researchers and synthetic biologists to explore new possibilities in plasmid engineering and design. For more information about our team and project, visit our [iGEM Team Wiki](https://igem.skule.ca/).
<br><br>

## Features

- **Plasmid Sequence Tokenization**: Utilizes custom tokenizers tailored for encoding plasmid sequences.
- **Data Preprocessing Pipelines**: Includes robust modules for loading, preprocessing, and visualizing plasmid data.
- **Advanced Sampling Techniques**: Provides cutting-edge sampling functions for generating novel plasmids based on trained models.
- **Lightning Integration**: Seamlessly integrates with PyTorch Lightning for distributed training and model scalability.
- **Custom Model Components**: Features specialized optimizers and callbacks for enhanced model performance.
<br><br>

## Installation

### Using pip

To install the Plasmid.ai package, run the following command:

```bash
pip install --upgrade pip setuptools wheel
pip install plasmidai
```

### Using git

For development or to access the latest features, you can clone the repository:

```bash
git clone https://github.com/igem-toronto/plasmidai.git
cd plasmidai
pip install --upgrade pip setuptools wheel
pip install -e .
```

You can use `conda` or `poetry` to manage dependencies.
<br><br>


## Usage
Here's a basic example of how to use Plasmid.ai:

```python
import plasmidai as pai

# Training
python -m pai.experimental.train \
    --backend.matmul_precision=medium \
    --data.batch_size=64 --data.num_workers=4 \
    --lit.fused_add_norm=true --lit.scheduler_span=50000 --lit.top_p=0.9 \
    --trainer.accelerator=gpu  --trainer.devices=2 --trainer.precision=bf16-mixed \
    --trainer.wandb=true --trainer.wandb_dir="${REPO_ROOT}/logs" \
    --trainer.checkpoint=true --trainer.checkpoint_dir="${REPO_ROOT}/checkpoints" \
    --trainer.progress_bar=true \
    --trainer.max_epochs=175

# Generation
python -m pai.experimental.sample \
    --backend.matmul_precision=medium \
    --sample.checkpoint_path="${REPO_ROOT}/checkpoints/last.ckpt" \
    --sample.precision=bfloat16 --sample.num_samples=10000 --sample.top_p=0.9 \
    --sample.wandb_dir="${REPO_ROOT}/logs"
```
Checkout the `slurm` directory for more examples!
<br><br>


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
<br><br>


## Authors and acknowledgment

This project is developed by the iGEM Toronto 2024 team. We would like to extend our gratitude to all the team members and contributors who have made this project possible. Special thanks to our mentors and collaborators for their guidance and support.
<br><br>


## Contributing

We welcome contributions from the community! Please open an issue first.
<br><br>

## License

We use the Apache-2.0 license.
