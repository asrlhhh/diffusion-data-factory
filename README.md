# The Future of Data Labeling: Turning Your Foundation Model into a High-Volume Data Factory

This repository contains code, examples, and resources for a blog post. The blog explores the posibility of using foundation models, particularly large diffusion models like Stable Diffusion, to generate large-scale labeled datasets efficiently. It focuses on a use case involving the CelebA facial dataset and demonstrates how these generative models can produce high-quality, annotated facial images for various AI applications.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Overview

The primary goal of this repository is to provide an easy-to-understand and practical implementation of Stable Diffusion for data labeling. The ideas and techniques demonstrated in this repository are on top of the work done in the Semantic Image Editing project, which can be found at the [original GitHub repository](https://github.com/ml-research/semantic-image-editing). Additionally, the last part of the pipeline uses optional face restoration model [CodeFormer](https://github.com/sczhou/CodeFormer). Follow the instructions inside the repo to install CodeFormer.

## Installation

To get started with the code and examples, you'll need to install the required dependencies. We recommend using a virtual environment to manage your dependencies. You can install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

Also download the [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) file and put it under the root folder

For CodeFormer, please follow the instructions in its repo.

## Usage

Use **collect_seeds.py** file to generate initial annotated face images. Use **augment_seeds.py** file to use latent space interpolation to augment the existing data. 


## Citation

```bibtex
@article{brack2023Sega,
      title={SEGA: Instructing Diffusion using Semantic Dimensions}, 
      author={Manuel Brack and Felix Friedrich and Dominik Hintersdorf and Lukas Struppek and Patrick Schramowski and Kristian Kersting},
      year={2023},
      journal={arXiv preprint arXiv:2301.12247}
}
```

