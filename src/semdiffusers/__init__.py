from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

import PIL
from PIL import Image

from diffusers.utils import BaseOutput, is_torch_available, is_transformers_available


@dataclass
class SemanticEditPipelineOutput(BaseOutput):
    """
    Output class for Latent editing pipeline.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        inappropriate_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents inappropriate content,
            or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    inappropriate_content_detected: Optional[List[bool]]


if is_transformers_available() and is_torch_available():
    from .pipeline_latent_edit_diffusion import SemanticEditPipeline
