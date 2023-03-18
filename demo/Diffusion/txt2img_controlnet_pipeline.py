#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import PIL
import time
import torch
import tensorrt as trt
from utilities import TRT_LOGGER
from stable_diffusion_pipeline import StableDiffusionPipeline

from diffusers.utils import PIL_INTERPOLATION

class Txt2ImgControlNetPipeline(StableDiffusionPipeline):
    """
    Application showcasing the acceleration of Stable Diffusion Txt2Img v1.4, v1.5, v2.0, v2.0-base, v2.1, v2.1-base pipeline using NVidia TensorRT w/ Plugins.
    """
    def __init__(
        self,
        scheduler="DDIM",
        *args, **kwargs
    ):
        """
        Initializes the Txt2Img Diffusion pipeline.

        Args:
            scheduler (str):
                The scheduler to guide the denoising process. Must be one of the [DPM, LMSD, DDIM, EulerA, PNDM].
        """
        super(Txt2ImgControlNetPipeline, self).__init__(*args, **kwargs, \
            scheduler=scheduler, stages=['clip','unet','vae','controlnet'])

    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype):
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        return image

    def infer(
        self,
        prompt,
        negative_prompt,
        image_height,
        image_width,
        seed=None,
        warmup=False,
        verbose=False,
        controlnet_image=None
    ):
        """
        Run the diffusion pipeline.

        Args:
            prompt (str):
                The text prompt to guide image generation.
            negative_prompt (str):
                The prompt not to guide the image generation.
            image_height (int):
                Height (in pixels) of the image to be generated. Must be a multiple of 8.
            image_width (int):
                Width (in pixels) of the image to be generated. Must be a multiple of 8.
            seed (int):
                Seed for the random generator
            warmup (bool):
                Indicate if this is a warmup run.
            verbose (bool):
                Verbose in logging
        """
        assert len(prompt) == len(negative_prompt)

        with torch.inference_mode(), torch.autocast("cuda"), trt.Runtime(TRT_LOGGER):
            # Pre-initialize latents
            latents = self.initialize_latents( \
                batch_size=len(prompt), \
                unet_channels=4, \
                latent_height=(image_height // 8), \
                latent_width=(image_width // 8)
            )

            torch.cuda.synchronize()
            e2e_tic = time.perf_counter()

            # CLIP text encoder
            text_embeddings = self.encode_prompt(prompt, negative_prompt)


            controlnet_image = self.prepare_image(controlnet_image,
                                    image_width,
                                    image_height,
                                    len(prompt) * 1,
                                    1,
                                    self.device,
                                    torch.float16,
            )
            controlnet_image = torch.cat([controlnet_image] * 2)

            # UNet denoiser
            latents = self.denoise_latent(latents, text_embeddings, controlnet_image=controlnet_image, controlnet_conditioning_scale=1.0)

            # VAE decode latent
            images = self.decode_latent(latents)

            torch.cuda.synchronize()
            e2e_toc = time.perf_counter()

            if not warmup:
                self.print_summary(self.denoising_steps, e2e_tic, e2e_toc)
                self.save_image(images, 'txt2img', prompt)
