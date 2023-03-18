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

from collections import OrderedDict
from copy import deepcopy
from diffusers.models import AutoencoderKL, UNet2DConditionModel, ControlNetModel
import numpy as np
from onnx import shape_inference
import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from cuda import cudart
import onnx

class Optimizer():
    def __init__(
        self,
        onnx_graph,
        verbose=False
    ):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs")

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

def get_path(version, inpaint=False):
    if version == "1.4":
        if inpaint:
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "CompVis/stable-diffusion-v1-4"
    elif version == "1.5":
        if inpaint:
            return "runwayml/stable-diffusion-inpainting"
        else:
            return "runwayml/stable-diffusion-v1-5"
    elif version == "2.0-base":
        if inpaint:
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2-base"
    elif version == "2.0":
        if inpaint:
            return "stabilityai/stable-diffusion-2-inpainting"
        else:
            return "stabilityai/stable-diffusion-2"
    elif version == "2.1":
        return "stabilityai/stable-diffusion-2-1"
    elif version == "2.1-base":
        return "stabilityai/stable-diffusion-2-1-base"
    else:
        raise ValueError(f"Incorrect version {version}")

def get_embedding_dim(version):
    if version in ("1.4", "1.5"):
        return 768
    elif version in ("2.0", "2.0-base", "2.1", "2.1-base"):
        return 1024
    else:
        raise ValueError(f"Incorrect version {version}")

class BaseModel():
    def __init__(
        self,
        hf_token,
        fp16=False,
        device='cuda',
        verbose=True,
        path="",
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
    ):
        self.name = "SD Model"
        self.hf_token = hf_token
        self.fp16 = fp16
        self.device = device
        self.verbose = verbose
        self.path = path

        self.min_batch = 1
        self.max_batch = max_batch_size
        self.min_image_shape = 256   # min image resolution: 256x256
        self.max_image_shape = 1024  # max image resolution: 1024x1024
        self.min_latent_shape = self.min_image_shape // 8
        self.max_latent_shape = self.max_image_shape // 8

        self.embedding_dim = embedding_dim
        self.text_maxlen = text_maxlen

    def get_model(self):
        pass

    def get_input_names(self):
        pass

    def get_output_names(self):
        pass

    def get_dynamic_axes(self):
        return None

    def get_sample_input(self, batch_size, image_height, image_width):
        pass

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        return None

    def get_shape_dict(self, batch_size, image_height, image_width):
        return None

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.cleanup()
        opt.info(self.name + ': cleanup')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        onnx_opt_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return onnx_opt_graph

    def check_dims(self, batch_size, image_height, image_width):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        assert image_height % 8 == 0 or image_width % 8 == 0
        latent_height = image_height // 8
        latent_width = image_width // 8
        assert latent_height >= self.min_latent_shape and latent_height <= self.max_latent_shape
        assert latent_width >= self.min_latent_shape and latent_width <= self.max_latent_shape
        return (latent_height, latent_width)

    def get_minmax_dims(self, batch_size, image_height, image_width, static_batch, static_shape):
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        latent_height = image_height // 8
        latent_width = image_width // 8
        min_image_height = image_height if static_shape else self.min_image_shape
        max_image_height = image_height if static_shape else self.max_image_shape
        min_image_width = image_width if static_shape else self.min_image_shape
        max_image_width = image_width if static_shape else self.max_image_shape
        min_latent_height = latent_height if static_shape else self.min_latent_shape
        max_latent_height = latent_height if static_shape else self.max_latent_shape
        min_latent_width = latent_width if static_shape else self.min_latent_shape
        max_latent_width = latent_width if static_shape else self.max_latent_shape
        return (min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, min_latent_height, max_latent_height, min_latent_width, max_latent_width)

class CLIP(BaseModel):
    def __init__(self,
        hf_token,
        device,
        verbose,
        path,
        max_batch_size,
        embedding_dim
    ):
        super(CLIP, self).__init__(hf_token, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "CLIP"

    def get_model(self):
        return CLIPTextModel.from_pretrained(self.path,
            subfolder="text_encoder",
            use_auth_token=self.hf_token).to(self.device)

    def get_input_names(self):
        return ['input_ids']

    def get_output_names(self):
       return ['text_embeddings', 'pooler_output']

    def get_dynamic_axes(self):
        return {
            'input_ids': {0: 'B'},
            'text_embeddings': {0: 'B'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, _, _, _, _ = self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'input_ids': [(min_batch, self.text_maxlen), (batch_size, self.text_maxlen), (max_batch, self.text_maxlen)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return {
            'input_ids': (batch_size, self.text_maxlen),
            'text_embeddings': (batch_size, self.text_maxlen, self.embedding_dim)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.zeros(batch_size, self.text_maxlen, dtype=torch.int32, device=self.device)

    def optimize(self, onnx_graph):
        opt = Optimizer(onnx_graph, verbose=self.verbose)
        opt.info(self.name + ': original')
        opt.select_outputs([0]) # delete graph output#1
        opt.cleanup()
        opt.info(self.name + ': remove output[1]')
        opt.fold_constants()
        opt.info(self.name + ': fold constants')
        opt.infer_shapes()
        opt.info(self.name + ': shape inference')
        opt.select_outputs([0], names=['text_embeddings']) # rename network output
        opt.info(self.name + ': remove output[0]')
        opt_onnx_graph = opt.cleanup(return_onnx=True)
        opt.info(self.name + ': finished')
        return opt_onnx_graph

def make_CLIP(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return CLIP(hf_token=hf_token, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
                max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version))

class UNet(BaseModel):
    def __init__(self,
        hf_token,
        fp16=False,
        device='cuda',
        verbose=True,
        path="",
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(UNet, self).__init__(hf_token, fp16=fp16, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim, text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.name = "UNet"
        self.controlnet = False

    def get_model(self):
        model_opts = {'revision': 'fp16', 'torch_dtype': torch.float16} if self.fp16 else {}
        return UNet2DConditionModel.from_pretrained(self.path,
            subfolder="unet",
            use_auth_token=self.hf_token,
            **model_opts).to(self.device)

    def get_input_names(self):
        input_names = ['sample', 'timestep', 'encoder_hidden_states']
        controlnet_input_names = ['dbar_0', 'dbar_1', 'dbar_2', 'dbar_3', 'dbar_4', 
                                      'dbar_5', 'dbar_6', 'dbar_7', 'dbar_8', 'dbar_9', 
                                      'dbar_10', 'dbar_11', 'mid_block_additional_residual']
        if self.controlnet:
            return input_names + controlnet_input_names
        return input_names
                

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        dynamic_axes = {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'latent': {0: '2B', 2: 'H', 3: 'W'},
        }
        controlnet_dynamic_axes = {'dbar_0': {0: '2B', 2: 'H', 3: 'W'},
            'dbar_1': {0: '2B', 2: 'H', 3: 'W'},
            'dbar_2': {0: '2B', 2: 'H', 3: 'W'},
            'dbar_3': {0: '2B', 2: 'H/2', 3: 'W/2'}, 
            'dbar_4': {0: '2B', 2: 'H/2', 3: 'W/2'}, 
            'dbar_5': {0: '2B', 2: 'H/2', 3: 'W/2'}, 
            'dbar_6': {0: '2B', 2: 'H/4', 3: 'W/4'}, 
            'dbar_7': {0: '2B', 2: 'H/4', 3: 'W/4'}, 
            'dbar_8': {0: '2B', 2: 'H/4', 3: 'W/4'}, 
            'dbar_9': {0: '2B', 2: 'H/8', 3: 'W/8'}, 
            'dbar_10': {0: '2B', 2: 'H/8', 3: 'W/8'}, 
            'dbar_11': {0: '2B', 2: 'H/8', 3: 'W/8'},
            'mid_block_additional_residual': {0: '2B', 2: 'H/8', 3: 'W/8'},
        }
        if self.controlnet:
            dynamic_axes.update(controlnet_dynamic_axes)
        return dynamic_axes
        

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        input_profile =  {
            'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)],
        }
        controlnet_input_profile = {'dbar_0': [(2*min_batch, 320, min_latent_height, min_latent_width), (2*batch_size, 320, latent_height, latent_width), (2*max_batch, 320, max_latent_height, max_latent_width)],
            'dbar_1': [(2*min_batch, 320, min_latent_height, min_latent_width), (2*batch_size, 320, latent_height, latent_width), (2*max_batch, 320, max_latent_height, max_latent_width)],
            'dbar_2': [(2*min_batch, 320, min_latent_height, min_latent_width), (2*batch_size, 320, latent_height, latent_width), (2*max_batch, 320, max_latent_height, max_latent_width)],
            'dbar_3': [(2*min_batch, 320, min_latent_height//2, min_latent_width//2), (2*batch_size, 320, latent_height//2, latent_width//2), (2*max_batch, 320, max_latent_height//2, max_latent_width//2)],
            'dbar_4': [(2*min_batch, 640, min_latent_height//2, min_latent_width//2), (2*batch_size, 640, latent_height//2, latent_width//2), (2*max_batch, 640, max_latent_height//2, max_latent_width//2)],
            'dbar_5': [(2*min_batch, 640, min_latent_height//2, min_latent_width//2), (2*batch_size, 640, latent_height//2, latent_width//2), (2*max_batch, 640, max_latent_height//2, max_latent_width//2)],
            'dbar_6': [(2*min_batch, 640, min_latent_height//4, min_latent_width//4), (2*batch_size, 640, latent_height//4, latent_width//4), (2*max_batch, 640, max_latent_height//4, max_latent_width//4)],
            'dbar_7': [(2*min_batch, 1280, min_latent_height//4, min_latent_width//4), (2*batch_size, 1280, latent_height//4, latent_width//4), (2*max_batch, 1280, max_latent_height//4, max_latent_width//4)],
            'dbar_8': [(2*min_batch, 1280, min_latent_height//4, min_latent_width//4), (2*batch_size, 1280, latent_height//4, latent_width//4), (2*max_batch, 1280, max_latent_height//4, max_latent_width//4)],
            'dbar_9': [(2*min_batch, 1280, min_latent_height//8, min_latent_width//8), (2*batch_size, 1280, latent_height//8, latent_width//8), (2*max_batch, 1280, max_latent_height//8, max_latent_width//8)],
            'dbar_10': [(2*min_batch, 1280, min_latent_height//8, min_latent_width//8), (2*batch_size, 1280, latent_height//8, latent_width//8), (2*max_batch, 1280, max_latent_height//8, max_latent_width//8)],
            'dbar_11': [(2*min_batch, 1280, min_latent_height//8, min_latent_width//8), (2*batch_size, 1280, latent_height//8, latent_width//8), (2*max_batch, 1280, max_latent_height//8, max_latent_width//8)],
            'mid_block_additional_residual':[(2*min_batch, 1280, min_latent_height//8, min_latent_width//8), (2*batch_size, 1280, latent_height//8, latent_width//8), (2*max_batch, 1280, max_latent_height//8, max_latent_width//8)],
        }
        if self.controlnet:
            input_profile.update(controlnet_input_profile)
        return input_profile

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        shape_dict = {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'latent': (2*batch_size, 4, latent_height, latent_width),
        }
        controlnet_shape_dict = {'dbar_0': (2*batch_size, 320, latent_height, latent_width),
            'dbar_1': (2*batch_size, 320, latent_height, latent_width),
            'dbar_2': (2*batch_size, 320, latent_height, latent_width),
            'dbar_3': (2*batch_size, 320, latent_height//2, latent_width//2),
            'dbar_4': (2*batch_size, 640, latent_height//2, latent_width//2),
            'dbar_5': (2*batch_size, 640, latent_height//2, latent_width//2),
            'dbar_6': (2*batch_size, 640, latent_height//4, latent_width//4),
            'dbar_7': (2*batch_size, 1280, latent_height//4, latent_width//4),
            'dbar_8': (2*batch_size, 1280, latent_height//4, latent_width//4),
            'dbar_9': (2*batch_size, 1280, latent_height//8, latent_width//8),
            'dbar_10': (2*batch_size, 1280, latent_height//8, latent_width//8),
            'dbar_11': (2*batch_size, 1280, latent_height//8, latent_width//8),
            'mid_block_additional_residual': (2*batch_size, 1280, latent_height//8, latent_width//8),
        }
        if self.controlnet:
            shape_dict.update(controlnet_shape_dict)
        return shape_dict

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        sample_input = [
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
        ]
        controlnet_input = [
            torch.randn(2*batch_size, 320, latent_height, latent_width, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 320, latent_height, latent_width, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 320, latent_height, latent_width, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 320, latent_height//2, latent_width//2, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 640, latent_height//2, latent_width//2, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 640, latent_height//2, latent_width//2, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 640, latent_height//4, latent_width//4, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 1280, latent_height//4, latent_width//4, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 1280, latent_height//4, latent_width//4, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float16, device=self.device),
            torch.randn(2*batch_size, 1280, latent_height//8, latent_width//8, dtype=torch.float16, device=self.device),
        ]
        if self.controlnet:
            sample_input = sample_input + controlnet_input
        return tuple(sample_input)
            
def make_UNet(version, hf_token, device, verbose, max_batch_size, inpaint=False, controlnet=False):
    unet = UNet(hf_token=hf_token, fp16=True, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version), unet_dim=(9 if inpaint else 4))
    unet.controlnet = controlnet
    return unet

class ControlNet(BaseModel):
    def __init__(self,
        hf_token,
        fp16=False,
        device='cuda',
        verbose=True,
        path="",
        max_batch_size=16,
        embedding_dim=768,
        text_maxlen=77,
        unet_dim=4
    ):
        super(ControlNet, self).__init__(hf_token, fp16=fp16, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim, text_maxlen=text_maxlen)
        self.unet_dim = unet_dim
        self.name = "UNet"

    def get_model(self):
        model_opts = {'torch_dtype': torch.float16} if self.fp16 else {}
        return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
            **model_opts).to(self.device)

    def get_input_names(self):
        return ['sample', 'timestep', 'encoder_hidden_states', 'controlnet_cond']

    def get_output_names(self):
       return [ 'down_block_res_samples_0',
                'down_block_res_samples_1',
                'down_block_res_samples_2',
                'down_block_res_samples_3',
                'down_block_res_samples_4',
                'down_block_res_samples_5',
                'down_block_res_samples_6',
                'down_block_res_samples_7',
                'down_block_res_samples_8',
                'down_block_res_samples_9',
                'down_block_res_samples_10',
                'down_block_res_samples_11',
                'mid_block_res_sample']

    def get_dynamic_axes(self):
        return {
            'sample': {0: '2B', 2: 'H', 3: 'W'},
            'encoder_hidden_states': {0: '2B'},
            'controlnet_cond': {0: '2B', 2: 'H*8', 3: 'W*8'},
            'down_block_res_samples_0': {0: '2B', 2: 'H', 3: 'W'},
            'down_block_res_samples_1': {0: '2B', 2: 'H', 3: 'W'},
            'down_block_res_samples_2': {0: '2B', 2: 'H', 3: 'W'},
            'down_block_res_samples_3': {0: '2B', 2: 'H/2', 3: 'W/2'}, 
            'down_block_res_samples_4': {0: '2B', 2: 'H/2', 3: 'W/2'}, 
            'down_block_res_samples_5': {0: '2B', 2: 'H/2', 3: 'W/2'}, 
            'down_block_res_samples_6': {0: '2B', 2: 'H/4', 3: 'W/4'}, 
            'down_block_res_samples_7': {0: '2B', 2: 'H/4', 3: 'W/4'}, 
            'down_block_res_samples_8': {0: '2B', 2: 'H/4', 3: 'W/4'}, 
            'down_block_res_samples_9': {0: '2B', 2: 'H/8', 3: 'W/8'}, 
            'down_block_res_samples_10': {0: '2B', 2: 'H/8', 3: 'W/8'}, 
            'down_block_res_samples_11': {0: '2B', 2: 'H/8', 3: 'W/8'},
            'mid_block_res_sample': {0: '2B', 2: 'H/8', 3: 'W/8'},
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'sample': [(2*min_batch, self.unet_dim, min_latent_height, min_latent_width), (2*batch_size, self.unet_dim, latent_height, latent_width), (2*max_batch, self.unet_dim, max_latent_height, max_latent_width)],
            'encoder_hidden_states': [(2*min_batch, self.text_maxlen, self.embedding_dim), (2*batch_size, self.text_maxlen, self.embedding_dim), (2*max_batch, self.text_maxlen, self.embedding_dim)],
            'controlnet_cond': [(2*min_batch, 3, min_latent_height*8, min_latent_width*8), (2*batch_size, 3, latent_height*8, latent_width*8), (2*max_batch, 3, max_latent_height*8, max_latent_width*8)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'sample': (2*batch_size, self.unet_dim, latent_height, latent_width),
            'encoder_hidden_states': (2*batch_size, self.text_maxlen, self.embedding_dim),
            'controlnet_cond': (2*batch_size, 3, latent_height*8, latent_width*8),
            'down_block_res_samples_0': (2*batch_size, 320, latent_height, latent_width),
            'down_block_res_samples_1': (2*batch_size, 320, latent_height, latent_width),
            'down_block_res_samples_2': (2*batch_size, 320, latent_height, latent_width),
            'down_block_res_samples_3': (2*batch_size, 320, latent_height//2, latent_width//2),
            'down_block_res_samples_4': (2*batch_size, 640, latent_height//2, latent_width//2),
            'down_block_res_samples_5': (2*batch_size, 640, latent_height//2, latent_width//2),
            'down_block_res_samples_6': (2*batch_size, 640, latent_height//4, latent_width//4),
            'down_block_res_samples_7': (2*batch_size, 1280, latent_height//4, latent_width//4),
            'down_block_res_samples_8': (2*batch_size, 1280, latent_height//4, latent_width//4),
            'down_block_res_samples_9': (2*batch_size, 1280, latent_height//8, latent_width//8),
            'down_block_res_samples_10': (2*batch_size, 1280, latent_height//8, latent_width//8),
            'down_block_res_samples_11': (2*batch_size, 1280, latent_height//8, latent_width//8),
            'mid_block_res_sample': (2*batch_size, 1280, latent_height//8, latent_width//8),
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        dtype = torch.float16 if self.fp16 else torch.float32
        return (
            torch.randn(2*batch_size, self.unet_dim, latent_height, latent_width, dtype=torch.float32, device=self.device),
            torch.tensor([1.], dtype=torch.float32, device=self.device),
            torch.randn(2*batch_size, self.text_maxlen, self.embedding_dim, dtype=dtype, device=self.device),
            torch.randn(2*batch_size, 3, latent_height*8, latent_width*8, dtype=torch.float16, device=self.device),
        )

def make_ControlNet(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return ControlNet(hf_token=hf_token, fp16=True, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version), unet_dim=(9 if inpaint else 4))

class VAE(BaseModel):
    def __init__(self,
        hf_token,
        device,
        verbose,
        path,
        max_batch_size,
        embedding_dim
    ):
        super(VAE, self).__init__(hf_token, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "VAE decoder"

    def get_model(self):
        vae = AutoencoderKL.from_pretrained(self.path,
            subfolder="vae",
            use_auth_token=self.hf_token).to(self.device)
        vae.forward = vae.decode
        return vae

    def get_input_names(self):
        return ['latent']

    def get_output_names(self):
       return ['images']

    def get_dynamic_axes(self):
        return {
            'latent': {0: 'B', 2: 'H', 3: 'W'},
            'images': {0: 'B', 2: '8H', 3: '8W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, _, _, _, _, min_latent_height, max_latent_height, min_latent_width, max_latent_width = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)
        return {
            'latent': [(min_batch, 4, min_latent_height, min_latent_width), (batch_size, 4, latent_height, latent_width), (max_batch, 4, max_latent_height, max_latent_width)]
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'latent': (batch_size, 4, latent_height, latent_width),
            'images': (batch_size, 3, image_height, image_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 4, latent_height, latent_width, dtype=torch.float32, device=self.device)

def make_VAE(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return VAE(hf_token=hf_token, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version))

class TorchVAEEncoder(torch.nn.Module):
    def __init__(self, token, device, path):
        super().__init__()
        self.path = path
        self.vae_encoder = AutoencoderKL.from_pretrained(self.path, subfolder="vae", use_auth_token=token).to(device)

    def forward(self, x):
        return self.vae_encoder.encode(x).latent_dist.sample()

class VAEEncoder(BaseModel):
    def __init__(self,
        hf_token,
        device,
        verbose,
        path,
        max_batch_size,
        embedding_dim
    ):
        super(VAEEncoder, self).__init__(hf_token, device=device, verbose=verbose, path=path, max_batch_size=max_batch_size, embedding_dim=embedding_dim)
        self.name = "VAE encoder"

    def get_model(self):
        vae_encoder = TorchVAEEncoder(self.hf_token, self.device, self.path)
        return vae_encoder

    def get_input_names(self):
        return ['images']

    def get_output_names(self):
       return ['latent']

    def get_dynamic_axes(self):
        return {
            'images': {0: 'B', 2: '8H', 3: '8W'},
            'latent': {0: 'B', 2: 'H', 3: 'W'}
        }

    def get_input_profile(self, batch_size, image_height, image_width, static_batch, static_shape):
        assert batch_size >= self.min_batch and batch_size <= self.max_batch
        min_batch = batch_size if static_batch else self.min_batch
        max_batch = batch_size if static_batch else self.max_batch
        self.check_dims(batch_size, image_height, image_width)
        min_batch, max_batch, min_image_height, max_image_height, min_image_width, max_image_width, _, _, _, _ = \
            self.get_minmax_dims(batch_size, image_height, image_width, static_batch, static_shape)

        return {
            'images': [(min_batch, 3, min_image_height, min_image_width), (batch_size, 3, image_height, image_width), (max_batch, 3, max_image_height, max_image_width)],
        }

    def get_shape_dict(self, batch_size, image_height, image_width):
        latent_height, latent_width = self.check_dims(batch_size, image_height, image_width)
        return {
            'images': (batch_size, 3, image_height, image_width),
            'latent': (batch_size, 4, latent_height, latent_width)
        }

    def get_sample_input(self, batch_size, image_height, image_width):
        self.check_dims(batch_size, image_height, image_width)
        return torch.randn(batch_size, 3, image_height, image_width, dtype=torch.float32, device=self.device)

def make_VAEEncoder(version, hf_token, device, verbose, max_batch_size, inpaint=False):
    return VAEEncoder(hf_token=hf_token, device=device, verbose=verbose, path=get_path(version, inpaint=inpaint),
            max_batch_size=max_batch_size, embedding_dim=get_embedding_dim(version))

def make_tokenizer(version, hf_token):
    return CLIPTokenizer.from_pretrained(get_path(version),
            subfolder="tokenizer",
            use_auth_token=hf_token)
