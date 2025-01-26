import os
import torch
import json
import gc
from .utils import log, print_memory
from diffusers.video_processor import VideoProcessor
from typing import List, Dict, Any, Tuple

from .hyvideo.constants import PROMPT_TEMPLATE
from .hyvideo.text_encoder import TextEncoder
from .hyvideo.utils.data_utils import align_to
from .hyvideo.diffusion.schedulers import FlowMatchDiscreteScheduler

from .hyvideo.diffusion.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from .hyvideo.diffusion.schedulers.scheduling_sasolver import SASolverScheduler
from. hyvideo.diffusion.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# from diffusers.schedulers import ( 
#     DDIMScheduler, 
#     PNDMScheduler, 
#     DPMSolverMultistepScheduler, 
#     EulerDiscreteScheduler, 
#     EulerAncestralDiscreteScheduler,
#     UniPCMultistepScheduler,
#     HeunDiscreteScheduler,
#     SASolverScheduler,
#     DEISMultistepScheduler,
#     LCMScheduler
#     )

scheduler_mapping = {
    "FlowMatchDiscreteScheduler": FlowMatchDiscreteScheduler,
    "SDE-DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "SASolverScheduler": SASolverScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
}

available_schedulers = list(scheduler_mapping.keys())
from .hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from .hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .hyvideo.modules.models import HYVideoDiffusionTransformer
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
folder_paths.add_model_folder_path("hyvid_embeds", os.path.join(folder_paths.get_output_directory(), "hyvid_embeds"))

import comfy.model_management as mm
from comfy.utils import load_torch_file, save_torch_file
import comfy.model_base
import comfy.latent_formats
import logging

#--- Logging Configuration ---
# Set to DEBUG to see detailed information, INFO for normal operation
log.setLevel(logging.DEBUG)

script_directory = os.path.dirname(os.path.abspath(__file__))

VAE_SCALING_FACTOR = 0.476986

def filter_state_dict_by_blocks(state_dict, blocks_mapping):
    filtered_dict = {}

    for key in state_dict:
        if 'double_blocks.' in key or 'single_blocks.' in key:
            block_pattern = key.split('diffusion_model.')[1].split('.', 2)[0:2]
            block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

            if block_key in blocks_mapping:
                filtered_dict[key] = state_dict[key]

    return filtered_dict

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
        new_sd[k] = v
    return new_sd

class HyVideoLoraBlockEdit:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("BOOLEAN", {"default": True})

        for i in range(20):
            arg_dict["double_blocks.{}.".format(i)] = argument

        for i in range(40):
            arg_dict["single_blocks.{}.".format(i)] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("SELECTEDBLOCKS", )
    RETURN_NAMES = ("blocks", )
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "select"

    CATEGORY = "HunyuanVideoWrapper"

    def select(self, **kwargs):
        selected_blocks = {k: v for k, v in kwargs.items() if v is True}
        print("Selected blocks: ", selected_blocks)
        return (selected_blocks,)
class HyVideoLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA models are expected to be in ComfyUI/models/loras with .safetensors extension"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LORA strength, set to 0.0 to unmerge the LORA"}),
            },
            "optional": {
                "prev_lora":("HYVIDLORA", {"default": None, "tooltip": "For loading multiple LoRAs"}),
                "blocks":("SELECTEDBLOCKS", ),
            }
        }

    RETURN_TYPES = ("HYVIDLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Select a LoRA model from ComfyUI/models/loras"

    def getlorapath(self, lora, strength, blocks=None, prev_lora=None, fuse_lora=False):
        loras_list = []

        lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "fuse_lora": fuse_lora,
            "blocks": blocks
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)

class HyVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "double_blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 20, "step": 1, "tooltip": "Number of double blocks to swap"}),
                "single_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1, "tooltip": "Number of single blocks to swap"}),
                "offload_txt_in": ("BOOLEAN", {"default": False, "tooltip": "Offload txt_in layer"}),
                "offload_img_in": ("BOOLEAN", {"default": False, "tooltip": "Offload img_in layer"}),
            },
        }
    RETURN_TYPES = ("BLOCKSWAPARGS",)
    RETURN_NAMES = ("block_swap_args",)
    FUNCTION = "setargs"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Settings for block swapping, reduces VRAM use by swapping blocks to CPU memory"

    def setargs(self, **kwargs):
        return (kwargs, )
    
class HyVideoEnhanceAVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.01, "tooltip": "The feta Weight of the Enhance-A-Video"}),
                "single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable Enhance-A-Video for single blocks"}),
                "double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable Enhance-A-Video for double blocks"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply Enhance-A-Video"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply Enhance-A-Video"}),
            },
        }
    RETURN_TYPES = ("FETAARGS",)
    RETURN_NAMES = ("feta_args",)
    FUNCTION = "setargs"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video"

    def setargs(self, **kwargs):
        return (kwargs, )

class HyVideoSTG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stg_mode": (["STG-A", "STG-R"],),
                "stg_block_idx": ("INT", {"default": 0, "min": -1, "max": 39, "step": 1, "tooltip": "Block index to apply STG"}),
                "stg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Recommended values are ≤2.0"}),
                "stg_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply STG"}),
                "stg_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply STG"}),
            },
        }
    RETURN_TYPES = ("STGARGS",)
    RETURN_NAMES = ("stg_args",)
    FUNCTION = "setargs"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Spatio Temporal Guidance, https://github.com/junhahyung/STGuidance"

    def setargs(self, **kwargs):
        return (kwargs, )

class HyVideoTeaCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Higher values will make TeaCache more aggressive, faster, but may cause artifacts"}),
            },
        }
    RETURN_TYPES = ("TEACACHEARGS",)
    RETURN_NAMES = ("teacache_args",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "TeaCache settings for HunyuanVideo to speed up inference"

    def process(self, rel_l1_thresh):
        teacache_args = {
            "rel_l1_thresh": rel_l1_thresh,
        }
        return (teacache_args,)


class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        # Don't know what this is. Value taken from ComfyUI Mochi model.
        self.memory_usage_factor = 2.0
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True


#region Model loading
class HyVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_scaled', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_varlen",
                    "sageattn_varlen",
                    "comfy",
                    ], {"default": "flash_attn"}),
                "compile_args": ("COMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("HYVIDLORA", {"default": None}),
                "auto_cpu_offload": ("BOOLEAN", {"default": False, "tooltip": "Enable auto offloading for reduced VRAM usage, implementation from DiffSynth-Studio, slightly different from block swapping and uses even less VRAM, but can be slower as you can't define how much VRAM to use"}),
                "upcast_rope": ("BOOLEAN", {"default": True, "tooltip": "Upcast RoPE to fp32 for better accuracy, this is the default behaviour, disabling can improve speed and reduce memory use slightly"}),
            }
        }

    RETURN_TYPES = ("HYVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, auto_cpu_offload=False, upcast_rope=True):
        transformer = None
        #mm.unload_all_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn_varlen
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[base_precision]

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        in_channels = out_channels = 16
        factor_kwargs = {"device": transformer_load_device, "dtype": base_dtype}
        HUNYUAN_VIDEO_CONFIG = {
            "mm_double_blocks_depth": 20,
            "mm_single_blocks_depth": 40,
            "rope_dim_list": [16, 56, 56],
            "hidden_size": 3072,
            "heads_num": 24,
            "mlp_width_ratio": 4,
            "guidance_embed": True,
        }
        with init_empty_weights():
            transformer = HYVideoDiffusionTransformer(
                in_channels=in_channels,
                out_channels=out_channels,
                attention_mode=attention_mode,
                main_device=device,
                offload_device=offload_device,
                **HUNYUAN_VIDEO_CONFIG,
                **factor_kwargs
            )
        transformer.eval()

        transformer.upcast_rope = upcast_rope

        comfy_model = HyVideoModel(
            HyVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )        
        
        scheduler_config = {
            "flow_shift": 9.0,
            "reverse": True,
            "solver": "euler",
            "use_flow_sigmas": True, 
            "prediction_type": 'flow_prediction'
        }
        scheduler = FlowMatchDiscreteScheduler.from_config(scheduler_config)
        
        pipe = HunyuanVideoPipeline(
            transformer=transformer,
            scheduler=scheduler,
            progress_bar_config=None,
            base_dtype=base_dtype,
            comfy_model=comfy_model,
        )

        if not "torchao" in quantization:
            log.info("Using accelerate to load and assign model weights to device...")
            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
                dtype = torch.float8_e4m3fn
            else:
                dtype = base_dtype
            params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
            for name, param in transformer.named_parameters():
                dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
            pipe.comfy_model = patcher

            del sd
            gc.collect()
            mm.soft_empty_cache()

            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    if l["blocks"]:
                        lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])

                    #for k in lora_sd.keys():
                     #   print(k)

                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

            comfy.model_management.load_models_gpu([patcher])
            if load_device == "offload_device":
                patcher.model.diffusion_model.to(offload_device)

            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(patcher.model.diffusion_model, base_dtype, params_to_keep=params_to_keep)
            elif quantization == "fp8_scaled":
                from .hyvideo.modules.fp8_optimization import convert_fp8_linear
                convert_fp8_linear(patcher.model.diffusion_model, base_dtype)

            if auto_cpu_offload:
                transformer.enable_auto_offload(dtype=dtype, device=device)

            #compile
            if compile_args is not None:
                torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
                if compile_args["compile_single_blocks"]:
                    for i, block in enumerate(patcher.model.diffusion_model.single_blocks):
                        patcher.model.diffusion_model.single_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_double_blocks"]:
                    for i, block in enumerate(patcher.model.diffusion_model.double_blocks):
                        patcher.model.diffusion_model.double_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_txt_in"]:
                    patcher.model.diffusion_model.txt_in = torch.compile(patcher.model.diffusion_model.txt_in, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_vector_in"]:
                    patcher.model.diffusion_model.vector_in = torch.compile(patcher.model.diffusion_model.vector_in, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                if compile_args["compile_final_layer"]:
                    patcher.model.diffusion_model.final_layer = torch.compile(patcher.model.diffusion_model.final_layer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
        elif "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                int4_weight_only
            )
            except:
                raise ImportError("torchao is not installed")

            # def filter_fn(module: nn.Module, fqn: str) -> bool:
            #     target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
            #     if any(sub in fqn for sub in target_submodules):
            #         return isinstance(module, nn.Linear)
            #     return False

            if "fp6" in quantization:
                quant_func = fpx_weight_only(3, 2)
            elif "int4" in quantization:
                quant_func = int4_weight_only()
            elif "int8" in quantization:
                quant_func = int8_weight_only()
            elif "fp8dq" in quantization:
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()

            log.info(f"Quantizing model with {quant_func}")
            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)

            comfy.model_management.load_models_gpu([patcher])

            for i, block in enumerate(patcher.model.diffusion_model.single_blocks):
                log.info(f"Quantizing single_block {i}")
                for name, _ in block.named_parameters(prefix=f"single_blocks.{i}"):
                    #print(f"Parameter name: {name}")
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=patcher.model.diffusion_model_load_device, dtype=base_dtype, value=sd[name])
                if compile_args is not None:
                    patcher.model.diffusion_model.single_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                quantize_(block, quant_func)
                print(block)
                block.to(offload_device)
            for i, block in enumerate(patcher.model.diffusion_model.double_blocks):
                log.info(f"Quantizing double_block {i}")
                for name, _ in block.named_parameters(prefix=f"double_blocks.{i}"):
                    #print(f"Parameter name: {name}")
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=patcher.model.diffusion_model_load_device, dtype=base_dtype, value=sd[name])
                if compile_args is not None:
                    patcher.model.diffusion_model.double_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                quantize_(block, quant_func)
            for name, param in patcher.model.diffusion_model.named_parameters():
                if "single_blocks" not in name and "double_blocks" not in name:
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=patcher.model.diffusion_model_load_device, dtype=base_dtype, value=sd[name])

            manual_offloading = False # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")
            for name, param in patcher.model.diffusion_model.named_parameters():
                print(name, param.dtype)
                #param.data = param.data.to(self.vae_dtype).to(device)

            del sd
            mm.soft_empty_cache()

        patcher.model["pipe"] = pipe
        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = "disabled"
        patcher.model["block_swap_args"] = block_swap_args
        patcher.model["auto_cpu_offload"] = auto_cpu_offload
        patcher.model["scheduler_config"] = scheduler_config

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)            

        return (patcher,)

#region load VAE

class HyVideoVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
                "compile_args":("COMPILEARGS", ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision, compile_args=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        with open(os.path.join(script_directory, 'configs', 'hy_vae_config.json')) as f:
            vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        vae = AutoencoderKLCausal3D.from_config(vae_config)
        vae.load_state_dict(vae_sd)
        del vae_sd
        vae.requires_grad_(False)
        vae.eval()
        vae.to(device = device, dtype = dtype)

        #compile
        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            vae = torch.compile(vae, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            

        return (vae,)



class HyVideoTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
                "compile_txt_in": ("BOOLEAN", {"default": False, "tooltip": "Compile txt_in layers"}),
                "compile_vector_in": ("BOOLEAN", {"default": False, "tooltip": "Compile vector_in layers"}),
                "compile_final_layer": ("BOOLEAN", {"default": False, "tooltip": "Compile final layer"}),

            },
        }
    RETURN_TYPES = ("COMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks, compile_txt_in, compile_vector_in, compile_final_layer):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks,
            "compile_txt_in": compile_txt_in,
            "compile_vector_in": compile_vector_in,
            "compile_final_layer": compile_final_layer
        }

        return (compile_args, )

#region TextEncode

class DownloadAndLoadHyVideoTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_model": (["Kijai/llava-llama-3-8b-text-encoder-tokenizer","xtuner/llava-llama-3-8b-v1_1-transformers"],),
                "clip_model": (["disabled","openai/clip-vit-large-patch14",],),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            },
            "optional": {
                "apply_final_norm": ("BOOLEAN", {"default": False}),
                "hidden_state_skip_layer": ("INT", {"default": 2}),
                "quantization": (['disabled', 'bnb_nf4', "fp8_e4m3fn"], {"default": 'disabled'}),
            },
            # Note: Changing max_context_length is experimental and may cause instability or higher memory use
            # max_context_length = 8192 is based on: https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers/blob/main/config.json
            "max_context_length": (
                "INT",
                {
                    "default": 256,  # HunyuanVideo default is 256
                    "min": 1,
                    "max": 8192,  # Max context length for the LLM
                    "step": 1,
                    "tooltip": "The maximum context length / max tokens for the LLM. Default is 256. Changing this is experimental and may cause instability in generation and performance, higher memory usage, or system crashes.",
                },
            ),
        }

    RETURN_TYPES = ("HYVIDTEXTENCODER",)
    RETURN_NAMES = ("hyvid_text_encoder", )
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Loads Hunyuan text_encoder model from 'ComfyUI/models/LLM'"

    def loadmodel(self, llm_model, clip_model, precision,  apply_final_norm=False, hidden_state_skip_layer=2, quantization="disabled", max_context_length=256):
        lm_type_mapping = {
            "Kijai/llava-llama-3-8b-text-encoder-tokenizer": "llm",
            "xtuner/llava-llama-3-8b-v1_1-transformers": "vlm",
        }
        lm_type = lm_type_mapping[llm_model]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        quantization_config = None
        if quantization == "bnb_nf4":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            )
            
        if clip_model != "disabled":
            clip_model_path = os.path.join(folder_paths.models_dir, "clip", "clip-vit-large-patch14")
            if not os.path.exists(clip_model_path):
                log.info(f"Downloading clip model to: {clip_model_path}")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=clip_model,
                    ignore_patterns=["*.msgpack", "*.bin", "*.h5"],
                    local_dir=clip_model_path,
                    local_dir_use_symlinks=False,
                )

            text_encoder_2 = TextEncoder(
            text_encoder_path=clip_model_path,
            text_encoder_type="clipL",
            max_length=77,
            text_encoder_precision=precision,
            tokenizer_type="clipL",
            logger=log,
            device=device,
        )
        else:
            text_encoder_2 = None

        download_path = os.path.join(folder_paths.models_dir,"LLM")
        base_path = os.path.join(download_path, (llm_model.split("/")[-1]))
        if not os.path.exists(base_path):
            log.info(f"Downloading model to: {base_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=llm_model,
                local_dir=base_path,
                local_dir_use_symlinks=False,
            )
            
        # Validation for max_context_length
        if max_context_length <= PROMPT_TEMPLATE["dit-llm-encode-video"]["crop_start"]:
            raise ValueError(
                f"max_context_length ({max_context_length}) must be greater than crop_start ({PROMPT_TEMPLATE['dit-llm-encode-video']['crop_start']}) for the 'video' template."
            )

        # DEBUG: Log the max_length used by the TextEncoder
        log.debug(
            f"DownloadAndLoadHyVideoTextEncoder: Loading text encoder with max_context_length: {max_context_length}"
        )
        
        text_encoder = TextEncoder(
            text_encoder_path=base_path,
            text_encoder_type=lm_type,
            max_length=max_context_length,  # max number of tokens / context for the LLM - default is 256
            text_encoder_precision=precision,
            tokenizer_type=lm_type,
            hidden_state_skip_layer=hidden_state_skip_layer,
            apply_final_norm=apply_final_norm,
            logger=log,
            device=device,
            dtype=dtype,
            quantization_config=quantization_config
        )
        
        # DEBUG: Log the max_length used by the TextEncoder
        log.debug(
            f"[[DEBUG]] DownloadAndLoadHyVideoTextEncoder: TextEncoder initialized with max_length: {text_encoder.max_length}"
        )
        
        if quantization == "fp8_e4m3fn":
            text_encoder.is_fp8 = True
            text_encoder.to(torch.float8_e4m3fn)
            def forward_hook(module):
                def forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                    return module.weight.to(input_dtype) * hidden_states.to(input_dtype)
                return forward

            for module in text_encoder.model.modules():
                if module.__class__.__name__ in ["Embedding"]:
                    module.to(dtype)
                if module.__class__.__name__ in ["LlamaRMSNorm"]:
                    module.forward = forward_hook(module)
        else:
            text_encoder.is_fp8 = False

        hyvid_text_encoders = {
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
        }

        return (hyvid_text_encoders,)

class HyVideoCustomPromptTemplate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "custom_prompt_template": ("STRING", {"default": f"{PROMPT_TEMPLATE['dit-llm-encode-video']['template']}", "multiline": True}),
            "crop_start": ("INT", {"default": PROMPT_TEMPLATE['dit-llm-encode-video']["crop_start"], "tooltip": "To cropt the system prompt"}),
            },
        }

    RETURN_TYPES = ("PROMPT_TEMPLATE", )
    RETURN_NAMES = ("hyvid_prompt_template",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, custom_prompt_template, crop_start):
        prompt_template_dict = {
            "template": custom_prompt_template,
            "crop_start": crop_start,
        }
        return (prompt_template_dict,)

class HyVideoTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text_encoders": ("HYVIDTEXTENCODER",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "prompt_template": (["video", "image", "custom", "disabled"], {"default": "video", "tooltip": "Use the default prompt templates for the llm text encoder"}),
                "custom_prompt_template": ("PROMPT_TEMPLATE", {"default": PROMPT_TEMPLATE["dit-llm-encode-video"], "multiline": True}),
                "clip_l": ("CLIP", {"tooltip": "Use comfy clip model instead, in this case the text encoder loader's clip_l should be disabled"}),
                "hyvid_cfg": ("HYVID_CFG", ),
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS", )
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, text_encoders, prompt, force_offload=True, prompt_template="video", custom_prompt_template=None, clip_l=None, image_token_selection_expr="::4", hyvid_cfg=None, image1=None, image2=None, clip_text_override=None):
        if clip_text_override is not None and len(clip_text_override) == 0:
            clip_text_override = None
        device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        text_encoder_1 = text_encoders["text_encoder"]
        if clip_l is None:
            text_encoder_2 = text_encoders["text_encoder_2"]
        else:
            text_encoder_2 = None

        if hyvid_cfg is not None:
            negative_prompt = hyvid_cfg["negative_prompt"]
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance = False
            negative_prompt = None

        if prompt_template != "disabled":
            if prompt_template == "custom":
                prompt_template_dict = custom_prompt_template
            elif prompt_template == "video":
                prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode-video"]
            elif prompt_template == "image":
                prompt_template_dict = PROMPT_TEMPLATE["dit-llm-encode"]
            else:
                raise ValueError(f"Invalid prompt_template: {prompt_template_dict}")
            assert (
                isinstance(prompt_template_dict, dict)
                and "template" in prompt_template_dict
            ), f"`prompt_template` must be a dictionary with a key 'template', got {prompt_template_dict}"
            assert "{}" in str(prompt_template_dict["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {prompt_template_dict['template']}"
            )
        else:
            prompt_template_dict = None

        def encode_prompt(self, prompt, negative_prompt, text_encoder, image_token_selection_expr="::4", image1=None, image2=None, clip_text_override=None):
            batch_size = 1
            num_videos_per_prompt = 1

            text_inputs = text_encoder.text2tokens(prompt, 
                                                   prompt_template=prompt_template_dict,
                                                   image1=image1,
                                                   image2=image2,
                                                   clip_text_override=clip_text_override)
            prompt_outputs = text_encoder.encode(text_inputs, 
                                                 prompt_template=prompt_template_dict, 
                                                 image_token_selection_expr=image_token_selection_expr, 
                                                 device=device
                                                 )
            prompt_embeds = prompt_outputs.hidden_state

            attention_mask = prompt_outputs.attention_mask
            log.info(f"{text_encoder.text_encoder_type} prompt attention_mask shape: {attention_mask.shape}, masked tokens: {attention_mask[0].sum().item()}")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

            prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                # max_length = prompt_embeds.shape[1]
                uncond_input = text_encoder.text2tokens(uncond_tokens, prompt_template=prompt_template_dict)

                negative_prompt_outputs = text_encoder.encode(
                    uncond_input, prompt_template=prompt_template_dict, device=device
                )
                negative_prompt_embeds = negative_prompt_outputs.hidden_state

                negative_attention_mask = negative_prompt_outputs.attention_mask
                if negative_attention_mask is not None:
                    negative_attention_mask = negative_attention_mask.to(device)
                    _, seq_len = negative_attention_mask.shape
                    negative_attention_mask = negative_attention_mask.repeat(
                        1, num_videos_per_prompt
                    )
                    negative_attention_mask = negative_attention_mask.view(
                        batch_size * num_videos_per_prompt, seq_len
                    )
            else:
                negative_prompt_embeds = None
                negative_attention_mask = None

            return (
                prompt_embeds,
                negative_prompt_embeds,
                attention_mask,
                negative_attention_mask,
            )
        text_encoder_1.to(device)
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=text_encoder_1.dtype, enabled=text_encoder_1.is_fp8):
            prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask = encode_prompt(self,
                                                                                                            prompt,
                                                                                                            negative_prompt, 
                                                                                                            text_encoder_1, 
                                                                                                            image_token_selection_expr=image_token_selection_expr,
                                                                                                            image1=image1,
                                                                                                            image2=image2)
        if force_offload:
            text_encoder_1.to(offload_device)
            mm.soft_empty_cache()

        if text_encoder_2 is not None:
            text_encoder_2.to(device)
            prompt_embeds_2, negative_prompt_embeds_2, attention_mask_2, negative_attention_mask_2 = encode_prompt(self, prompt, negative_prompt, text_encoder_2, clip_text_override=clip_text_override)
            if force_offload:
                text_encoder_2.to(offload_device)
                mm.soft_empty_cache()
        elif clip_l is not None:
            clip_l.cond_stage_model.to(device)
            tokens = clip_l.tokenize(prompt if clip_text_override is None else clip_text_override, return_word_ids=True)
            prompt_embeds_2 = clip_l.encode_from_tokens(tokens, return_pooled=True, return_dict=False)[1]
            prompt_embeds_2 = prompt_embeds_2.to(device=device)

            if negative_prompt is not None:
                tokens = clip_l.tokenize(negative_prompt, return_word_ids=True)
                negative_prompt_embeds_2 = clip_l.encode_from_tokens(tokens, return_pooled=True, return_dict=False)[1]
                negative_prompt_embeds_2 = negative_prompt_embeds_2.to(device=device)
            else:
                negative_prompt_embeds_2 = None
            attention_mask_2, negative_attention_mask_2 = None, None

            if force_offload:
                clip_l.cond_stage_model.to(offload_device)
                mm.soft_empty_cache()
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            attention_mask_2 = None
            negative_attention_mask_2 = None

        prompt_embeds_dict = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "attention_mask": attention_mask,
                "negative_attention_mask": negative_attention_mask,
                "prompt_embeds_2": prompt_embeds_2,
                "negative_prompt_embeds_2": negative_prompt_embeds_2,
                "attention_mask_2": attention_mask_2,
                "negative_attention_mask_2": negative_attention_mask_2,
                "cfg": torch.tensor(hyvid_cfg["cfg"]) if hyvid_cfg is not None else None,
                "start_percent": torch.tensor(hyvid_cfg["start_percent"]) if hyvid_cfg is not None else None,
                "end_percent": torch.tensor(hyvid_cfg["end_percent"]) if hyvid_cfg is not None else None,
            }
        return (prompt_embeds_dict,)

class HyVideoTextImageEncode(HyVideoTextEncode):
    # Experimental Image Prompt to Video (IP2V) via VLM implementation by @Dango233
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text_encoders": ("HYVIDTEXTENCODER",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            "image_token_selection_expr": ("STRING", {"default": "::4", "multiline": False} ),
            },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "prompt_template": (["video", "image", "custom", "disabled"], {"default": "video", "tooltip": "Use the default prompt templates for the llm text encoder"}),
                "custom_prompt_template": ("PROMPT_TEMPLATE", {"default": PROMPT_TEMPLATE["dit-llm-encode-video"], "multiline": True}),
                "clip_l": ("CLIP", {"tooltip": "Use comfy clip model instead, in this case the text encoder loader's clip_l should be disabled"}),
                "image1": ("IMAGE", {"default": None}),
                "image2": ("IMAGE", {"default": None}),
                "clip_text_override": ("STRING", {"default": "", "multiline": True} ),
                "hyvid_cfg": ("HYVID_CFG", ),
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS", )
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

# region CFG    
class HyVideoCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "negative_prompt": ("STRING", {"default": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion", "multiline": True} ),
            "cfg": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "guidance scale"} ),
            "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply CFG, rest of the steps use guidance_embeds"} ),
            "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "End percentage of the steps to apply CFG, rest of the steps use guidance_embeds"} ),
            },
        }

    RETURN_TYPES = ("HYVID_CFG", )
    RETURN_NAMES = ("hyvid_cfg",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "To use CFG with HunyuanVideo"

    def process(self, negative_prompt, cfg, start_percent, end_percent):
        cfg_dict = {
            "negative_prompt": negative_prompt,
            "cfg": cfg,
            "start_percent": start_percent,
            "end_percent": end_percent,
        }
        
        return (cfg_dict,)

#region embeds
class HyVideoTextEmbedsSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "hyvid_embeds": ("HYVIDEMBEDS",),
            "filename_prefix": ("STRING", {"default": "hyvid_embeds/hyvid_embed"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Save the text embeds"


    def save(self, hyvid_embeds, prompt, filename_prefix, extra_pnginfo=None):
        from comfy.cli_args import args
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}_.safetensors"
        file = os.path.join(full_output_folder, file)

        tensors_to_save = {}
        for key, value in hyvid_embeds.items():
            if value is not None:
                tensors_to_save[key] = value

        prompt_info = ""
        if prompt is not None:
            prompt_info = json.dumps(prompt)
        metadata = None
        if not args.disable_metadata:
            metadata = {"prompt": prompt_info}
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])
        
        save_torch_file(tensors_to_save, file, metadata=metadata)
        
        return (file,)

class HyVideoTextEmbedsLoad:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"embeds": (folder_paths.get_filename_list("hyvid_embeds"), {"tooltip": "The saved embeds to load from output/hyvid_embeds."})}}

    RETURN_TYPES = ("HYVIDEMBEDS", )
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "load"
    CATEGORY = "HunyuanVideoWrapper"
    DESCTIPTION = "Load the saved text embeds"


    def load(self, embeds):
        embed_path = folder_paths.get_full_path_or_raise("hyvid_embeds", embeds)
        loaded_tensors = load_torch_file(embed_path, safe_load=True)
        # Reconstruct original dictionary with None for missing keys
        prompt_embeds_dict = {
            "prompt_embeds": loaded_tensors.get("prompt_embeds", None),
            "negative_prompt_embeds": loaded_tensors.get("negative_prompt_embeds", None),
            "attention_mask": loaded_tensors.get("attention_mask", None),
            "negative_attention_mask": loaded_tensors.get("negative_attention_mask", None),
            "prompt_embeds_2": loaded_tensors.get("prompt_embeds_2", None),
            "negative_prompt_embeds_2": loaded_tensors.get("negative_prompt_embeds_2", None),
            "attention_mask_2": loaded_tensors.get("attention_mask_2", None),
            "negative_attention_mask_2": loaded_tensors.get("negative_attention_mask_2", None),
            "cfg": loaded_tensors.get("cfg", None),
            "start_percent": loaded_tensors.get("start_percent", None),
            "end_percent": loaded_tensors.get("end_percent", None),
        }
        
        return (prompt_embeds_dict,)
    
class HyVideoContextOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context_schedule": (["uniform_standard", "uniform_looped", "static_standard"],),
            "context_frames": ("INT", {"default": 65, "min": 2, "max": 1000, "step": 1, "tooltip": "Number of pixel frames in the context, NOTE: the latent space has 4 frames in 1"} ),
            "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context stride as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "context_overlap": ("INT", {"default": 4, "min": 4, "max": 100, "step": 1, "tooltip": "Context overlap as pixel frames, NOTE: the latent space has 4 frames in 1"} ),
            "freenoise": ("BOOLEAN", {"default": True, "tooltip": "Shuffle the noise"}),
            }
        }

    RETURN_TYPES = ("HYVIDCONTEXT", )
    RETURN_NAMES = ("context_options",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Context options for HunyuanVideo, allows splitting the video into context windows and attemps blending them for longer generations than the model and memory otherwise would allow."

    def process(self, context_schedule, context_frames, context_stride, context_overlap, freenoise):
        context_options = {
            "context_schedule":context_schedule,
            "context_frames":context_frames,
            "context_stride":context_stride,
            "context_overlap":context_overlap,
            "freenoise":freenoise
        }

        return (context_options,)
#region Sampler
class HyVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "num_frames": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 4}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "embedded_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),

            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stg_args": ("STGARGS", ),
                "context_options": ("HYVIDCONTEXT", ),
                "feta_args": ("FETAARGS", ),
                "teacache_args": ("TEACACHEARGS", ),
                "scheduler": (available_schedulers,
                    {
                        "default": 'FlowMatchDiscreteScheduler'
                    }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, model, hyvid_embeds, flow_shift, steps, embedded_guidance_scale, seed, width, height, num_frames, 
                samples=None, denoise_strength=1.0, force_offload=True, stg_args=None, context_options=None, feta_args=None, teacache_args=None, scheduler=None):
        model = model.model

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model["dtype"]
        transformer = model["pipe"].transformer

        #handle STG
        if stg_args is not None:
            if stg_args["stg_mode"] == "STG-A" and transformer.attention_mode != "sdpa":
                raise ValueError(
                    f"STG-A requires attention_mode to be 'sdpa', but got {transformer.attention_mode}."
            )
        #handle CFG
        if hyvid_embeds.get("cfg") is not None:
            cfg = float(hyvid_embeds.get("cfg", 1.0))
            cfg_start_percent = float(hyvid_embeds.get("start_percent", 0.0))
            cfg_end_percent = float(hyvid_embeds.get("end_percent", 1.0))
        else:
            cfg = 1.0
            cfg_start_percent = 0.0
            cfg_end_percent = 1.0
        
        if embedded_guidance_scale == 0.0:
            embedded_guidance_scale = None

        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        if width <= 0 or height <= 0 or num_frames <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={num_frames}"
            )
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"`video_length - 1 (that's minus one frame)` must be a multiple of 4, got {num_frames}"
            )

        log.info(
            f"Input (height, width, video_length) = ({height}, {width}, {num_frames})"
        )

        target_height = align_to(height, 16)
        target_width = align_to(width, 16)

        scheduler_config = model["scheduler_config"]

        scheduler_config["flow_shift"] = flow_shift
        if scheduler == "SDE-DPMSolverMultistepScheduler":
            scheduler_config["algorithm_type"] = "sde-dpmsolver++"
        elif scheduler == "SASolverScheduler":
            scheduler_config["algorithm_type"] = "data_prediction"
        else:
            scheduler_config.pop("algorithm_type", None)
        #model["scheduler_config"]["use_beta_flow_sigmas"] = True
        
        noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
        model["pipe"].scheduler = noise_scheduler

        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                if "single" not in name and "double" not in name:
                    param.data = param.data.to(device)

            transformer.block_swap(
                model["block_swap_args"]["double_blocks_to_swap"] - 1 ,
                model["block_swap_args"]["single_blocks_to_swap"] - 1,
                offload_txt_in = model["block_swap_args"]["offload_txt_in"],
                offload_img_in = model["block_swap_args"]["offload_img_in"],
            )
        elif model["auto_cpu_offload"]:
            for name, param in transformer.named_parameters():
                if "single" not in name and "double" not in name:
                    param.data = param.data.to(device)
        elif model["manual_offloading"]:
            transformer.to(device)

        # Initialize TeaCache if enabled
        if teacache_args is not None:
            # Check if dimensions have changed since last run
            if (not hasattr(transformer, 'last_dimensions') or
                    transformer.last_dimensions != (height, width, num_frames) or
                    not hasattr(transformer, 'last_frame_count') or
                    transformer.last_frame_count != num_frames):
                # Reset TeaCache state on dimension change
                transformer.cnt = 0
                transformer.accumulated_rel_l1_distance = 0
                transformer.previous_modulated_input = None
                transformer.previous_residual = None
                transformer.last_dimensions = (height, width, num_frames)
                transformer.last_frame_count = num_frames

            transformer.enable_teacache = True
            transformer.num_steps = steps
            transformer.rel_l1_thresh = teacache_args["rel_l1_thresh"]
        else:
            transformer.enable_teacache = False

        mm.soft_empty_cache()
        gc.collect()

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        #for name, param in transformer.named_parameters():
        #    print(name, param.data.device)

        leapfusion_img2vid = False
        if samples is not None:
            input_latents = samples["samples"] * VAE_SCALING_FACTOR
            if input_latents.shape[2] == 1:
                leapfusion_img2vid = True
        else:
            input_latents = None            

        out_latents = model["pipe"](
            num_inference_steps=steps,
            height = target_height,
            width = target_width,
            video_length = num_frames,
            guidance_scale=cfg,
            cfg_start_percent=cfg_start_percent,
            cfg_end_percent=cfg_end_percent,
            embedded_guidance_scale=embedded_guidance_scale,
            latents=input_latents,
            denoise_strength=denoise_strength,
            prompt_embed_dict=hyvid_embeds,
            generator=generator,
            stg_mode=stg_args["stg_mode"] if stg_args is not None else None,
            stg_block_idx=stg_args["stg_block_idx"] if stg_args is not None else -1,
            stg_scale=stg_args["stg_scale"] if stg_args is not None else 0.0,
            stg_start_percent=stg_args["stg_start_percent"] if stg_args is not None else 0.0,
            stg_end_percent=stg_args["stg_end_percent"] if stg_args is not None else 1.0,
            context_options=context_options,
            feta_args=feta_args,
            leapfusion_img2vid = leapfusion_img2vid
        )

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        return ({
            "samples": out_latents.cpu() / VAE_SCALING_FACTOR
            },)

#region VideoDecode
class HyVideoDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "samples": ("LATENT",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64, any other value will cause stutter"}),
                    "spatial_tile_sample_min_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32, "tooltip": "Spatial tile minimum size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Automatically set tile size based on defaults, above settings are ignored"}),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "HunyuanVideoWrapper"

    def decode(self, vae, samples, enable_vae_tiling, temporal_tiling_sample_size, spatial_tile_sample_min_size, auto_tile_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        latents = samples["samples"]
        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        if not auto_tile_size:
            vae.tile_latent_min_tsize = temporal_tiling_sample_size // 4
            vae.tile_sample_min_size = spatial_tile_sample_min_size
            vae.tile_latent_min_size = spatial_tile_sample_min_size // 8
            if temporal_tiling_sample_size != 64:
                vae.t_tile_overlap_factor = 0.0
            else:
                vae.t_tile_overlap_factor = 0.25
        else:
            #defaults
            vae.tile_latent_min_tsize = 16
            vae.tile_sample_min_size = 256
            vae.tile_latent_min_size = 32


        expand_temporal_dim = False
        if len(latents.shape) == 4:
            if isinstance(vae, AutoencoderKLCausal3D):
                latents = latents.unsqueeze(2)
                expand_temporal_dim = True
        elif len(latents.shape) == 5:
            pass
        else:
            raise ValueError(
                f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
            )
        #latents = latents / vae.config.scaling_factor
        latents = latents.to(vae.dtype).to(device)

        if enable_vae_tiling:
            vae.enable_tiling()
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]
        else:
            video = vae.decode(
                latents, return_dict=False, generator=generator
            )[0]

        if expand_temporal_dim or video.shape[2] == 1:
            video = video.squeeze(2)

        vae.to(offload_device)
        mm.soft_empty_cache()

        if len(video.shape) == 5:
            video_processor = VideoProcessor(vae_scale_factor=8)
            video_processor.config.do_resize = False

            video = video_processor.postprocess_video(video=video, output_type="pt")
            out = video[0].permute(0, 2, 3, 1).cpu().float()
        else:
            out = (video / 2 + 0.5).clamp(0, 1)
            out = out.permute(0, 2, 3, 1).cpu().float()

        return (out,)

#region VideoEncode
class HyVideoEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("VAE",),
                    "image": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "temporal_tiling_sample_size": ("INT", {"default": 64, "min": 4, "max": 256, "tooltip": "Smaller values use less VRAM, model default is 64, any other value will cause stutter"}),
                    "spatial_tile_sample_min_size": ("INT", {"default": 256, "min": 32, "max": 2048, "step": 32, "tooltip": "Spatial tile minimum size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "auto_tile_size": ("BOOLEAN", {"default": True, "tooltip": "Automatically set tile size based on defaults, above settings are ignored"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanVideoWrapper"

    def encode(self, vae, image, enable_vae_tiling, temporal_tiling_sample_size, auto_tile_size, spatial_tile_sample_min_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        generator = torch.Generator(device=torch.device("cpu"))#.manual_seed(seed)
        vae.to(device)
        if not auto_tile_size:
            vae.tile_latent_min_tsize = temporal_tiling_sample_size // 4
            vae.tile_sample_min_size = spatial_tile_sample_min_size
            vae.tile_latent_min_size = spatial_tile_sample_min_size // 8
            if temporal_tiling_sample_size != 64:
                vae.t_tile_overlap_factor = 0.0
            else:
                vae.t_tile_overlap_factor = 0.25
        else:
            #defaults
            vae.tile_latent_min_tsize = 16
            vae.tile_sample_min_size = 256
            vae.tile_latent_min_size = 32

        image = (image * 2.0 - 1.0).to(vae.dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        if enable_vae_tiling:
            vae.enable_tiling()
        latents = vae.encode(image).latent_dist.sample(generator)
        #latents = latents * vae.config.scaling_factor
        vae.to(offload_device)
        print("encoded latents shape",latents.shape)


        return ({"samples": latents},)

class HyVideoLatentPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "min_val": ("FLOAT", {"default": -0.15, "min": -1.0, "max": 0.0, "step": 0.001}),
                 "max_val": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "r_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                 "g_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
                 "b_bias": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("images", "latent_rgb_factors",)
    FUNCTION = "sample"
    CATEGORY = "HunyuanVideoWrapper"

    def sample(self, samples, seed, min_val, max_val, r_bias, g_bias, b_bias):
        mm.soft_empty_cache()

        latents = samples["samples"].clone()
        print("in sample", latents.shape)
        #latent_rgb_factors =[[-0.02531045419704009, -0.00504800612542497, 0.13293717293982546], [-0.03421835830845858, 0.13996708548892614, -0.07081038680118075], [0.011091819063647063, -0.03372949685846012, -0.0698232210116172], [-0.06276524604742019, -0.09322986677909442, 0.01826383612148913], [0.021290659938126788, -0.07719530444034409, -0.08247812477766273], [0.04401102991215147, -0.0026401932105894754, -0.01410913586718443], [0.08979717602613707, 0.05361221258740831, 0.11501425309699129], [0.04695121980405198, -0.13053491609675175, 0.05025986885867986], [-0.09704684176098193, 0.03397687417738002, -0.1105886644677771], [0.14694697234804935, -0.12316902186157716, 0.04210404546699645], [0.14432470831243552, -0.002580008133591355, -0.08490676947390643], [0.051502750076553944, -0.10071695490292451, -0.01786223610178095], [-0.12503276881774464, 0.08877830923879379, 0.1076584501927316], [-0.020191205513213406, -0.1493425056303128, -0.14289740371758308], [-0.06470138952271293, -0.07410426095060325, 0.00980804676890873], [0.11747671720735695, 0.10916082743849789, -0.12235599365235904]]
        latent_rgb_factors = [[-0.41, -0.25, -0.26],
                              [-0.26, -0.49, -0.24],
                              [-0.37, -0.54, -0.3],
                              [-0.04, -0.29, -0.29],
                              [-0.52, -0.59, -0.39],
                              [-0.56, -0.6, -0.02],
                              [-0.53, -0.06, -0.48],
                              [-0.51, -0.28, -0.18],
                              [-0.59, -0.1, -0.33],
                              [-0.56, -0.54, -0.41],
                              [-0.61, -0.19, -0.5],
                              [-0.05, -0.25, -0.17],
                              [-0.23, -0.04, -0.22],
                              [-0.51, -0.56, -0.43],
                              [-0.13, -0.4, -0.05],
                              [-0.01, -0.01, -0.48]]

        import random
        random.seed(seed)
        #latent_rgb_factors = [[random.uniform(min_val, max_val) for _ in range(3)] for _ in range(16)]
        out_factors = latent_rgb_factors
        print(latent_rgb_factors)

        #latent_rgb_factors_bias = [0.138, 0.025, -0.299]
        latent_rgb_factors_bias = [r_bias, g_bias, b_bias]

        latent_rgb_factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        print("latent_rgb_factors", latent_rgb_factors.shape)

        latent_images = []
        for t in range(latents.shape[2]):
            latent = latents[:, :, t, :, :]
            latent = latent[0].permute(1, 2, 0)
            latent_image = torch.nn.functional.linear(
                latent,
                latent_rgb_factors,
                bias=latent_rgb_factors_bias
            )
            latent_images.append(latent_image)
        latent_images = torch.stack(latent_images, dim=0)
        print("latent_images", latent_images.shape)
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)

        return (latent_images.float().cpu(), out_factors)

NODE_CLASS_MAPPINGS = {
    "HyVideoSampler": HyVideoSampler,
    "HyVideoDecode": HyVideoDecode,
    "HyVideoTextEncode": HyVideoTextEncode,
    "HyVideoTextImageEncode": HyVideoTextImageEncode,
    "HyVideoModelLoader": HyVideoModelLoader,
    "HyVideoVAELoader": HyVideoVAELoader,
    "DownloadAndLoadHyVideoTextEncoder": DownloadAndLoadHyVideoTextEncoder,
    "HyVideoEncode": HyVideoEncode,
    "HyVideoBlockSwap": HyVideoBlockSwap,
    "HyVideoTorchCompileSettings": HyVideoTorchCompileSettings,
    "HyVideoSTG": HyVideoSTG,
    "HyVideoCFG": HyVideoCFG,
    "HyVideoCustomPromptTemplate": HyVideoCustomPromptTemplate,
    "HyVideoLatentPreview": HyVideoLatentPreview,
    "HyVideoLoraSelect": HyVideoLoraSelect,
    "HyVideoLoraBlockEdit": HyVideoLoraBlockEdit,
    "HyVideoTextEmbedsSave": HyVideoTextEmbedsSave,
    "HyVideoTextEmbedsLoad": HyVideoTextEmbedsLoad,
    "HyVideoContextOptions": HyVideoContextOptions,
    "HyVideoEnhanceAVideo": HyVideoEnhanceAVideo,
    "HyVideoTeaCache": HyVideoTeaCache,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideoSampler": "HunyuanVideo Sampler",
    "HyVideoDecode": "HunyuanVideo Decode",
    "HyVideoTextEncode": "HunyuanVideo TextEncode",
    "HyVideoTextImageEncode": "HunyuanVideo TextImageEncode (IP2V)",
    "HyVideoModelLoader": "HunyuanVideo Model Loader",
    "HyVideoVAELoader": "HunyuanVideo VAE Loader",
    "DownloadAndLoadHyVideoTextEncoder": "(Down)Load HunyuanVideo TextEncoder",
    "HyVideoEncode": "HunyuanVideo Encode",
    "HyVideoBlockSwap": "HunyuanVideo BlockSwap",
    "HyVideoTorchCompileSettings": "HunyuanVideo Torch Compile Settings",
    "HyVideoSTG": "HunyuanVideo STG",
    "HyVideoCFG": "HunyuanVideo CFG",
    "HyVideoCustomPromptTemplate": "HunyuanVideo Custom Prompt Template",
    "HyVideoLatentPreview": "HunyuanVideo Latent Preview",
    "HyVideoLoraSelect": "HunyuanVideo Lora Select",
    "HyVideoLoraBlockEdit": "HunyuanVideo Lora Block Edit",
    "HyVideoTextEmbedsSave": "HunyuanVideo TextEmbeds Save",
    "HyVideoTextEmbedsLoad": "HunyuanVideo TextEmbeds Load",
    "HyVideoContextOptions": "HunyuanVideo Context Options",
    "HyVideoEnhanceAVideo": "HunyuanVideo Enhance A Video",
    "HyVideoTeaCache": "HunyuanVideo TeaCache",
    }
