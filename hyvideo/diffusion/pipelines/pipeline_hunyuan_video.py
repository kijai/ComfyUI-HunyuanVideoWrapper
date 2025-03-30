# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

from ...modules import HYVideoDiffusionTransformer
from comfy.utils import ProgressBar

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""
from ...modules.posemb_layers import get_nd_rotary_pos_embed
from ....enhance_a_video.globals import enable_enhance, disable_enhance, set_enhance_weight

def get_rotary_pos_embed(transformer, latent_video_length, height, width, k=0):
        target_ndim = 3
        ndim = 5 - 2
        rope_theta = 225
        patch_size = transformer.patch_size
        rope_dim_list = transformer.rope_dim_list
        hidden_size = transformer.hidden_size
        heads_num = transformer.heads_num
        head_dim = hidden_size // heads_num

        # 884
        latents_size = [latent_video_length, height // 8, width // 8]

        if isinstance(patch_size, int):
            assert all(s % patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // patch_size for s in latents_size]
        elif isinstance(patch_size, list):
            assert all(
                s % patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=rope_theta,
            use_real=True,
            theta_rescale_factor=1,
            num_frames=latent_video_length,
            k=k,
        )
        return freqs_cos, freqs_sin
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(
        self,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        comfy_model = None,
        progress_bar_config: Dict[str, Any] = None,
        base_dtype = torch.bfloat16,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.base_dtype = base_dtype
        self.comfy_model = comfy_model
        # ==========================================================================================

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler
        )
        self.vae_scale_factor = 8

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps.to(device), num_inference_steps - t_start


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        num_inference_steps,
        height,
        width,
        video_length,
        device,
        timesteps,
        generator,
        latents=None,
        denoise_strength=1.0,
        freenoise=False, 
        context_size=None, 
        context_overlap=None,
        i2v_condition_type=None,
        image_cond_latents=None,
        i2v_stability=True,
        
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is not None:
            latents = latents.to(device)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=self.base_dtype)
    
        if freenoise:
            logger.info("Applying FreeNoise")
            # code and comments from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
            #video_length = video_length // 4
            delta = context_size - context_overlap
            for start_idx in range(0, video_length-context_size, delta):
                # start_idx corresponds to the beginning of a context window
                # goal: place shuffled in the delta region right after the end of the context window
                #       if space after context window is not enough to place the noise, adjust and finish
                place_idx = start_idx + context_size
                # if place_idx is outside the valid indexes, we are already finished
                if place_idx >= video_length:
                    break
                end_idx = place_idx - 1
                #print("video_length:", video_length, "start_idx:", start_idx, "end_idx:", end_idx, "place_idx:", place_idx, "delta:", delta)

                # if there is not enough room to copy delta amount of indexes, copy limited amount and finish
                if end_idx + delta >= video_length:
                    final_delta = video_length - place_idx
                    # generate list of indexes in final delta region
                    list_idx = torch.tensor(list(range(start_idx,start_idx+final_delta)), device=torch.device("cpu"), dtype=torch.long)
                    # shuffle list
                    list_idx = list_idx[torch.randperm(final_delta, generator=generator)]
                    # apply shuffled indexes
                    noise[:, :, place_idx:place_idx + final_delta, :, :] = noise[:, :, list_idx, :, :]
                    break
                # otherwise, do normal behavior
                # generate list of indexes in delta region
                list_idx = torch.tensor(list(range(start_idx,start_idx+delta)), device=torch.device("cpu"), dtype=torch.long)
                # shuffle list
                list_idx = list_idx[torch.randperm(delta, generator=generator)]
                # apply shuffled indexes
                #print("place_idx:", place_idx, "delta:", delta, "list_idx:", list_idx)
                noise[:, :, place_idx:place_idx + delta, :, :] = noise[:, :, list_idx, :, :]
        
        i2v_mask = None
        if image_cond_latents is not None:
            if i2v_condition_type == "latent_concat":
                # Create mask
                i2v_mask = torch.zeros(shape[0], 1, shape[2], shape[3], shape[4], device=device)
                i2v_mask[:, :, 0, ...] = 1.0
                if image_cond_latents.shape[2] == 1:      
                    padding = torch.zeros(shape, device=device)
                    padding[:, :, 0:1, :, :] = image_cond_latents
                    image_cond_latents = padding

        if denoise_strength < 1.0:
            if i2v_condition_type == "latent_concat":
                latents = torch.cat((latents[:,:,0].unsqueeze(2), latents), dim=2)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, denoise_strength, device)
            latent_timestep = timesteps[:1]
            frames_needed = noise.shape[2]
            current_frames = latents.shape[2]
            
            if frames_needed > current_frames:
                repeat_factor = frames_needed - current_frames
                additional_frame = torch.randn((latents.shape[0], latents.shape[1], repeat_factor, latents.shape[3], latents.shape[4]), dtype=latents.dtype, device=latents.device)
                latents = torch.cat((additional_frame, latents), dim=2)
                logger.info(f"Frames needed more than current frames, adding {repeat_factor} frames")
            elif frames_needed < current_frames:
                latents = latents[:, :, :frames_needed, :, :]
                logger.info(f"Frames needed less than current frames, cutting down to {frames_needed}")
    
            latents = latents * (1 - latent_timestep / 1000) + latent_timestep / 1000 * noise
            print("latents shape:", latents.shape)

        elif image_cond_latents is not None and i2v_stability:
            if image_cond_latents.shape[2] == 1:
                img_latents = image_cond_latents.repeat(1, 1, video_length, 1, 1)
            else:
                img_latents = image_cond_latents
            t = torch.tensor([0.999]).to(device=device)
            latents = noise * t + img_latents * (1 - t)
            latents = latents.to(dtype=self.base_dtype)
        else:
            logger.info("Using random noise only")
            latents = noise

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents.to(device), timesteps, i2v_mask, image_cond_latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def do_spatio_temporal_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._stg_scale > 0

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        height: int,
        width: int,
        video_length: int,
        prompt_embed_dict: dict,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 1.0,
        cfg_start_percent: float = 0.0,
        cfg_end_percent: float = 1.0,
        batched_cfg: bool = True,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        denoise_strength: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        embedded_guidance_scale: Optional[float] = None,
        stg_mode: Optional[str] = None,
        stg_block_idx: Optional[int] = -1,
        stg_scale: Optional[float] = 0.0,
        stg_start_percent: Optional[float] = 0.0,
        stg_end_percent: Optional[float] = 1.0,
        context_options: Optional[Dict[str, Any]] = None,
        feta_args: Optional[Dict] = None,
        leapfusion_img2vid: Optional[bool] = False,
        image_cond_latents: Optional[torch.Tensor] = None,
        riflex_freq_index: Optional[int] = None,
        i2v_stability=True,
        taylorseer: Optional[dict] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
                
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        self._stg_scale = stg_scale
        # 2. Define call parameters
       
        batch_size = 1
        device = self._execution_device

        prompt_embeds = prompt_embed_dict.get("prompt_embeds", None)
        negative_prompt_embeds = prompt_embed_dict.get("negative_prompt_embeds", None)
        #prompt_mask = prompt_embed_dict.get("attention_mask", None)
        #negative_prompt_mask = prompt_embed_dict.get("negative_attention_mask", None)
        prompt_embeds_2 = prompt_embed_dict.get("prompt_embeds_2", None)
        negative_prompt_embeds_2 = prompt_embed_dict.get("negative_prompt_embeds_2", None)

        # Handle primary embeds
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            max_length = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
            if prompt_embeds.shape[1] < max_length:
                prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, max_length - prompt_embeds.shape[1]))
            if negative_prompt_embeds.shape[1] < max_length:
                negative_prompt_embeds = torch.nn.functional.pad(negative_prompt_embeds, (0, 0, 0, max_length - negative_prompt_embeds.shape[1]))

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance and not self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            # if prompt_mask is not None:
            #     prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
        elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds, prompt_embeds]
            )
            # if prompt_mask is not None:
            #     prompt_mask = torch.cat([negative_prompt_mask, prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat(
                    [negative_prompt_embeds_2, prompt_embeds_2, prompt_embeds_2]
                )
        elif self.do_spatio_temporal_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])
            # if prompt_mask is not None:
            #     prompt_mask = torch.cat([prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([prompt_embeds_2, prompt_embeds_2])

        prompt_embeds = prompt_embeds.to(device = device, dtype = self.base_dtype)
        #prompt_mask = prompt_mask.to(device)
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(device = device, dtype = self.base_dtype)

        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {}
        )
        if hasattr(self.scheduler, "set_begin_index") and denoise_strength == 1.0:
            self.scheduler.set_begin_index(begin_index=0)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        
        latent_video_length = (video_length - 1) // 4 + 1
        original_image_latents = image_cond_latents
        i2v_condition_type = self.transformer.i2v_condition_type
        #if i2v_condition_type == "latent_concat":
                #latent_video_length += 1
        if feta_args is not None:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            enable_enhance(feta_args["single_blocks"], feta_args["double_blocks"])
        else:
            disable_enhance()

        #  context windows
        use_context_schedule = False
        freenoise = False
        context_stride = 1
        context_overlap = 1
        context_frames = 65
        if context_options is not None:
            context_schedule = context_options["context_schedule"]
            context_frames =  (context_options["context_frames"] - 1) // 4 + 1
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
            freenoise = context_options["freenoise"]
             
            logger.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            use_context_schedule = True
            from ....context import get_context_scheduler
            context = get_context_scheduler(context_schedule)
            freqs_cos, freqs_sin = get_rotary_pos_embed(
                self.transformer, context_frames, height, width
            )
        else:
            # rotary embeddings
            freqs_cos, freqs_sin = get_rotary_pos_embed(
                self.transformer, latent_video_length, height, width, k=riflex_freq_index
            )
        if not self.transformer.upcast_rope:
            freqs_cos = freqs_cos.to(self.base_dtype).to(device)
            freqs_sin = freqs_sin.to(self.base_dtype).to(device)
        else:
            freqs_cos = freqs_cos.to(device)
            freqs_sin = freqs_sin.to(device)
        
        if leapfusion_img2vid:
            logger.info("Single input latent frame detected, LeapFusion img2vid enabled")
            original_latents = latents
        # 5. Prepare latent variables
        #num_channels_latents = self.transformer.config.in_channels
        num_channels_latents = 16
        latents, timesteps, i2v_mask, image_cond_latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_inference_steps,
            height,
            width,
            latent_video_length,
            device,
            timesteps,
            generator,
            latents,
            denoise_strength=denoise_strength,
            freenoise=freenoise,
            context_size=context_frames,
            context_overlap=context_overlap,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability,
            image_cond_latents=image_cond_latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # 8. Preview callback
        from latent_preview import prepare_callback
        callback = prepare_callback(self.comfy_model, num_inference_steps)

        #print(self.scheduler.sigmas)

        tseercache_dict, tseer_current = None, None
        if taylorseer:
            print(taylorseer)
            from ...modules.cache_functions import cache_init
            tseercache_dict, tseer_current = cache_init(self._num_timesteps, cache_device=taylorseer["cache_device"], compute_device=taylorseer["compute_device"])
            tseercache_dict["max_order"] = taylorseer["max_order"]
            tseercache_dict["fresh_threshold"] = taylorseer["fresh_threshold"]
        
        logger.info(f"Sampling {video_length} frames in {latents.shape[2]} latents at {width}x{height} with {len(timesteps)} inference steps")

        comfy_pbar = ProgressBar(len(timesteps))
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if image_cond_latents is not None and i2v_condition_type == "token_replace":
                    latents = torch.concat([original_image_latents, latents[:, :, 1:, :, :]], dim=2)

                latent_model_input = latents
                input_prompt_embeds = prompt_embeds
                #input_prompt_mask = prompt_mask 
                input_prompt_embeds_2 = prompt_embeds_2
                cfg_enabled = False
                stg_enabled = False

                current_step_percentage = i / len(timesteps)

                if taylorseer:
                    tseer_current['step'] = i

                if self.do_spatio_temporal_guidance:
                    if stg_start_percent <= current_step_percentage <= stg_end_percent:
                        stg_enabled = True
                        if self.do_classifier_free_guidance:
                            latent_model_input = torch.cat([latents] * 3)
                        else:
                            latent_model_input = torch.cat([latents] * 2)
                    else:
                        stg_mode = None
                        stg_block_idx = -1
                        input_prompt_embeds = prompt_embeds[0].unsqueeze(0)
                        #input_prompt_mask = prompt_mask[0].unsqueeze(0)
                        input_prompt_embeds_2 = prompt_embeds_2[0].unsqueeze(0)
                        latent_model_input = latents
                else:
                    stg_enabled = False
                    # expand the latents if we are doing classifier free guidance
                    
                    if self.do_classifier_free_guidance:
                        if cfg_start_percent <= current_step_percentage <= cfg_end_percent:
                            #print("applying CFG at step", i + 1, "with strength", guidance_scale)
                            latent_model_input = torch.cat([latents] * 2)
                            cfg_enabled = True
                        else:
                            input_prompt_embeds = prompt_embeds[1].unsqueeze(0)
                            #input_prompt_mask = prompt_mask[1].unsqueeze(0)
                            input_prompt_embeds_2 = prompt_embeds_2[1].unsqueeze(0)
                
                if feta_args is not None:
                    if feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enable_enhance(feta_args["single_blocks"], feta_args["double_blocks"])
                    else:
                        disable_enhance()

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])

                if leapfusion_img2vid:
                    latent_model_input[:, :, [0], :, :] = original_latents[:, :, [0], :, :].to(latent_model_input)

                if image_cond_latents is not None and not use_context_schedule:
                    if i2v_condition_type == "latent_concat":
                        latent_image_input = (torch.cat([image_cond_latents] * 2) if cfg_enabled else image_cond_latents)
                        if self.transformer.in_channels == 33:
                            i2v_mask = torch.cat([i2v_mask] * 2) if cfg_enabled else i2v_mask
                            latent_image_input = torch.cat([latent_image_input, i2v_mask], dim=1)
                        latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=1)

                if self.transformer.guidance_embed:
                    if cfg_enabled:
                        guidance_expand = (
                            torch.tensor([embedded_guidance_scale] * latents.shape[0] * 2, dtype=self.base_dtype, device=device)
                            * 1000.0
                        )
                    else:
                        guidance_expand = (
                            torch.tensor([embedded_guidance_scale] * latents.shape[0], dtype=self.base_dtype, device=device)
                            * 1000.0
                        )
                else:
                    guidance_expand = None

                if use_context_schedule:
                    counter = torch.zeros_like(latent_model_input)[:, :16]
                    noise_pred = torch.zeros_like(latent_model_input)[:, :16]
                    print("noise_pred", noise_pred.shape)
                    print("counter", counter.shape)
                    context_queue = list(context(
                            i, num_inference_steps, latents.shape[2], context_frames, context_stride, context_overlap,
                        ))
                    
                    if image_cond_latents is not None:
                        latent_image_input = (
                            torch.cat([image_cond_latents] * 2) if cfg_enabled else image_cond_latents
                        )
                        if i2v_mask is not None:
                            i2v_mask = torch.cat([i2v_mask] * 2) if cfg_enabled else i2v_mask

                    for c in context_queue:
                        partial_latent_model_input = latent_model_input[:, :, c]
                        if i2v_mask is not None:
                            #doesn't work properly
                            new_mask = torch.zeros_like(i2v_mask)
                            new_mask[..., 0, :, :] = 1.0
                            new_image_input = torch.zeros_like(latent_image_input)
                            new_image_input[..., 0, :, :] = latent_image_input[..., 0, :, :]

                            partial_latent_image_input = torch.cat([new_image_input[..., c, :, :], new_mask[..., c, :, :]], dim=1)
                            partial_latent_model_input = torch.cat([partial_latent_model_input, partial_latent_image_input], dim=1)
                        
                        #print("partial_latent_model_input", partial_latent_model_input.shape)
                        with torch.autocast(
                        device_type="cuda", dtype=self.base_dtype, enabled=True):
                            noise_pred_context = self.transformer(
                                partial_latent_model_input, 
                                t_expand,
                                text_states=input_prompt_embeds,
                                #text_mask=input_prompt_mask,
                                text_states_2=input_prompt_embeds_2,
                                freqs_cos=freqs_cos,
                                freqs_sin=freqs_sin,
                                guidance=guidance_expand,
                                stg_block_idx=stg_block_idx,
                                stg_mode=stg_mode,
                                return_dict=True,
                            )["x"]
                            window_mask = torch.ones_like(noise_pred_context)


                            # Apply left-side blending for all except first chunk
                            if min(c) > 0: 
                                ramp_up = torch.linspace(0, 1, context_overlap, device=noise_pred_context.device)
                                ramp_up = ramp_up.view(1, 1, -1, 1, 1)
                                window_mask[:, :, :context_overlap] = ramp_up
                            # Apply right-side blending for all except last chunk
                            if max(c) < latent_video_length - 1:
                                ramp_down = torch.linspace(1, 0, context_overlap, device=noise_pred_context.device)
                                ramp_down = ramp_down.view(1, 1, -1, 1, 1)
                                window_mask[:, :, -context_overlap:] = ramp_down
                            noise_pred[:, :, c, :, :] += noise_pred_context * window_mask
                            counter[:, :, c, :, :] += window_mask
                            noise_pred = noise_pred.float()
                    noise_pred /= counter
                else:
                    # predict the noise residual
                    with torch.autocast(
                        device_type="cuda", dtype=self.base_dtype, enabled=True
                    ):
                        if batched_cfg or not cfg_enabled:
                            noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                                latent_model_input,  # [2, 16, 33, 24, 42]
                                t_expand,  # [2]
                                text_states=input_prompt_embeds,  # [2, 256, 4096]
                                #text_mask=input_prompt_mask,  # [2, 256]
                                text_states_2=input_prompt_embeds_2,  # [2, 768]
                                freqs_cos=freqs_cos,  # [seqlen, head_dim]
                                freqs_sin=freqs_sin,  # [seqlen, head_dim]
                                guidance=guidance_expand,
                                stg_block_idx=stg_block_idx,
                                stg_mode=stg_mode,
                                return_dict=True,
                                tseercache_dict = tseercache_dict, #taylorseer
                                tseer_current = tseer_current, #taylorseer
                            )["x"]
                        else:
                            uncond = self.transformer(
                                latent_model_input[0].unsqueeze(0),
                                t_expand[0].unsqueeze(0),
                                text_states=input_prompt_embeds[0].unsqueeze(0), 
                                #text_mask=input_prompt_mask[0].unsqueeze(0), 
                                text_states_2=input_prompt_embeds_2[0].unsqueeze(0), 
                                freqs_cos=freqs_cos,
                                freqs_sin=freqs_sin,
                                guidance=guidance_expand[0].unsqueeze(0),
                                stg_block_idx=stg_block_idx,
                                stg_mode=stg_mode,
                                return_dict=True,
                                tseercache_dict = tseercache_dict, #taylorseer
                                tseer_current = tseer_current, #taylorseer
                            )["x"]
                            cond = self.transformer(
                                latent_model_input[1].unsqueeze(0),
                                t_expand[1].unsqueeze(0),
                                text_states=input_prompt_embeds[1].unsqueeze(0), 
                                #text_mask=input_prompt_mask[1].unsqueeze(0), 
                                text_states_2=input_prompt_embeds_2[1].unsqueeze(0), 
                                freqs_cos=freqs_cos,
                                freqs_sin=freqs_sin,
                                guidance=guidance_expand[1].unsqueeze(0),
                                stg_block_idx=stg_block_idx,
                                stg_mode=stg_mode,
                                return_dict=True,
                                tseercache_dict = tseercache_dict, #taylorseer
                                tseer_current = tseer_current, #taylorseer
                            )["x"]

                        # perform guidance
                        if cfg_enabled and not self.do_spatio_temporal_guidance:
                            if batched_cfg:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + self.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            else:
                                noise_pred = uncond + self.guidance_scale * (cond - uncond)
                        
            
                        elif self.do_classifier_free_guidance and self.do_spatio_temporal_guidance:
                            raise NotImplementedError
                            noise_pred_uncond, noise_pred_text, noise_pred_perturb = noise_pred.chunk(3)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            ) + self._stg_scale * (
                                noise_pred_text - noise_pred_perturb
                            )
                        elif self.do_spatio_temporal_guidance and stg_enabled:
                            noise_pred_text, noise_pred_perturb = noise_pred.chunk(2)
                            noise_pred = noise_pred_text + self._stg_scale * (
                                noise_pred_text - noise_pred_perturb
                            )

                # compute the previous noisy sample x_t -> x_t-1
                if image_cond_latents is not None and i2v_condition_type == "token_replace":
                    latents = self.scheduler.step(
                        noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                    )[0]
                    latents = torch.concat(
                        [original_image_latents, latents], dim=2
                    )
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None:
                        if leapfusion_img2vid or i2v_condition_type == "token_replace":
                            callback_latent = (latent_model_input[:, :, 1:, :, :] - noise_pred[:, :, 1:, :, :] * t / 1000).detach()[0].permute(1,0,2,3)
                        else:
                            callback_latent = (latent_model_input[:, :16, :, :, :] - noise_pred * t / 1000).detach()[0].permute(1,0,2,3)
                        callback(
                            i, 
                            callback_latent,
                            None, 
                            num_inference_steps
                        )
                    else:
                        comfy_pbar.update(1)

        if image_cond_latents is not None:
            if leapfusion_img2vid or i2v_condition_type == "latent_concat":
                latents = latents[:, :, 1:, :, :]
        return latents