#based on https://github.com/DarkMnDragon/rf-inversion-diffuser/blob/main/inversion_editing_cli.py
import torch
import gc
import os
from .utils import log, print_memory

from diffusers.utils.torch_utils import randn_tensor
import comfy.model_management as mm
from hyvideo.diffusion.pipelines.pipeline_hunyuan_video import get_rotary_pos_embed
from hyvideo.enhance_a_video.globals import enable_enhance, disable_enhance, set_enhance_weight

script_directory = os.path.dirname(os.path.abspath(__file__))

VAE_SCALING_FACTOR = 0.476986

def generate_eta_values(
    timesteps, 
    start_step, 
    end_step, 
    eta, 
    eta_trend,
):
    assert start_step < end_step and start_step >= 0 and end_step <= len(timesteps), "Invalid start_step and end_step"
    # timesteps are monotonically decreasing, from 1.0 to 0.0
    
    eta_values = [0.0] * (len(timesteps) - 1)
    
    if eta_trend == 'constant':
        for i in range(start_step, end_step):
            eta_values[i] = eta
    elif eta_trend == 'linear_increase':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[start_step] - timesteps[i]) / total_time
    elif eta_trend == 'linear_decrease':
        total_time = timesteps[start_step] - timesteps[end_step - 1]
        for i in range(start_step, end_step):
            eta_values[i] = eta * (timesteps[i] - timesteps[end_step - 1]) / total_time
    else:
        raise NotImplementedError(f"Unsupported eta_trend: {eta_trend}")
    print("eta_values", eta_values)
    return eta_values

class HyVideoEmptyTextEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            }
        }

    RETURN_TYPES = ("HYVIDEMBEDS", )
    RETURN_NAMES = ("hyvid_embeds",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "Empty Text Embeds for HunyuanVideoWrapper, to avoid having to encode prompts for inverse sampling"

    def process(self):
        device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()
        
        prompt_embeds_dict = torch.load(os.path.join(script_directory, "hunyuan_empty_prompt_embeds_dict.pt"))
        return (prompt_embeds_dict,)

#region Inverse Sampling
class HyVideoInverseSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "embedded_guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_step": ("INT", {"default": 0, "min": 0}),
                "end_step": ("INT", {"default": 18, "min": 0}),
                "gamma_trend": (['constant', 'linear_increase', 'linear_decrease'], {"default": "constant"}),
            },
            "optional": {
                "interpolation_curve": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "forceInput": True, "tooltip": "The strength of the inversed latents along time, in latent space"}),
            }    
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, model, hyvid_embeds, flow_shift, steps, embedded_guidance_scale, seed, samples, gamma, start_step, end_step, gamma_trend, force_offload, interpolation_curve=None):
        comfy_model_patcher = model
        model = model.model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model["dtype"]
        transformer = model["pipe"].transformer
        pipeline = model["pipe"]
        
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        latents = samples["samples"] * VAE_SCALING_FACTOR if samples is not None else None
        batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width = latents.shape
        height = latent_height * pipeline.vae_scale_factor
        width = latent_width * pipeline.vae_scale_factor
        num_frames = (latent_num_frames - 1) * 4 + 1


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

        freqs_cos, freqs_sin = get_rotary_pos_embed(transformer, latent_num_frames, height, width)
        freqs_cos = freqs_cos.to(device)
        freqs_sin = freqs_sin.to(device)

        pipeline.scheduler.flow_shift = flow_shift
  
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                #print(name, param.data.device)
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

        mm.soft_empty_cache()
        gc.collect()
        
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        pipeline.scheduler.set_timesteps(steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        timesteps = timesteps.flip(0)
        print("timesteps", timesteps)
        print("pipeline.scheduler.order", pipeline.scheduler.order)
        print("len(timesteps)", len(timesteps))

        latent_video_length = (num_frames - 1) // 4 + 1

        # 5. Prepare latent variables
        num_channels_latents = transformer.config.in_channels
       
        
        latents = latents.to(device)

        shape = (
            1,
            num_channels_latents,
            latent_video_length,
            int(height) // pipeline.vae_scale_factor,
            int(width) // pipeline.vae_scale_factor,
        )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        
        frames_needed = noise.shape[1]
        current_frames = latents.shape[1]
        
        if frames_needed > current_frames:
            repeat_factor = frames_needed - current_frames
            additional_frame = torch.randn((latents.size(0), repeat_factor, latents.size(2), latents.size(3), latents.size(4)), dtype=latents.dtype, device=latents.device)
            latents = torch.cat((additional_frame, latents), dim=1)
            self.additional_frames = repeat_factor
        elif frames_needed < current_frames:
            latents = latents[:, :frames_needed, :, :, :]
            
        gamma_values = generate_eta_values(timesteps / 1000, start_step, end_step, gamma, gamma_trend)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - steps * pipeline.scheduler.order
        self._num_timesteps = len(timesteps)

        latents = latents.to(dtype)

        from latent_preview import prepare_callback
        callback = prepare_callback(comfy_model_patcher, steps)

        from comfy.utils import ProgressBar
        from tqdm import tqdm
        log.info(f"Sampling {num_frames} frames in {latents.shape[2]} latents at {width}x{height} with {len(timesteps)} inference steps")
        comfy_pbar = ProgressBar(len(timesteps))
        with tqdm(total=len(timesteps)) as progress_bar:
            for idx, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                latent_model_input = latents

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(pipeline.base_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=pipeline.base_dtype, enabled=True
                ):
                    noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=hyvid_embeds["prompt_embeds"],  # [2, 256, 4096]
                        text_mask=hyvid_embeds["attention_mask"],  # [2, 256]
                        text_states_2=hyvid_embeds["prompt_embeds_2"],  # [2, 768]
                        freqs_cos=freqs_cos,  # [seqlen, head_dim]
                        freqs_sin=freqs_sin,  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        stg_block_idx=-1,
                        stg_mode=None,
                        return_dict=True,
                    )["x"]
                sigma = t / 1000.0
                sigma_prev = t_prev / 1000.0
                latents = latents.to(torch.float32)
                noise_pred = noise_pred.to(torch.float32)
                target_noise_velocity = (noise - latents) / (1.0 - sigma)

                if interpolation_curve is not None:
                    time_weights = torch.tensor(interpolation_curve, device=latents.device)
                    assert time_weights.shape[0] == latents.shape[2], f"Weight list length {len(interpolation_curve)} must match temporal dimension {latents.shape[2]}"
                    gamma = gamma_values[idx] * time_weights.view(1, 1, -1, 1, 1)  # shape [1, 1, 33, 1, 1]
                else:
                    gamma = gamma_values[idx]
                interpolated_velocity = gamma * target_noise_velocity + (1 - gamma) * noise_pred

                latents = latents + (sigma_prev - sigma) * interpolated_velocity
                latents = latents.to(torch.bfloat16)

                # compute the previous noisy sample x_t -> x_t-1
                #latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                
                progress_bar.update()
                if callback is not None:
                    callback(idx, (latent_model_input - noise_pred * t / 1000).detach()[0].permute(1,0,2,3), None, steps)
                else:
                    comfy_pbar.update(1)
                  

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
            "samples": latents / VAE_SCALING_FACTOR
            },)

#region ReSampler
class HyVideoReSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "inversed_latents": ("LATENT", {"tooltip": "inversed latents from HyVideoInverseSampler"} ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "embedded_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "tooltip": "The step to start the effect of the inversed latents"}),
                "end_step": ("INT", {"default": 18, "min": 0, "tooltip": "The step to end the effect of the inversed latents"}),
                "eta_base": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The base value of the eta, overall strength of the effect from the inversed latents"}),
                "eta_trend": (['constant', 'linear_increase', 'linear_decrease'], {"default": "constant", "tooltip": "The trend of the eta value over steps"}),
            },
            "optional": {
                "interpolation_curve": ("FLOAT", {"forceInput": True, "tooltip": "The strength of the inversed latents along time, in latent space"}),
                "feta_args": ("FETAARGS", ),

            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"

    def process(self, model, hyvid_embeds, flow_shift, steps, embedded_guidance_scale, 
                samples, inversed_latents, force_offload, start_step, end_step, eta_base, eta_trend, interpolation_curve=None, feta_args=None):
        comfy_model_patcher = model
        model = model.model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model["dtype"]
        transformer = model["pipe"].transformer
        pipeline = model["pipe"]
        
        target_latents = samples["samples"] * VAE_SCALING_FACTOR

        batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width = target_latents.shape
        height = latent_height * pipeline.vae_scale_factor
        width = latent_width * pipeline.vae_scale_factor
        num_frames = (latent_num_frames - 1) * 4 + 1

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

        freqs_cos, freqs_sin = get_rotary_pos_embed(transformer, latent_num_frames, height, width)
        freqs_cos = freqs_cos.to(device)
        freqs_sin = freqs_sin.to(device)

        pipeline.scheduler.flow_shift = flow_shift
  
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                #print(name, param.data.device)
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

        mm.soft_empty_cache()
        gc.collect()
        
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        pipeline.scheduler.set_timesteps(steps, device=device)
        timesteps = pipeline.scheduler.timesteps

        eta_values = generate_eta_values(timesteps / 1000, start_step, end_step, eta_base, eta_trend)
           
        target_latents = target_latents.to(device=device, dtype=dtype)
        latents = inversed_latents["samples"] * VAE_SCALING_FACTOR
        latents = latents.to(device=device, dtype=dtype)

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)

        from latent_preview import prepare_callback
        callback = prepare_callback(comfy_model_patcher, steps)

        if feta_args is not None:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            enable_enhance(feta_args["single_blocks"], feta_args["double_blocks"])
        else:
            disable_enhance()

        from comfy.utils import ProgressBar
        from tqdm import tqdm
        log.info(f"Sampling {num_frames} frames in {latents.shape[2]} latents at {width}x{height} with {len(timesteps)} inference steps")
        comfy_pbar = ProgressBar(len(timesteps))

        with tqdm(total=len(timesteps)) as progress_bar:
             for idx, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):

                current_step_percentage = idx / len(timesteps)

                if feta_args is not None:
                    if feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enable_enhance(feta_args["single_blocks"], feta_args["double_blocks"])
                    else:
                        disable_enhance()

                latent_model_input = latents

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(pipeline.base_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=pipeline.base_dtype, enabled=True
                ):
                    noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=hyvid_embeds["prompt_embeds"],  # [2, 256, 4096]
                        text_mask=hyvid_embeds["attention_mask"],  # [2, 256]
                        text_states_2=hyvid_embeds["prompt_embeds_2"],  # [2, 768]
                        freqs_cos=freqs_cos,  # [seqlen, head_dim]
                        freqs_sin=freqs_sin,  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        stg_block_idx=-1,
                        stg_mode=None,
                        return_dict=True,
                    )["x"]
                    sigma = t / 1000.0
                    sigma_prev = t_prev / 1000.0
                    noise_pred = noise_pred.to(torch.float32)
                    latents = latents.to(torch.float32)
                    target_latents = target_latents.to(torch.float32)
                    target_img_velocity = -(target_latents - latents) / sigma

                    # interpolated velocity
                    # Add time-varying weights
                    if interpolation_curve is not None:
                        time_weights = torch.tensor(interpolation_curve, device=latents.device)
                        assert time_weights.shape[0] == latents.shape[2], f"Weight list length {len(interpolation_curve)} must match temporal dimension {latents.shape[2]}"
                        eta = eta_values[idx] * time_weights.view(1, 1, -1, 1, 1)  # shape [1, 1, 33, 1, 1]
                    else:
                        eta = eta_values[idx]

                    # Time-varying interpolation
                    interpolated_velocity = eta * target_img_velocity + (1 - eta) * noise_pred
                    latents = latents + (sigma_prev - sigma) * interpolated_velocity
                  
                    #print(f"X_{sigma_prev:.3f} = X_{sigma:.3f} + {sigma_prev - sigma:.3f} * ({eta:.3f} * target_img_velocity + {1 - eta:.3f} * noise_pred)")
                    latents = latents.to(torch.bfloat16)

                    progress_bar.update()
                    if callback is not None:
                        callback(idx, (latent_model_input - noise_pred * t / 1000).detach()[0].permute(1,0,2,3), None, steps)
                    else:
                        comfy_pbar.update(1)

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
            "samples": latents / VAE_SCALING_FACTOR
            },)

#region PromptMix
class HyVideoPromptMixSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "hyvid_embeds": ("HYVIDEMBEDS", ),
                "hyvid_embeds_2": ("HYVIDEMBEDS", ),
                "width": ("INT", {"default": 512, "min": 1}),
                "height": ("INT", {"default": 512, "min": 1}),
                "num_frames": ("INT", {"default": 17, "min": 1}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "embedded_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 9.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "alpha": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Adjusts the blending sharpness"}),
            },
            "optional": {
                "interpolation_curve": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "forceInput": True, "tooltip": "The strength of the inversed latents along time, in latent space"}),
                "feta_args": ("FETAARGS", ),
            }                
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "HunyuanVideoWrapper"
    EXPERIMENTAL = True

    def process(self, model, width, height, num_frames, hyvid_embeds, hyvid_embeds_2, flow_shift, steps, embedded_guidance_scale, 
                seed, force_offload, alpha, interpolation_curve=None, feta_args=None):
        comfy_model_patcher = model
        model = model.model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model["dtype"]
        transformer = model["pipe"].transformer
        pipeline = model["pipe"]
        

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
        latent_video_length = (num_frames - 1) // 4 + 1
        freqs_cos, freqs_sin = get_rotary_pos_embed(transformer, latent_video_length, height, width)
        freqs_cos = freqs_cos.to(device)
        freqs_sin = freqs_sin.to(device)

        pipeline.scheduler.flow_shift = flow_shift
  
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                #print(name, param.data.device)
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

        mm.soft_empty_cache()
        gc.collect()
        
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        pipeline.scheduler.set_timesteps(steps, device=device)
        timesteps = pipeline.scheduler.timesteps

     
        #latents = samples["samples"]
        shape = (
            1,
            16,
            latent_video_length,
            int(height) // pipeline.vae_scale_factor,
            int(width) // pipeline.vae_scale_factor,
        )
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)


        llm_embeds_1 = hyvid_embeds["prompt_embeds"].to(dtype).to(device)
        clip_embeds_1 = hyvid_embeds["prompt_embeds_2"].to(dtype).to(device)
        mask_1 = hyvid_embeds["attention_mask"].to(device)
        llm_embeds_2 = hyvid_embeds_2["prompt_embeds"].to(dtype).to(device)
        clip_embeds_2 = hyvid_embeds_2["prompt_embeds_2"].to(dtype).to(device)
        mask_2 = hyvid_embeds_2["attention_mask"].to(device)
        text_embeds = torch.cat((llm_embeds_1, llm_embeds_2), dim=0)
        text_mask = torch.cat((mask_1, mask_2), dim=0)
        clip_embeds = torch.cat((clip_embeds_1, clip_embeds_2), dim=0)
        assert len(interpolation_curve) == latents.shape[2], f"Weight list length {len(interpolation_curve)} must match temporal dimension {latents.shape[2]}"
        latents_1 = latents.clone()
        latents_2 = latents.clone()

        if feta_args is not None:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            enable_enhance(feta_args["single_blocks"], feta_args["double_blocks"])
        else:
            disable_enhance()

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)

        from latent_preview import prepare_callback
        callback = prepare_callback(comfy_model_patcher, steps)

        from comfy.utils import ProgressBar
        from tqdm import tqdm
        log.info(f"Sampling {num_frames} frames in {latents.shape[2]} latents at {width}x{height} with {len(timesteps)} inference steps")
        comfy_pbar = ProgressBar(len(timesteps))

        with tqdm(total=len(timesteps)) as progress_bar:
            for idx, t in enumerate(timesteps):
                current_step_percentage = idx / len(timesteps)

                if feta_args is not None:
                    if feta_start_percent <= current_step_percentage <= feta_end_percent:
                        enable_enhance(feta_args["single_blocks"], feta_args["double_blocks"])
                    else:
                        disable_enhance()

                # Pre-compute weighted latents
                weighted_latents_1 = torch.zeros_like(latents_1)
                weighted_latents_2 = torch.zeros_like(latents_2)
                
                for t_idx in range(latents_1.shape[2]):
                    weight = interpolation_curve[t_idx]
                    weighted_latents_1[..., t_idx, :, :] = (
                        (1 - alpha * weight) * latents_1[..., t_idx, :, :] + 
                        (alpha * weight) * latents_2[..., t_idx, :, :]
                    )
                    weighted_latents_2[..., t_idx, :, :] = (
                        (1 - alpha * (1-weight)) * latents_2[..., t_idx, :, :] + 
                        (alpha * (1-weight)) * latents_1[..., t_idx, :, :]
                    )

                # Use weighted inputs for model
                latent_model_input = torch.cat([weighted_latents_1, weighted_latents_2])

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(pipeline.base_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )
                
                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=pipeline.base_dtype, enabled=True
                ):
                    noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=text_embeds,  # [2, 256, 4096]
                        text_mask=text_mask,  # [2, 256]
                        text_states_2=clip_embeds,  # [2, 768]
                        freqs_cos=freqs_cos,  # [seqlen, head_dim]
                        freqs_sin=freqs_sin,  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        stg_block_idx=-1,
                        stg_mode=None,
                        return_dict=True,
                    )["x"]

                 
                    noise_pred = noise_pred.to(torch.float32)
                    # 1. Get noise predictions for both prompts
                    noise_pred_prompt_1, noise_pred_prompt_2 = noise_pred.chunk(2)

                    # 2. Update latents separately for each prompt

                    dt = pipeline.scheduler.sigmas[idx + 1] - pipeline.scheduler.sigmas[idx]

                    latents_1 = latents_1 + noise_pred_prompt_1 * dt
                    latents_2 = latents_2 + noise_pred_prompt_2 * dt


                    # 3. Interpolate latents based on temporal curve
                    interpolated_latents = torch.zeros_like(latents_1)
                    for t_idx in range(latents.shape[2]):
                        weight = interpolation_curve[t_idx]
                        interpolated_latents[..., t_idx, :, :] = (
                            (1 - weight) * latents_1[..., t_idx, :, :] + 
                            weight * latents_2[..., t_idx, :, :]
                        )

                    latents = interpolated_latents

                    progress_bar.update()
                    if callback is not None:
                        callback(idx, (latent_model_input - noise_pred * t / 1000).detach()[0].permute(1,0,2,3), None, steps)
                    else:
                        comfy_pbar.update(1)

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
            "samples": latents / VAE_SCALING_FACTOR
            },)

NODE_CLASS_MAPPINGS = {
    "HyVideoInverseSampler": HyVideoInverseSampler,
    "HyVideoReSampler": HyVideoReSampler,
    "HyVideoEmptyTextEmbeds": HyVideoEmptyTextEmbeds,
    "HyVideoPromptMixSampler": HyVideoPromptMixSampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyVideoInverseSampler": "HunyuanVideo Inverse Sampler",
    "HyVideoReSampler": "HunyuanVideo ReSampler",
    "HyVideoEmptyTextEmbeds": "HunyuanVideo Empty Text Embeds",
    "HyVideoPromptMixSampler": "HunyuanVideo Prompt Mix Sampler"
}