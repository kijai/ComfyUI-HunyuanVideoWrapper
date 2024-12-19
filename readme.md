# ComfyUI wrapper nodes for [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)

# Multi GPU Setup - For systems with more then 1 cuda device.
The max I was able to get on a 4080 and 4090 distributed setup with no unloading..

![image](https://github.com/user-attachments/assets/0752b3d2-7559-40c8-8345-314582602496)

![image](https://github.com/user-attachments/assets/8dd16b7f-ce39-4ad2-9932-e911986e31c4)

https://github.com/user-attachments/assets/4a2e78b9-5b41-41c1-a460-3c3ce945aa61



# Experimental IP2V - Image Prompting to Video via VLM by @Dango233
## WORK IN PROGRESS - But it should work now!

Now you can feed image to the VLM as condition of generations! This is different from image2video where the image become the first frame of the video. IP2V uses image as a part of the prompt, to extract the concept and style of the image.
So - very much like IPAdapter - but VLM will do the heavy lifting for you!

Now this is a tuning free approach but with further task specific tuning we can expand the use scenarios.

## Guide to Using `xtuner/llava-llama-3-8b-v1_1-transformers` for Image-Text Tasks

## Step 1: Model Selection
Use the original `xtuner/llava-llama-3-8b-v1_1-transformers` model which includes the vision tower. You have two options:
- Download the model and place it in the `models/LLM` folder.
- Rely on the auto-download mechanism.

**Note:** It's recommended to offload the text encoder since the vision tower requires additional VRAM.

## Step 2: Load and Connect Image
- Use the comfy native node to load the image.
- Connect the loaded image to the `Hunyuan TextImageEncode` node.
  - You can connect up to 2 images to this node.

## Step 3: Prompting with Images
- Reference the image in your prompt by including `<image>`.
- The number of `<image>` tags should match the number of images provided to the sampler.
  - Example prompt: `Describe this <image> in great detail.`

You can also choose to give CLIP a prompt that does not reference the image separately.

## Step 4: Advanced Configuration - `image_token_selection_expression`
This expression is for advanced users and serves as a boolean mask to select which part of the image hidden state will be used for conditioning. Here are some details and recommendations:

- The hidden state sequence length (or number of tokens) per image in llava-llama-3 is 576.
- The default setting is `::4`, meaning every four tokens, one token goes into conditioning, interleaved, resulting in 144 tokens per image.
- Generally, more tokens lean more towards the conditional image.
- However, too many tokens (especially if the overall token count exceeds 256) will degrade generation quality. It's recommended not to use more than half the tokens (`::2`).
- Interleaved tokens generally perform better, but you might also want to try the following expressions:
  - `:128` - First 128 tokens.
  - `-128:` - Last 128 tokens.
  - `:128, -128:` - First 128 tokens and last 128 tokens.
- With a proper prompting strategy, even not passing in any image tokens (leaving the expression blank) can yield decent effects.

# Update

Scaled dot product attention (sdpa) should now be working (only tested on Windows, torch 2.5.1+cu124 on 4090), sageattention is still recommended for speed, but should not be necessary anymore making installation much easier.

Vid2vid test:
[source video](https://www.pexels.com/video/a-4x4-vehicle-speeding-on-a-dirt-road-during-a-competition-15604814/)

https://github.com/user-attachments/assets/12940721-4168-4e2b-8a71-31b4b0432314


text2vid (old test):

https://github.com/user-attachments/assets/3750da65-9753-4bd2-aae2-a688d2b86115


Transformer and VAE (single files, no autodownload):

https://huggingface.co/Kijai/HunyuanVideo_comfy/tree/main

Go to the usual ComfyUI folders (diffusion_models and vae)

LLM text encoder (has autodownload):

https://huggingface.co/Kijai/llava-llama-3-8b-text-encoder-tokenizer

Files go to `ComfyUI/models/LLM/llava-llama-3-8b-text-encoder-tokenizer`

Clip text encoder (has autodownload)

Either use any Clip_L model supported by ComfyUI by disabling the clip_model in the text encoder loader and plugging in ClipLoader to the text encoder node, or 
allow the autodownloader to fetch the original clip model from:

https://huggingface.co/openai/clip-vit-large-patch14, (only need the .safetensor from the weights, and all the config files) to:

`ComfyUI/models/clip/clip-vit-large-patch14`

Memory use is entirely dependant on resolution and frame count, don't expect to be able to go very high even on 24GB. 

Good news is that the model can do functional videos even at really low resolutions.
