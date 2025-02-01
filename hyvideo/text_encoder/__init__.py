from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel, LlavaForConditionalGeneration, AutoProcessor
from transformers.utils import ModelOutput

from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH
from ..constants import PRECISION_TO_TYPE
from ..utils.token_helper import find_subsequence, multi_slice_to_mask
from PIL import Image

def use_default(value, default):
    return value if value is not None else default

def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
    dtype=None,
    quantization_config=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "vlm":
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if text_encoder_precision is not None and quantization_config is None and dtype != torch.float8_e4m3fn:
        text_encoder = text_encoder.to(dtype)

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {dtype}")

    if device is not None and quantization_config is None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path

def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm" or tokenizer_type == "vlm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path

@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None

class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        input_max_length: Optional[int] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
        dtype=torch.float16,
        quantization_config=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        
        self.input_max_length = (
            input_max_length if input_max_length is not None else max_length
        )
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger
        self.is_fp8 = False
        self.processor = None

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or "glm" in text_encoder_type or "vlm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
            if "glm" in text_encoder_type or "vlm" in text_encoder_type:
                self.processor = AutoProcessor.from_pretrained(text_encoder_path, device=device)
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
            dtype=dtype,
            quantization_config=quantization_config,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text, prompt_template, image1=None, image2=None, clip_text_override=None):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        if self.text_encoder_type != "vlm" and image1 is not None:
            raise ValueError("Only vision_languague models support image input")
        tokenize_input_type = "str"
        if prompt_template is not None and self.text_encoder_type == "llm" or self.text_encoder_type == "vlm":
            if isinstance(text, (list, tuple)):
                text = [
                    self.apply_text_to_template(one_text, prompt_template["template"])
                    for one_text in text
                ]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template["template"])
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")
        elif clip_text_override is not None and self.text_encoder_type == "clipL":
            text = clip_text_override

        kwargs = dict(
            truncation=True,
            max_length=self.max_length,
            padding="max_length" if self.text_encoder_type != "vlm" else "do_not_pad",
            return_tensors="pt",
        )
        if tokenize_input_type == "str":
            text_tokens = self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
            if self.text_encoder_type == "vlm":
                raw_images = []
                if image1 is not None:
                    raw_images.append(image1.squeeze(0)*255)
                if image2 is not None:
                    raw_images.append(image2.squeeze(0)*255)
                text_tokens = self.processor(
                    raw_images, 
                    text, 
                    **kwargs,
                    ).to(0, torch.float16)
            return text_tokens #text_tokens
        elif tokenize_input_type == "list":
            return self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        prompt_template=None,
        image_token_selection_expr="::4",
        device=None,
        max_context_length=None,  # Add max_context_length here
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
            max_context_length (int): Maximum context length for the text encoder.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)
        attention_mask = (
            batch_encoding["attention_mask"].to(device) if use_attention_mask else None
        )
        for k,v in batch_encoding.items():
            batch_encoding[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        
        # Use max_context_length if provided, otherwise fall back to self.max_length
        max_length = max_context_length if max_context_length is not None else self.max_length

        if self.text_encoder_type == "vlm":
            outputs = self.model.language_model(
                **batch_encoding,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
                max_length=max_length  # Pass max_context_length to the model
            )
        else:
            outputs = self.model(
                **batch_encoding,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
            )

        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        if prompt_template is not None and self.text_encoder_type == "llm":
            crop_start = prompt_template.get("crop_start", -1)
            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                attention_mask = (
                    attention_mask[:, crop_start:] if use_attention_mask else None
                )
        elif prompt_template is not None and self.text_encoder_type == "vlm":
            # Temporory implementation for one round chat template to get rid of system prompts aand chat header
            user_start_tokens = self.tokenizer(
                text="<|start_header_id|>user<|end_header_id|>",
                add_special_tokens=False, 
                return_tensors="pt"
                )
            image_token = self.tokenizer(
                text="<image>",
                add_special_tokens=False, 
                return_tensors="pt"
                )
            image_token = image_token["input_ids"].to(device)
            user_start_tokens["input_ids"] = user_start_tokens["input_ids"].to(device)
            tk_idx, tk_n, tk_len = find_subsequence(batch_encoding["input_ids"], user_start_tokens["input_ids"])
            if tk_n != 1:
                raise ValueError("Template seems not in the required format, do you have <|start_header_id|>user<|end_header_id|> in place, and only one round of user input?")
            user_tokens = batch_encoding["input_ids"][:,tk_idx[0]+tk_len:]
            img_idx, img_n, _ = find_subsequence(user_tokens, image_token)
            img_seq_len=outputs["image_hidden_states"].shape[1]
            last_hidden_state = last_hidden_state[:, tk_idx[0]+tk_len:]
            # create image_mask to subset non-image hidden state
            seq_mask = torch.ones_like(last_hidden_state, device=device, dtype=torch.bool)
            img_mask=torch.zeros_like(outputs["image_hidden_states"][0:1], device=device, dtype=torch.bool)
            img_mask[:, multi_slice_to_mask(image_token_selection_expr, img_mask.shape[1])]=True
                
            drift=0  
            for i in img_idx:
                i = i+drift
                seq_mask[:,i:i+img_seq_len,:] = img_mask
                drift+=img_seq_len 
            
            last_hidden_state = last_hidden_state[seq_mask].view(1,-1,outputs["image_hidden_states"].shape[-1])

            attention_mask = torch.ones(last_hidden_state.shape[0], last_hidden_state.shape[1], device=device, dtype=torch.int64)

        elif prompt_template is None and self.text_encoder_type == "vlm":
            raise ValueError("Vlm encoders must use compatiable chat template.")

        if output_hidden_states:
            return TextEncoderModelOutput(
                last_hidden_state, attention_mask, outputs.hidden_states
            )
        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )