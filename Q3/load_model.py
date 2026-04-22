import torch
import os
from diffusers import DiffusionPipeline
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# switch to "mps" for apple devices

# load the model and move it to GPU
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.bfloat16)


# load dataset
from datasets import load_dataset

dataset_name = 'lambda/naruto-blip-captions'
dataset = load_dataset(dataset_name)


# count the number of trainable parameters in the base model
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

total_params = count_parameters(pipe.unet) + count_parameters(pipe.vae) + count_parameters(pipe.text_encoder)

print(f"Total number of trainable parameters in the base model: {total_params:,}")


# Apply LoRA to the UNET and count the number of trainable parameters after applying LoRA 

from peft import LoraConfig, get_peft_model
from diffusers.models.attention_processor import AttnProcessor2_0, AttentionProcessor

# Set AttnProcessor
pipe.unet.set_attn_processor(AttnProcessor2_0())

# Configure LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=4,
    init_lora_weights="gaussian",
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
)

# Apply LoRA to the UNET
pipe.unet = get_peft_model(pipe.unet, lora_config)

# Print trainable parameters after LoRA
total_lora_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
print(f"Total number of trainable parameters after LoRA: {total_lora_params:,}")


# Calculate total parameters including both base and LoRA parameters
total_params_with_lora = total_params + total_lora_params
print(f"Total parameter count (Base + LoRA): {total_params_with_lora:,}")