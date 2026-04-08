
import torch
import torch.nn as nn
import timm
from peft import LoraConfig, get_peft_model, TaskType

def get_vit_model(num_classes=100, use_lora=False, lora_rank=4, lora_alpha=4, lora_dropout=0.1):
    """
    Load pretrained ViT-S model and modify classification head
    """
    # Load pretrained ViT-Small from timm
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    
    # Replace classification head with 100 classes
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    
    if use_lora:
        # Define LoRA config - inject into Q, K, V of attention
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["attn.qkv"],  # Q, K, V in ViT attention
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        print(f"LoRA applied | Rank: {lora_rank} | Alpha: {lora_alpha} | Dropout: {lora_dropout}")
        model.print_trainable_parameters()
    else:
        # Without LoRA - freeze everything except classification head
        for name, param in model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"No LoRA | Trainable: {trainable:,} / Total: {total:,}")
    
    return model


def get_trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
