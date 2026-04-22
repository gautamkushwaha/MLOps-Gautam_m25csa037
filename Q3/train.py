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


# Fine-tuning the UNET with LoRA

from torchvision import transforms
from torch.utils.data import DataLoader

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def collate_fn(examples):
    input_ids = [pipe.tokenizer(example['text'], max_length=pipe.tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids[0] for example in examples]
    pixel_values = [transform(example['image']) for example in examples]

    pixel_values = torch.stack(pixel_values)
    input_ids = torch.stack(input_ids)

    return {
        'input_ids': input_ids,
        'pixel_values': pixel_values,
    }

# Create DataLoader
train_dataloader = DataLoader(
    dataset['train'], batch_size=1, shuffle=True, collate_fn=collate_fn
)

print(f"Number of batches in training DataLoader: {len(train_dataloader)}")


# Fine-tuning loop with gradient accumulation and CUDA memory management 

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm

# Optimizer and scheduler
optimizer = AdamW(pipe.unet.parameters(), lr=1e-4)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=100)

# Training loop parameters
num_epochs = 10
gradient_accumulation_steps = 2 # New: Accumulate gradients over 2 steps

# Move model to device
pipe.unet.to(pipe.device)
pipe.unet.train() # Set UNET to training mode

losses = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for step, batch in enumerate(tqdm(train_dataloader)):
        # Clear CUDA cache before each step to free up memory
        torch.cuda.empty_cache()

        # Encode the text to get prompt embeddings
        with torch.no_grad():
            text_embeddings = pipe.text_encoder(batch['input_ids'].to(pipe.device))[0]

        # Preprocess images to create latents
        latents = pipe.vae.encode(batch['pixel_values'].to(pipe.device)).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

        # Sample noise to add to the latents
        noise = torch.randn_like(latents)
        bs = latents.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bs,), device=pipe.device).long()

        # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict the noise residual
        noise_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample

        # Calculate the loss (L2 loss between the target and the prediction)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss = loss / gradient_accumulation_steps # New: Scale loss for gradient accumulation
        loss.backward()

        # Only perform optimizer step and zero_grad every `gradient_accumulation_steps`
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        if step % 10 == 0:
            print(f"  Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

# Final optimizer step if remaining gradients
if (step + 1) % gradient_accumulation_steps != 0:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

print("Fine-tuning complete!")
print(f"Final training loss: {losses[-1]:.4f}")


# 
import matplotlib.pyplot as plt

# Save LoRA weights
output_dir = "lora_weights"
pipe.unet.save_pretrained(output_dir)
print(f"LoRA weights saved to {output_dir}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.grid(True)
plt.show()