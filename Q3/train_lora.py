import torch
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model
from diffusers.models.attention_processor import AttnProcessor2_0
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm.auto import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Initialize Accelerator
# -----------------------------------------------------------------------------
accelerator = Accelerator(gradient_accumulation_steps=2)

# -----------------------------------------------------------------------------
# 2. Model Loading and LoRA Configuration
# -----------------------------------------------------------------------------
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", 
                                       torch_dtype=torch.bfloat16) # Use torch_dtype instead of dtype

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

# Ensure the VAE and text_encoder are not trained and are on the correct device
pipe.vae.to(accelerator.device)
pipe.text_encoder.to(accelerator.device)
pipe.vae.eval()
pipe.text_encoder.eval()

# -----------------------------------------------------------------------------
# 3. Dataset Loading and DataLoader Setup
# -----------------------------------------------------------------------------
dataset_name = 'lambda/naruto-blip-captions'
dataset = load_dataset(dataset_name)

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

# -----------------------------------------------------------------------------
# 4. Prepare model, optimizer, and dataloader with accelerator
# -----------------------------------------------------------------------------
optimizer = AdamW(pipe.unet.parameters(), lr=1e-4)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=100)

pipe.unet, optimizer, train_dataloader, scheduler = accelerator.prepare(
    pipe.unet, optimizer, train_dataloader, scheduler
)

# -----------------------------------------------------------------------------
# 5. Training Loop
# -----------------------------------------------------------------------------
num_epochs = 10
losses = []

pipe.unet.train() # Set UNET to training mode

for epoch in range(num_epochs):
    accelerator.print(f"Epoch {epoch+1}/{num_epochs}")
    for step, batch in enumerate(tqdm(train_dataloader, disable=not accelerator.is_main_process)):
        # Clear CUDA cache before each step to free up memory (optional, accelerator manages devices)
        # torch.cuda.empty_cache()

        with accelerator.accumulate(pipe.unet):
            # Encode the text to get prompt embeddings
            with torch.no_grad():
                text_embeddings = pipe.text_encoder(batch['input_ids'])[0]

            # Preprocess images to create latents
            latents = pipe.vae.encode(batch['pixel_values']).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            # Sample noise to add to the latents
            noise = torch.randn_like(latents)
            bs = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bs,), device=accelerator.device).long()

            # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict the noise residual
            noise_pred = pipe.unet(noisy_latents, timesteps, text_embeddings).sample

            # Calculate the loss (L2 loss between the target and the prediction)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        if step % 10 == 0 and accelerator.is_main_process:
            accelerator.print(f"  Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

# -----------------------------------------------------------------------------
# 6. Save LoRA weights and Plot Training Loss (only on main process)
# -----------------------------------------------------------------------------
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    accelerator.print("Fine-tuning complete!")
    accelerator.print(f"Final training loss: {losses[-1]:.4f}")

    # Save LoRA weights
    output_dir = "lora_weights"
    accelerator.unwrap_model(pipe.unet).save_pretrained(output_dir)
    accelerator.print(f"LoRA weights saved to {output_dir}")

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Steps")
    plt.grid(True)
    plt.savefig("training_loss.png")
    accelerator.print("Training loss plot saved as training_loss.png")