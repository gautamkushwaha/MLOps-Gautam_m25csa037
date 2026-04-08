
# Assignment 5 - Deep Learning Operations
## ViT Fine-tuning with LoRA & Adversarial Attacks


**Name:** Gautam Kumar 
**Roll Number:** M25CSA037 
**Institution:** Indian Institute of Technology Jodhpur


---


## 🔗 Important Links
- 📊 **WandB:** https://wandb.ai/m25csa037-iit-jodhpur/DLops-Assignment-5
- 🤗 **HuggingFace:** https://huggingface.co/Gautam20/Assignment_5_mlops/tree/main
- 📋 **Report:** M25CSA037_Gautam_kushwaha_Ass5


---


```text


## 📁 Repository Structure
Assignment_5/
├── Q1/
│   ├── model.py           # ViT model with LoRA
│   ├── utils.py           # Data loading utilities
│   ├── train.py           # Training script
│   ├── best_model_lora_r8_a8.pth   # uploaded in huggingface
│   ├── test.py         # Testing script
│   └── optuna_search.py   # Hyperparameter search
├── Q2/
│   ├── train_clean.py     # Train ResNet18 on CIFAR10
│   ├── fgsm_scratch.py    # FGSM attack from scratch
│   ├── fgsm_art.py        # FGSM attack using IBM ART
│   ├── detector_pgd.py    # PGD adversarial detector
│   ├── detector_bim.py    # BIM adversarial detector
│   ├── detector_pgd.pth   # PGD detector weights
│   └── detector_bim.pth   # BIM detector weights
├── Dockerfile
└── requirements.txt
└── M25CSA037_Gautam_kushwaha_Ass5.pdf


---


## 🐳 Docker Setup


### Build container:
```bash
docker build -t dlops-assignment5 .
```


### Run container:
```bash
docker run --gpus all -it dlops-assignment5 bash
```


---


## ⚙️ Installation
```bash
pip install -r requirements.txt
```


Or install manually:
```bash
pip install torch==2.1.0 torchvision==0.16.0 timm==0.9.12 peft==0.7.1 wandb==0.16.0 optuna==3.4.0 adversarial-robustness-toolbox==1.16.0 huggingface_hub==0.20.1 matplotlib==3.8.0 numpy==1.24.0 scikit-learn==1.3.0 tqdm==4.66.0
```


---


## 🚀 Q1 - ViT Fine-tuning on CIFAR-100


#### 📈 Training Graphs
All training graphs, loss curves and accuracy plots are available on WandB:
👉 https://wandb.ai/m25csa037-iit-jodhpur/DLops-Assignment-5


### Train without LoRA:
```bash
cd Q1
python train.py --epochs 10 --lr 1e-3 --batch_size 64
```


### Train with LoRA:
```bash
python train.py --use_lora --rank 8 --alpha 8 --dropout 0.1 --epochs 10 --lr 1e-3 --batch_size 64
```


### Test best model:
```bash
python test.py --model_path best_model_lora_r8_a8.pth --use_lora --rank 8 --alpha 8 --dropout 0.1
```


### Optuna hyperparameter search:
```bash
python optuna_search.py
```


---


## 📊 Q1 Results


### Experiment Summary Table


| LoRA | Rank | Alpha | Dropout | Best Val Acc | Test Acc | Trainable Params |
|------|------|-------|---------|-------------|----------|-----------------|
| No   | -    | -     | 0.1     |83.08%       | -        | 38,500          |
| Yes  | 2    | 2     | 0.1     | 74.32%      | -        | 36,864          |
| Yes  | 2    | 4     | 0.1     | 76.74%      | -        | 36,864          |
| Yes  | 2    | 8     | 0.1     | 78.86%      | -        | 36,864          |
| Yes  | 4    | 2     | 0.1     | 80.54%      | -        | 73,728          |
| Yes  | 4    | 4     | 0.1     | 83.90%      | -        | 73,728          |
| Yes  | 4    | 8     | 0.1     | 83.02%      | -        | 73,728          |
| Yes  | 8    | 2     | 0.1     | 84.34%      | -        | 147,456         |
| Yes  | 8    | 4     | 0.1     | 86.70%      | -        | 147,456         |
| Yes  | 8    | 8     | 0.1     | 86.86%      | 86.40%   | 147,456         |


### No LoRA Training Progress


| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 0.9922    | 0.6802   | 73.78%    | 79.88%  |
| 2     | 0.5943    | 0.6451   | 82.39%    | 81.16%  |
| 3     | 0.5189    | 0.6499   | 84.18%    | 81.22%  |
| 4     | 0.469     | 0.6091   | 85.39%    | 82.42%  |
| 5     | 0.4295    | 0.6242   | 86.64%    | 82.22%  |
| 6     | 0.398     | 0.6167   | 87.56%    | 82.84%  |
| 7     | 0.3679    | 0.6187   | 88.56%    | 82.94%  |
| 8     | 0.3434    | 0.6144   | 89.30%    | 82.84%  |
| 9     | 0.3324    | 0.598    | 89.76%    | 83.08%  |
| 10    | 0.3206    | 0.6017   | 90.08%    | 82.62%  |


### Best LoRA (Rank=8, Alpha=8) Training Progress


| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1     | 2.1596    | 0.9492   | 48.34%    | 74.88%  |
| 2     | 0.7364    | 0.6757   | 80.25%    | 81.54%  |
| 3     | 0.5354    | 0.5949   | 85.44%    | 83.62%  |
| 4     | 0.4352    | 0.5448   | 88.18%    | 85.08%  |
| 5     | 0.3671    | 0.5376   | 90.00%    | 85.38%  |
| 6     | 0.3129    | 0.5149   | 91.55%    | 85.08%  |
| 7     | 0.2703    | 0.5121   | 92.94%    | 85.94%  |
| 8     | 0.2387    | 0.4877   | 93.89%    | 86.86%  |
| 9     | 0.2161    | 0.4981   | 94.66%    | 86.46%  |
| 10    | 0.2035   | 0.4825   | 95.09%    | 86.64%  |


### Optuna Best Configuration
- **Rank:** 8
- **Alpha:** 2
- **Dropout:** 0.1911
- **LR:** 0.002709
- **Best Val Accuracy:** 82.58%


---


## 🚀 Q2 - Adversarial Attacks


### Train ResNet18 on CIFAR-10:
```bash
cd Q2
python train_clean.py
```


### FGSM attack from scratch:
```bash
python fgsm_scratch.py
```


### FGSM attack using IBM ART:
```bash
python fgsm_art.py
```


### Train PGD detector:
```bash
python detector_pgd.py
```


### Train BIM detector:
```bash
python detector_bim.py
```


---


## 📊 Q2 Results


#### 📈 Training Graphs
All training graphs, loss curves and accuracy plots are available on WandB:
👉 [Wandb Link](https://github.com/gautamkushwaha/MLOps-Gautam_m25csa037/tree/Assignment_5)


### ResNet18 Clean Accuracy: 93.85% (Target: ≥72%) ✅


### FGSM Attack Results


| Epsilon | Clean Acc | Scratch Acc | ART Acc | Scratch Drop | ART Drop |
|---------|-----------|-------------|---------|--------------|----------|
| 0.01    | 93.85%    | 48.07%      | 31.60%  | 45.78%       | 61.90%   |
| 0.05    | 93.85%    | 37.07%      | 14.50%  | 56.82%       | 79.00%   |
| 0.10    | 93.85%    | 30.32%      | 10.80%  | 63.53%       | 82.70%   |
| 0.20    | 93.85%    | 23.63%      | 10.40%  | 70.22%       | 83.10%   |
| 0.30    | 93.85%    | 19.83%      | 10.10%  | 74.02%       | 83.40%   |


### Adversarial Detector Results


| Attack | Model    | Detection Accuracy | Target | Status    |
|--------|----------|--------------------|--------|-----------|
| PGD    | ResNet34 | 100.00%            | ≥70%   | ✅ Passed |
| BIM    | ResNet34 | 100.00%            | ≥70%   | ✅ Passed |


---




## 🏆 Model Weights


> **Note:** Best Q1 model weight (best_model_lora_r8_a8.pth) exceeds
> GitHub's 25MB file size limit and has been uploaded to HuggingFace instead.
> All Q2 weights are available on GitHub directly.


| Model | Size | Location |
|-------|------|----------|
| Best Q1 (LoRA Rank=8, Alpha=8) | >25MB | 🤗 HuggingFace: /best_model_lora_r8_a8.pth |
| Q2 ResNet18 Clean | >25MB | 🤗 HuggingFace: /resnet18_clean.pth |
| Q2 PGD Detector | >25MB | HuggingFace: /detector_pgd.pth |
| Q2 BIM Detector | >25MB | HuggingFace: /detector_bim.pth |


🤗 **HuggingFace Repository:** https://huggingface.co/Gautam20/Assignment_5_mlops/tree/main
| Model | Location |
|-------|----------|
| Best Q1 (LoRA Rank=8, Alpha=8) | 🤗 HuggingFace: /best_model_lora_r8_a8.pth |
| Q2 ResNet18 Clean | 🤗 HuggingFace: /resnet18_clean.pth |
| Q2 PGD Detector |  HuggingFace: /detector_pgd.pth |
| Q2 BIM Detector |  HuggingFace: /detector_bim.pth |
