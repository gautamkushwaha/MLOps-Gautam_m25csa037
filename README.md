# ResNet Docker Training (MLOps Assignment)

## Project Overview
Training ResNet18 using Docker container with GPU acceleration.

---

## Folder Structure

resnet_docker/
│
├── train.py
├── Dockerfile
├── README.md
└── data/train/

---

## Build Docker Image

```bash
docker build -t resnet-training:v1 .

## Running container

docker run -it --rm \
--gpus all \
--shm-size=8g \
-v $(pwd):/workspace \
resnet-training:v1


## 📊 Model Evaluation Results

The trained ResNet-18 model was evaluated on the test dataset after loading the saved weights.

✅ **Model Loaded Successfully**

---

### 🔹 Overall Performance

| Metric | Value |
|-------|------|
| Accuracy | **97.64%** |
| F1 Score | **0.9762** |

---

### 🔹 Classification Report
          precision    recall  f1-score   support

       0       0.97      0.99      0.98       490
       1       0.98      0.99      0.99       568
       2       1.00      0.97      0.98       516
       3       0.94      1.00      0.97       505
       4       0.99      0.96      0.97       491
       5       0.95      0.98      0.97       446
       6       0.99      0.98      0.98       479
       7       0.99      0.98      0.99       514
       8       1.00      0.95      0.97       487
       9       0.96      0.96      0.96       505


---

### 🔹 Class-wise Accuracy

| Class | Accuracy |
|------|----------|
| Class 0 | 99.39% |
| Class 1 | 99.12% |
| Class 2 | 97.09% |
| Class 3 | 99.80% |
| Class 4 | 95.52% |
| Class 5 | 98.43% |
| Class 6 | 97.91% |
| Class 7 | 98.25% |
| Class 8 | 94.87% |
| Class 9 | 95.84% |

---

### 🔹 Example Prediction

- **Image:** `data/data/test/5/340.png`
- **Predicted Class:** 5
- **Confidence:** 92.19%

---

### ✅ Summary

The pretrained **ResNet-18** model achieves high classification performance with strong precision and recall across all classes, demonstrating effective generalization on the test dataset.
