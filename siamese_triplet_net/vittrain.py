#!pip install transformers datasets      
import sys
#sys.path.insert(0,'f:/Meysam-Khodarahi/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/')
sys.path.insert(0,'C:/Users/Mey/Documents/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/')

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# بارگذاری مسیر داده‌ها
path_data = 'C:/Users/Mey/Documents/PlantDiseaseDiagnosisFewShotLearning/siamese_triplet_net/src/dataset'

# بارگذاری feature extractor از ViT
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# تعریف transform مناسب برای پردازش تصاویر
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # تغییر اندازه به 224x224
    transforms.ToTensor(),          # تبدیل تصویر به Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # نرمال‌سازی
])

# بارگذاری داده‌ها با استفاده از ImageFolder و transform مناسب
train_data = datasets.ImageFolder(root=path_data + '/train/', transform=train_transform)
train_loader = DataLoader(train_data, batch_size=16)

test_data = datasets.ImageFolder(root=path_data + '/test/', transform=train_transform)
test_loader = DataLoader(test_data, batch_size=64)

# بارگذاری مدل ViT برای طبقه‌بندی تصویر
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10)

# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir="./vit_finetuned",         # مکانی برای ذخیره مدل نهایی
    evaluation_strategy="epoch",          # ارزیابی در پایان هر epoch
    learning_rate=2e-5,                   # نرخ یادگیری برای fine-tuning
    per_device_train_batch_size=16,       # اندازه بچ برای آموزش
    per_device_eval_batch_size=64,        # اندازه بچ برای ارزیابی
    num_train_epochs=3,                   # تعداد epochs برای آموزش
    weight_decay=0.01,                    # تنظیمات weight decay
    save_total_limit=2,                   # ذخیره تنها آخرین ۲ checkpoint
    logging_dir='./logs',                 # مسیر ذخیره لاگ‌ها
)

# تابع محاسبه متریک‌ها
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# راه‌اندازی Trainer
trainer = Trainer(
    model=model,                           # مدل
    args=training_args,                    # تنظیمات آموزش
    train_dataset=train_data,              # داده‌های آموزشی
    eval_dataset=test_data,                # داده‌های ارزیابی
    compute_metrics=compute_metrics        # محاسبه متریک‌ها
)

# شروع آموزش مدل
trainer.train()

# ارزیابی مدل پس از آموزش
results = trainer.evaluate()

print("Evaluation Results:")
print(results)

# استخراج ویژگی‌ها و برچسب‌ها برای t-SNE
model.eval()
train_features = []
train_labels = []

# استخراج ویژگی‌ها از مدل
with torch.no_grad():
    for batch in train_loader:
        inputs, labels = batch
        outputs = model(pixel_values=inputs)  # ورودی‌های پیش پردازش شده به مدل می‌دهیم
        features = outputs.logits.cpu().numpy()
        train_features.extend(features)
        train_labels.extend(labels.cpu().numpy())

train_features = np.array(train_features)
train_labels = np.array(train_labels)

# استفاده از t-SNE برای کاهش ابعاد
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(train_features)

# رسم نتیجه t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=train_labels, cmap='viridis', s=10)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of ViT Features")
plt.show()

# ذخیره مدل آموزش‌دیده
trainer.save_model("./vit_finetuned_model")
