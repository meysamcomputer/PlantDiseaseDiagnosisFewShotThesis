from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys


def vis_tSNE(model, test_loader, device, backbone='VitTripletAttention'):
    model.eval()  # مدل را در حالت ارزیابی قرار می دهیم
    all_embeddings = []
    all_labels = []

    with torch.no_grad():  # بدون محاسبه گرادیان ها
        for batch in test_loader:
            # چک کردن تعداد مقادیر برگشتی
            if len(batch) == 4:
                anchor, positive, negative, labels = batch
            elif len(batch) == 3:
                anchor, positive, negative = batch
                labels = np.zeros(anchor.size(0))  # به صورت موقت برچسب ها را صفر قرار می دهیم
            else:
                continue  # اگر تعداد مقادیر به درستی تنظیم نشده بود، از این بخش عبور می کنیم

            # انتقال داده ها به دستگاه (GPU یا CPU)
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            labels = torch.tensor(labels).to(device)  # تبدیل برچسب ها به tensor و انتقال به دستگاه

            # استخراج ویژگی ها (embeddings)
            anchor_features = model.get_embedding(anchor)
            positive_features = model.get_embedding(positive)
            negative_features = model.get_embedding(negative)

            # اضافه کردن ویژگی ها و برچسب ها
            all_embeddings.append(anchor_features.cpu().numpy())  # می توانید فقط ویژگی های anchor را در نظر بگیرید
            all_labels.append(labels.cpu().numpy())

    # تبدیل لیست های ویژگی ها و برچسب ها به numpy arrays
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # بررسی برچسب ها
    print(f'Unique labels: {np.unique(all_labels)}')

    # اعمال t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(all_embeddings)

    # رسم نمودار t-SNE
    plt.figure(figsize=(16, 16))
    
    # انتخاب رنگ ها برای هر کلاس
    unique_labels = np.unique(all_labels)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    encoded_labels = label_encoder.transform(all_labels)

    # انتخاب رنگ های مناسب برای هر کلاس
    colors = plt.cm.get_cmap('tab20', len(unique_labels))  # خودکار انتخاب رنگ ها

    # اسامی کلاس ها به صورت دستی
    labels_name = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
                   'Potato___Late_blight', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
                   'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 
                   'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite']

    # رسم نمودارهای پراکندگی برای هر کلاس
    for i, label in enumerate(unique_labels):
        inds = np.where(encoded_labels == i)[0]
        plt.scatter(X_embedded[inds, 0], X_embedded[inds, 1], alpha=.8, color=colors(i), s=100, label=labels_name[i])

    # افزودن  (Legend)
    plt.legend(fontsize=15)
    plt.title(f't-SNE visualization of {backbone} model', fontsize=20)
    
    # ذخیره نمودار به فایل
    plt.savefig(f'./tsne_{backbone}.png')
    plt.show()