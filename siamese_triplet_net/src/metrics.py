import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

class MetricsFinal:
    def __init__(self):
        print("Final Metrics for evaluation")
        
    
# فرض بر این است که test_loader داده های تست را می گیرد
# و مدل قبلاً آموزش دیده است.

    def evaluate_triplet_model(model, test_loader, device):
        model.eval()  # مدل را در حالت ارزیابی قرار می دهیم.

        y_true = []  # برای ذخیره برچسب های واقعی
        y_pred = []  # برای ذخیره پیش بینی ها

        with torch.no_grad():  # بدون محاسبه گرادیان ها
            for anchor, positive, negative,anchor_Label in test_loader:
                # انتقال داده ها به دستگاه (GPU یا CPU)
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                # استخراج ویژگی ها (embeddings)
                anchor_features = model.get_embedding(anchor)
                positive_features = model.get_embedding(positive)
                negative_features = model.get_embedding(negative)
 

                # محاسبه فاصله ها
                distance_positive = F.pairwise_distance(anchor_features, positive_features)
                distance_negative = F.pairwise_distance(anchor_features, negative_features)

                # پیش بینی ها (پیش بینی 1 اگر فاصله مثبت کمتر از منفی باشد)
                predictions = (distance_positive < distance_negative).float()

                # اضافه کردن برچسب ها و پیش بینی ها به لیست ها
                y_true.extend([1] * len(anchor))  # جفت های مثبت
                y_pred.extend(predictions.cpu().numpy())

        # تبدیل لیست ها به آرایه های numpy برای محاسبه معیارها
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # محاسبه معیارهای ارزیابی
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')

        return accuracy, precision, recall, f1    

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
