
import torch
from torch import Tensor
import torch.nn.functional as F
import sys
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score, multiclass_auroc, multiclass_confusion_matrix

def mse(y_true, y_pred):
    return F.mse_loss(y_true, y_pred, reduction='mean')

def rmse(y_true, y_pred):
    return torch.sqrt(F.mse_loss(y_true, y_pred, reduction='mean'))

def mae(y_true, y_pred):
    return F.l1_loss(y_true, y_pred, reduction='mean')

def r2_score(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error
    return mean_absolute_percentage_error(y_true, y_pred)

def accuracy(y_true, y_pred):
    _, predicted = torch.max(y_pred, 1)  # Get predicted class indices
    # predicted = y_pred.argmax(1)
    correct = (y_true == predicted).sum().item()
    total = y_true.size(0)
    return correct / total

def precision(y_true, y_pred):
    _, predicted = torch.max(y_pred, 1)  # Get predicted class indices
    return multiclass_precision(predicted, y_true, num_classes=y_pred.size(1), average='macro')

def recall(y_true, y_pred):
    _, predicted = torch.max(y_pred, 1)  # Get predicted class indices
    return multiclass_recall(predicted, y_true, num_classes=y_pred.size(1), average='macro')

def f1_score(y_true, y_pred):
    _, predicted = torch.max(y_pred, 1)  # Get predicted class indices
    return multiclass_f1_score(predicted, y_true, num_classes=y_pred.size(1), average='macro')

def auroc(y_true, y_pred):
    predicted = F.softmax(y_pred, dim=1) # even if we don't apply softmax, multiclass_auroc will do it internally
    return multiclass_auroc(predicted, y_true, num_classes=y_pred.size(1), average='macro')

def conf_mat(y_true, y_pred):
    _, predicted = torch.max(y_pred, 1)  # Get predicted class indices
    conf_matrix = multiclass_confusion_matrix(predicted, y_true, num_classes=y_pred.size(1))
    print("Confusion Matrix:\n", conf_matrix)#, file=sys.stderr)
    return conf_matrix[0][0]