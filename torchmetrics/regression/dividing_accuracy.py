import torch
from tqdm import tqdm


def get_percentile_accuracy(y_true, y_preds, dividing_amount) -> float:
    """
    So this will be used for regression,
    So if we have a small preds and y true
    preds = [1000,2000,3000,4000,5000]
    y_true = [1250,2500,3250,4526,5262]
    we are going to round the preds and the y_true
    so then
    preds = [1,2,3,4,5]
    y_true = [1,2,3,4,5]
    """
    new_y_preds = []
    new_y_true = []
    for y_true_iter, y_preds_iter in tqdm(zip(y_true, y_preds)):
        new_y_preds.append(torch.tensor(int(int(y_preds_iter) / dividing_amount)).float())
        new_y_true.append(torch.tensor(int(y_true_iter) / dividing_amount).float())
    new_y_preds = torch.tensor(new_y_preds)
    new_y_true = torch.tensor(new_y_true)
    correct = 0
    total = 0
    for pred, yb in zip(new_y_preds, new_y_true):
        pred = int(torch.round(pred))
        yb = int(torch.round(yb))
        if pred == yb:
            correct += 1
        total += 1
    acc = round(correct / total, 3) * 100
    return acc
