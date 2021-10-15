from tqdm import tqdm

import torchmetrics


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
        new_y_preds.append(int(y_preds_iter) / dividing_amount)
        new_y_true.append(int(y_true_iter) / dividing_amount)
    new_y_preds = torch.tensor(new_y_preds)
    new_y_true = torch.tensor(new_y_true)
    return torchmetrics.functional.accuracy(new_y_preds, new_y_true), new_y_preds, new_y_true
