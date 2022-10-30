import torch
import numpy as np
import sklearn.metrics as sk_metrics


class Metrics(object):
    def __init__(self):
        self.cumsum = 0.0
        self.num_elements = 0.0
        self.ave_value = 0.0
        self.log = []

    def evaluate(self, preds, labels):
        raise NotImplementedError('evaluate method must be overwritten by subclasses')

    def reset_state(self):
        self.log.append(self.ave_value)
        self.cumsum = 0.0
        self.num_elements = 0.0
        self.ave_value = 0.0

    def update_state(self, preds, labels):
        current_value = self.evaluate(preds, labels)
        self.cumsum += current_value
        self.num_elements += 1
        self.ave_value = self.cumsum / self.num_elements

    def clear_all(self):
        self.cumsum = 0.0
        self.num_elements = 0.0
        self.ave_value = 0.0
        self.log = []


class TopKAccuracy(Metrics):

    def __init__(self, k=1):
        super(TopKAccuracy, self).__init__()
        self.k = k

    def evaluate(self, preds, labels):
        with torch.no_grad():
            topk_preds, topk_inds = torch.topk(preds, k=self.k)
            labels = labels.view(-1, 1).expand_as(topk_inds)
            num_corrects = torch.sum(torch.eq(topk_inds, labels).sum(dim=1))
            accuracy = 100 * num_corrects.type(torch.float32) / len(labels)
            accuracy = accuracy.item()
        return accuracy


class EvalScore(Metrics):

    def __init__(self, mode='macro'):
        super(EvalScore, self).__init__()
        self.mode = mode
        self.all_preds = []
        self.all_labels = []
        self.name = None
        self.scorer = None

    def update_state(self, preds, labels):
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def reset_state(self):
        preds_array = torch.cat(self.all_preds, dim=0)
        preds_array = torch.argmax(preds_array, dim=1).cpu().numpy()
        labels_array = torch.cat(self.all_labels, dim=0).cpu().numpy()
        score = self.scorer(labels_array, preds_array, average=self.mode)
        self.log.append(score)
        self.all_preds = []
        self.all_labels = []


class TopKAccuracyEvalMode(EvalScore):
    def __init__(self, **kwargs):
        self.k = kwargs.pop('k')
        super(TopKAccuracyEvalMode, self).__init__(**kwargs)
        self.name = 'top{}_accuracy'.format(self.k)
        self.scorer = sk_metrics.top_k_accuracy_score

    def reset_state(self):
        preds_array = torch.cat(self.all_preds, dim=0).cpu().numpy()
        labels_array = torch.cat(self.all_labels, dim=0).cpu().numpy()
        score = self.scorer(labels_array, preds_array, k=self.k)
        self.log.append(score)
        self.all_preds = []
        self.all_labels = []


class Precision(EvalScore):
    def __init__(self, **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.name = 'precision'
        self.scorer = sk_metrics.precision_score


class Recall(EvalScore):
    def __init__(self, **kwargs):
        super(Recall, self).__init__(**kwargs)
        self.name = 'recall'
        self.scorer = sk_metrics.recall_score


class MeanAveragePrecision(EvalScore):
    def __init__(self, **kwargs):
        super(MeanAveragePrecision, self).__init__(**kwargs)
        self.name = 'average_precision'
        self.scorer = sk_metrics.average_precision_score
        self.n_classes = 27
        if 'mode' in kwargs:
            self.mode = kwargs['mode']

    def compute_ap(self, preds, labels):
        score = self.scorer(labels, preds, average=self.mode)
        return score

    def reset_state(self):
        preds_array = torch.cat(self.all_preds, dim=0)
        preds_array = torch.softmax(preds_array, dim=-1).cpu().numpy()
        labels_array = torch.cat(self.all_labels, dim=0).cpu().numpy()
        final_score = []
        for c in range(self.n_classes):
            binary_targets = (labels_array == c).astype(int)
            binary_preds = preds_array[:, c]
            ap_score = self.compute_ap(binary_preds, binary_targets)
            final_score.append(ap_score)
        final_score = np.mean(final_score).item()
        self.log.append(final_score)
        self.all_preds = []
        self.all_labels = []


class FScore(EvalScore):
    def __init__(self, **kwargs):
        super(FScore, self).__init__(**kwargs)
        self.name = 'f1_score'
        self.scorer = sk_metrics.f1_score


class AUC(EvalScore):
    def __init__(self, **kwargs):
        super(AUC, self).__init__(**kwargs)
        self.name = 'auc_score'
        self.scorer = sk_metrics.roc_auc_score

    def reset_state(self):
        preds_array = torch.cat(self.all_preds, dim=0)
        preds_array = torch.softmax(preds_array, dim=-1).cpu().numpy()
        labels_array = torch.cat(self.all_labels, dim=0).cpu().numpy()
        score = self.scorer(labels_array, preds_array, average=self.mode, multi_class='ovr')
        self.log.append(score)
        self.all_preds = []
        self.all_labels = []


class LossRecorder(Metrics):

    def __init__(self):
        super(LossRecorder, self).__init__()

    def update_state(self, current_loss, **kwargs):
        self.cumsum += current_loss.item()
        self.num_elements += 1
        self.ave_value = self.cumsum / self.num_elements

    def evaluate(self, preds, labels):
        raise NotImplementedError('there is no need to call this function. just added for clarity')


class BasicRecorder(object):
    """Record Values During training"""
    def __init__(self):
        self.log = []

    def update_state(self, current_value):
        self.log.append(current_value)


if __name__ == '__main__':
    targets = torch.randint(0, 26, size=(32,))
    predictions = torch.rand(size=(32, 27))
    metric = MeanAveragePrecision()
    metric.update_state(predictions, targets)
    metric.reset_state()
    # metric = TopKAccuracy(k=5)
    # metric.update_state(predictions, targets)
