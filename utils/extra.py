import torch
import torch.nn.functional as F


class EarlyStopping(object):

    def __init__(self, patience):
        self.best_acc = 0.0
        self.patience = patience
        self.counter = 0

    def _count(self, acc):

        if acc < self.best_acc:
            self.counter += 1
        else:
            self.best_acc = acc
            self.counter = 0

    def stop_early(self, acc):
        self._count(acc)
        if self.counter > self.patience:
            return True
        else:
            return False


class SmoothedCrossEnropyLoss(torch.nn.NLLLoss):

    def __init__(self, smoothing=0.0, n_classes=27, add_softmax=True, **kwargs):
        super(SmoothedCrossEnropyLoss, self).__init__(**kwargs)
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.add_softmax = add_softmax

    def forward(self, raw_preds, labels):

        if self.add_softmax is True:
            log_softmax_preds = F.log_softmax(raw_preds, dim=1)
        else:
            softmax_preds = raw_preds
            log_softmax_preds = torch.log(softmax_preds)

        onehot_labels = torch.nn.functional.one_hot(labels, self.n_classes)
        smoothed_onehots = onehot_labels * (1 - self.smoothing)
        smoothed_onehots += (self.smoothing / self.n_classes)
        loss = - (log_softmax_preds * smoothed_onehots).sum(dim=1).mean()

        return loss


class CPCE(torch.nn.NLLLoss):
    """Cross Entropy that Penalizes the Confident Outputs"""

    def __init__(self, beta=0.0, n_classes=27, add_softmax=True, **kwargs):
        super(CPCE, self).__init__(**kwargs)
        self.beta = beta
        self.n_classes = n_classes
        self.add_softmax = add_softmax

    def forward(self, raw_preds, labels):

        if self.beta == 0:
            if self.add_softmax is True:
                criterion = torch.nn.CrossEntropyLoss()
            else:
                raw_preds = torch.log(raw_preds)
                criterion = torch.nn.NLLLoss()
            loss = criterion(raw_preds, labels)

        else:

            if self.add_softmax is True:
                softmax_preds = F.softmax(raw_preds, dim=1)
                log_softmax_preds = torch.log(softmax_preds)
            else:
                softmax_preds = raw_preds
                log_softmax_preds = torch.log(raw_preds)

            onehot_labels = torch.nn.functional.one_hot(labels, self.n_classes)
            ce_loss = - (log_softmax_preds * onehot_labels).sum(dim=1)
            output_entropy = -(softmax_preds * log_softmax_preds).sum(dim=1)
            loss = (ce_loss - self.beta * output_entropy).mean()

        return loss


class MultiLossWrapper(torch.nn.NLLLoss):

    def __init__(self, criterions, weights=None, **kwargs):
        super(MultiLossWrapper, self).__init__(**kwargs)
        self.criterions = criterions

        if weights is None:
            self.weights = torch.tensor(len(criterions) * [1.0])
        else:
            if isinstance(weights, torch.Tensor) is not True:
                weights = torch.tensor(weights)
            self.weights = weights

    def forward(self, raw_preds, labels):

        total_loss = []
        for w, c in zip(self.weights, self.criterions):
            total_loss.append(w * c(raw_preds, labels))
        total_loss = torch.stack(total_loss).sum()
        return total_loss


if __name__ == "__main__":

    torch.manual_seed(42)

    rand_labels = torch.randint(0, 26, size=(32,), dtype=torch.int64)
    rand_preds = torch.rand(size=(32, 27), dtype=torch.float32)

    smce = SmoothedCrossEnropyLoss(smoothing=0.0, n_classes=27)
    ce = torch.nn.CrossEntropyLoss()
    print(smce(rand_preds, rand_labels), ce(rand_preds, rand_labels))
