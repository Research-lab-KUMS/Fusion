import torch
import pandas as pd
from utils import metrics as metrics_module
from utils.extra import SmoothedCrossEnropyLoss


class ModelEvaluator(object):

    def __init__(self, *metrics, model=None, checkpoint_dir=None, mode='single',
                 n_frames=1, batch_size=256, n_workers=0,
                 evaluate_train=False, evaluate_test=True):

        self.evaluate_train = evaluate_train
        self.evaluate_test = evaluate_test

        self.raw_metrics = metrics
        self.model = model
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.mode = mode

        self.train_metrics = [scorer(**params) for (scorer, params) in metrics]
        self.test_metrics = [scorer(**params) for (scorer, params) in metrics]

    def load_state(self, path=None):
        if path is None:
            path = str(self.checkpoint_dir / 'best_states.pt')
        states = torch.load(path)
        model_state = states['model']
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()

    def register_metrics(self):
        if self.evaluate_train is True:
            for metric in self.train_metrics:
                metric.reset_state()

        if self.evaluate_test is True:
            for metric in self.test_metrics:
                metric.reset_state()

    def update_metrics_states(self, preds, y, loss_value, split='test'):
        if split == 'train':
            for metric in self.train_metrics:
                if isinstance(metric, metrics_module.LossRecorder) is True:
                    metric.update_state(loss_value)
                else:
                    metric.update_state(preds, y)

        elif split == 'test':
            for metric in self.test_metrics:
                if isinstance(metric, metrics_module.LossRecorder) is True:
                    metric.update_state(loss_value)
                else:
                    metric.update_state(preds, y)

    def get_labels_and_scores(self, data):
        if self.mode == 'single':
            x = data[0].to(self.device, non_blocking=True)
            y = data[1].to(self.device, non_blocking=True)
            preds = self.model(x)
            return preds, y

        elif self.mode == 'fusion' or self.mode == 'multimodal':
            x_vision = data[0][0].to(self.device, non_blocking=True)
            x_tactile = data[0][1].to(self.device, non_blocking=True)
            y = data[1].to(self.device, non_blocking=True)
            preds = self.model(x_vision, x_tactile)
            return preds, y

        else:
            raise ValueError('Modality is not defined properly...')

    def evaluate(self,
                 train_dataset,
                 test_dataset,
                 states_path=None):

        self.load_state(path=states_path)
        loss_function = SmoothedCrossEnropyLoss(smoothing=0.0)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                       num_workers=self.n_workers, pin_memory=True,
                                                       drop_last=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                      shuffle=True, num_workers=self.n_workers,
                                                      pin_memory=True, drop_last=False)

        with torch.no_grad():
            if self.evaluate_train is True:
                for data in train_dataloader:
                    predictions, y = self.get_labels_and_scores(data)
                    loss = loss_function(predictions, y)
                    self.update_metrics_states(predictions, y, loss, split='train')
            else:
                for metric in self.train_metrics:
                    metric.log = [0]

            if self.evaluate_test is True:
                for data in test_dataloader:
                    predictions, y = self.get_labels_and_scores(data)
                    loss = loss_function(predictions, y)
                    self.update_metrics_states(predictions, y, loss, split='test')
            else:
                for metric in self.test_metrics:
                    metric.log = [0]

        self.register_metrics()

    def __add__(self, other):
        new_train_metrics = [scorer(**params) for (scorer, params) in self.raw_metrics]
        new_test_metrics = [scorer(**params) for (scorer, params) in self.raw_metrics]
        new_evaluator = ModelEvaluator(*self.raw_metrics, model=self.model,
                                       checkpoint_dir=self.checkpoint_dir,
                                       mode=self.mode, n_frames=self.n_frames,
                                       batch_size=self.batch_size,
                                       n_workers=self.n_workers)

        if isinstance(other, (int, float)):
            if self.evaluate_train:
                for self_metric, new_metric in zip(self.train_metrics, new_train_metrics):
                    new_metric.log.append(self_metric.log[0] + other)

            if self.evaluate_test:
                for self_metric, new_metric in zip(self.test_metrics, new_test_metrics):
                    new_metric.log.append(self_metric.log[0] + other)

            new_evaluator.train_metrics = new_train_metrics
            new_evaluator.test_metrics = new_test_metrics
            return new_evaluator

        else:
            if self.evaluate_train:
                for self_metric, other_metric, new_metric in zip(self.train_metrics,
                                                                 other.train_metrics,
                                                                 new_train_metrics):
                    sum_metric = self_metric.log[0] + other_metric.log[0]
                    new_metric.log.append(sum_metric)

            if self.evaluate_test:
                for self_metric, other_metric, new_metric in zip(self.test_metrics,
                                                                 other.test_metrics,
                                                                 new_test_metrics):
                    sum_metric = self_metric.log[0] + other_metric.log[0]
                    new_metric.log.append(sum_metric)

            new_evaluator.train_metrics = new_train_metrics
            new_evaluator.test_metrics = new_test_metrics
            return new_evaluator

    def __radd__(self, other):
        return self.__add__(other)

    def __truediv__(self, other):
        new_train_metrics = [scorer(**params) for (scorer, params) in self.raw_metrics]
        new_test_metrics = [scorer(**params) for (scorer, params) in self.raw_metrics]
        new_evaluator = ModelEvaluator(*self.raw_metrics, model=self.model,
                                       checkpoint_dir=self.checkpoint_dir,
                                       mode=self.mode, n_frames=self.n_frames,
                                       batch_size=self.batch_size,
                                       n_workers=self.n_workers)

        if isinstance(other, (int, float)):
            if self.evaluate_train:
                for self_metric, new_metric in zip(self.train_metrics, new_train_metrics):
                    new_metric.log.append(self_metric.log[0] / other)

            if self.evaluate_test:
                for self_metric, new_metric in zip(self.test_metrics, new_test_metrics):
                    new_metric.log.append(self_metric.log[0] / other)

            new_evaluator.train_metrics = new_train_metrics
            new_evaluator.test_metrics = new_test_metrics
            return new_evaluator
        else:
            raise NotImplementedError


def make_evaluater(*datasets, model, checkpoint_dir, n_frames, batch_size=64, n_workers=0, mode='single'):
    train_dataset, test_dataset = datasets

    metrics = [
        (metrics_module.Precision, {'mode': 'macro'}),
        (metrics_module.Recall, {'mode': 'macro'}),
        (metrics_module.MeanAveragePrecision, {'mode': 'macro'}),
        (metrics_module.FScore, {'mode': 'macro'}),
        (metrics_module.AUC, {'mode': 'macro'})
    ]

    evaluator = ModelEvaluator(*metrics, model=model, checkpoint_dir=checkpoint_dir,
                               mode=mode, n_frames=n_frames, batch_size=batch_size, n_workers=n_workers)
    evaluator.evaluate(train_dataset, test_dataset)
    return evaluator


if __name__ == '__main__':

    import pathlib
    from models.fusion_model import FusionModel
    from utils.multiloader import MultiDataset
    import pickle

    base_dir = pathlib.Path()
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'pretrained_models' / 'fusionnet-a'
    plot_dir = model_dir / 'Plots'
    plot_dir.mkdir(exist_ok=True, parents=True)

    top1s = []
    top3s = []
    losses = []
    evaluator_results = []

    for N_FRAMES in range(1, 9):

        train_dataset = MultiDataset(data_dir, split='train', n_frames=N_FRAMES)
        test_dataset = MultiDataset(data_dir, split='test', n_frames=N_FRAMES)

        framepath = model_dir / '{}_frame'.format(N_FRAMES)
        num_runs = len([p for p in framepath.iterdir() if p.is_dir() and p.parts[-1] != 'results'])
        model = FusionModel(n_frames=N_FRAMES, mode='early')

        runs_top1s = []
        runs_top3s = []
        runs_losses = []
        agg_evaluator = 0

        for n in range(1, num_runs + 1):
            run_path = framepath / 'run{}'.format(n)
            metrics = torch.load(run_path / 'best_states.pt')['metrics']

            top1 = max(metrics['test_top1'])
            top3 = max(metrics['test_top3'])
            loss = min(metrics['test_loss'])
            runs_top1s.append(top1)
            runs_top3s.append(top3)
            runs_losses.append(loss)
            runs_top1s.append(top1)
            runs_top3s.append(top3)
            runs_losses.append(loss)

            single_run_evaluator = make_evaluater(train_dataset,
                                                  test_dataset,
                                                  model=model,
                                                  checkpoint_dir=run_path,
                                                  n_frames=N_FRAMES,
                                                  batch_size=64,
                                                  n_workers=0,
                                                  mode='fusion')

            agg_evaluator = agg_evaluator + single_run_evaluator

        agg_evaluator = agg_evaluator / num_runs
        agg_evaluator.model = None
        evaluator_results.append(agg_evaluator)
        top1s.append(max(runs_top1s))
        top3s.append(max(runs_top3s))
        losses.append(min(runs_losses))

    cols = ['Top-1(Orig)', 'Top-3(Orig)', 'Precision', 'Recall', 'MAP', 'F1-Score', 'AUC']
    inds = ['{}_frame'.format(n) for n in range(1, 9)]
    res_table = pd.DataFrame(columns=cols, index=inds)

    for nf in range(1, 9):
        nf_eval = evaluator_results[nf - 1]
        top1_orig = top1s[nf - 1]
        top3_orig = top3s[nf - 1]
        precision = nf_eval.test_metrics[0].log[0]
        recall = nf_eval.test_metrics[1].log[0]
        mean_ap = nf_eval.test_metrics[2].log[0]
        f1score = nf_eval.test_metrics[3].log[0]
        auc = nf_eval.test_metrics[4].log[0]

        res_table.iloc[nf - 1, 0] = top1_orig
        res_table.iloc[nf - 1, 1] = top3_orig
        res_table.iloc[nf - 1, 2] = precision
        res_table.iloc[nf - 1, 3] = recall
        res_table.iloc[nf - 1, 4] = mean_ap
        res_table.iloc[nf - 1, 5] = f1score
        res_table.iloc[nf - 1, 6] = auc

    res_table.to_csv(model_dir / 'res_fusionnet-a_best.csv')
    with open(model_dir / 'evaluators_best.pkl', 'wb') as file:
        pickle.dump(evaluator_results, file)
