import torch
import numpy as np
import matplotlib.pyplot as plt
from evaluate import ModelEvaluator
from utils.data_loader import TouchDataset


class ConfidenceAnalyser(ModelEvaluator):

    def __init__(self,
                 model,
                 model_dir,
                 data_dir,
                 n_frames=1,
                 batch_size=-1,
                 **kwargs):

        super(ConfidenceAnalyser, self).__init__(**kwargs)
        self.model = model
        self.model_dir = model_dir / '{}_frame'.format(n_frames)
        self.savefig_dir = self.model_dir / 'results'
        self.savefig_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_frames = n_frames

    def get_dataloaders(self, class_id=None):
        train_dataset = TouchDataset(self.data_dir,
                                     split='train',
                                     n_frames=self.n_frames,
                                     specific_class=class_id)

        test_dataset = TouchDataset(self.data_dir,
                                    split='test',
                                    n_frames=self.n_frames,
                                    specific_class=class_id)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.n_workers,
                                                       pin_memory=True,
                                                       drop_last=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=self.n_workers,
                                                      pin_memory=True,
                                                      drop_last=False)

        return train_dataloader, test_dataloader

    def predict_batch(self, X, prob_scores):
        raw_scores = self.model(X)
        probs = torch.softmax(raw_scores, dim=1)
        if prob_scores is True:
            return probs
        else:
            return raw_scores

    def predict_all(self, prob_scores, class_id=None):
        train_dataloader, test_dataloader = self.get_dataloaders(class_id=class_id)
        with torch.no_grad():
            train_preds = []
            for data in train_dataloader:
                X, y = data[0].to(self.device), data[0].to(self.device)
                preds = self.predict_batch(X, prob_scores)
                train_preds.append(preds)

            test_preds = []
            for data in test_dataloader:
                X, y = data[0].to(self.device), data[0].to(self.device)
                preds = self.predict_batch(X, prob_scores)
                test_preds.append(preds)

        train_preds = torch.cat(train_preds, dim=0).cpu().numpy()
        test_preds = torch.cat(test_preds, dim=0).cpu().numpy()
        return train_preds, test_preds

    @staticmethod
    def hist_builder(train_conf, test_conf, save_path, n_bins=20):

        train_hist = np.histogram(train_conf, bins=n_bins, range=(0, 1))
        test_hist = np.histogram(test_conf, bins=n_bins, range=(0, 1))
        mean_train_conf = np.mean(train_conf)
        mean_test_conf = np.mean(test_conf)

        bin_names = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bins = train_hist[1][1:]
        samper_train = train_hist[0] / train_hist[0].sum()
        samper_test = test_hist[0] / test_hist[0].sum()

        fig, axs = plt.subplots(2, 1, sharex='none', figsize=(12, 12))
        plt.subplots_adjust(hspace=0.4)

        axs[0].bar(bins, samper_train, width=-1 / len(bins),
                   align='edge', tick_label=None, linewidth=1.75,
                   edgecolor='black', color='olivedrab')

        axs[0].set_xlim(0, 1)
        axs[0].set_ylim(0, 1.04)

        axs[0].set_ylabel('% of Training Samples', fontsize=20, weight='bold')
        axs[0].set_xlabel('Output Confidence', fontsize=20, weight='bold')

        axs[0].set_xticks(bin_names)
        axs[0].set_xticklabels(bin_names, weight='bold')
        axs[0].set_yticks(np.arange(0.0, 1.04, 0.2))
        axs[0].set_yticklabels(np.round(np.arange(0.0, 1.04, 0.2), 2), weight='bold')

        axs[0].grid(linestyle='-.')
        axs[0].set_title('Confidence Histogram of the Train Set', fontsize=22, weight='bold')
        axs[0].axvline(mean_train_conf - 0.009, 0, 1, color='red',
                       linestyle='--', label='Mean Confidence', linewidth=2.5)
        axs[0].legend(loc='upper left', prop={'weight': 'bold', 'size': 16})
        axs[0].set_axisbelow(True)

        axs[0].set_ylim(0.0, 1.04)
        axs[0].tick_params(axis='y', length=6, width=2, labelsize=18)
        axs[0].tick_params(axis='x', length=6, width=2, labelsize=18)
        axs[0].spines['left'].set_linewidth(2)
        axs[0].spines['bottom'].set_linewidth(2)

        axs[1].bar(bins, samper_test, width=-1 / len(bins),
                   align='edge', tick_label=None, linewidth=1.75,
                   edgecolor='black', color='darkcyan')

        axs[1].set_xlim(0, 1)
        axs[1].set_ylim(0, 0.36)

        axs[1].set_ylabel('% of Test Samples', fontsize=20, weight='bold')
        axs[1].set_xlabel('Output Confidence', fontsize=20, weight='bold')

        axs[1].set_xticks(bin_names)
        axs[1].set_xticklabels(np.round(bin_names, 2), weight='bold')
        axs[1].set_yticks(np.arange(0.0, 0.36, 0.07))
        axs[1].set_yticklabels(np.round(np.arange(0.0, 0.36, 0.07), 2), weight='bold')

        axs[1].grid(which='both', linestyle='-.')
        axs[1].set_title('Confidence Histogram of the Test Set', fontsize=22, weight='bold')
        axs[1].axvline(mean_test_conf + 0.0075, 0, 1, color='red',
                       linestyle='--', label='Mean Confidence', linewidth=2.5)
        axs[1].legend(loc='upper left', prop={'weight': 'bold', 'size': 16})
        axs[1].set_axisbelow(True)

        axs[1].tick_params(axis='y', length=6, width=2, labelsize=18)
        axs[1].tick_params(axis='x', length=6, width=2, labelsize=18)
        axs[1].spines['left'].set_linewidth(2)
        axs[1].spines['bottom'].set_linewidth(2)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close('all')

    @staticmethod
    def plot_class_distribution(train_dist, test_dist, class_id, save_path=None):
        bins = np.arange(0, 27)
        fig, axs = plt.subplots(2, 1, sharex='none', figsize=(14, 12))
        plt.subplots_adjust(hspace=0.25)
        fig.suptitle('Average Output Distribution for Class {} '.format(class_id), fontsize=16)

        axs[0].bar(bins, train_dist, width=0.9, align='center',
                   linewidth=1.75, edgecolor='black', color='olivedrab')

        axs[0].set_xlim(-1, 27)
        axs[0].set_ylim(0.0, 1.04)

        axs[0].set_ylabel('Average Confidence', fontsize=20, weight='bold')

        axs[0].set_xticks(bins)
        axs[0].set_xticklabels(bins, weight='bold')
        axs[0].set_yticks(np.arange(0.0, 1.04, 0.2))
        axs[0].set_yticklabels(np.round(np.arange(0.0, 1.04, 0.2), 2), weight='bold')

        axs[0].grid(linestyle='-.')
        axs[0].set_title('Training Data', fontsize=22, weight='bold')
        axs[0].set_axisbelow(True)
        axs[0].tick_params(axis='y', length=6, width=2, labelsize=18)
        axs[0].tick_params(axis='x', length=6, width=2, labelsize=16)
        axs[0].spines['left'].set_linewidth(2)
        axs[0].spines['bottom'].set_linewidth(2)

        axs[1].bar(bins, test_dist, width=0.9, align='center',
                   linewidth=1.75, edgecolor='black', color='darkcyan')

        axs[1].set_xlim(-1, 27)
        axs[1].set_ylim(0.0, 0.26)

        axs[1].set_ylabel('Average Confidence', fontsize=20, weight='bold')
        axs[1].set_xlabel('Class ID', fontsize=20, weight='bold')

        axs[1].set_xticks(bins)
        axs[1].set_xticklabels(bins, weight='bold')
        axs[1].set_yticks(np.arange(0.0, 0.26, 0.05))
        axs[1].set_yticklabels(np.round(np.arange(0.0, 0.26, 0.05), 2), weight='bold')

        axs[1].grid(linestyle='-.')
        axs[1].set_title('Test Data', fontsize=22, weight='bold')
        axs[1].set_axisbelow(True)
        axs[1].tick_params(axis='y', length=6, width=2, labelsize=18)
        axs[1].tick_params(axis='x', length=6, width=2, labelsize=16)
        axs[1].spines['left'].set_linewidth(2)
        axs[1].spines['bottom'].set_linewidth(2)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close('all')

    def get_output_hist(self, mode='sep'):
        sum_trains = 0
        sum_tests = 0
        run_paths = [p for p in self.model_dir.iterdir() if p.is_dir() and p.parts[-1] != 'results']
        n_runs = len(run_paths)
        for run_path in run_paths:
            self.load_state(path=run_path / 'best_states.pt')
            train_preds, test_preds = self.predict_all(prob_scores=True)
            train_confidences = train_preds.max(axis=1)
            test_confidences = test_preds.max(axis=1)

            if mode == 'sep':
                file_name = run_path.parts[-1] + ' confidence histogram.png'
                save_path = self.savefig_dir / file_name
                self.hist_builder(train_confidences, test_confidences, save_path)
            elif mode == 'mean':
                sum_trains += train_confidences
                sum_tests += test_confidences

        if mode == 'mean':
            ave_train = sum_trains / n_runs
            ave_test = sum_tests / n_runs
            save_path = self.savefig_dir / 'mean_runs confidence histogram.png'
            self.hist_builder(ave_train, ave_test, save_path)

    def get_class_distribution(self, class_id, mode='sep'):
        sum_trains = 0
        sum_tests = 0
        run_paths = [p for p in self.model_dir.iterdir() if p.is_dir() and p.parts[-1] != 'results']
        n_runs = len(run_paths)
        for run_path in run_paths:
            self.load_state(path=run_path / 'best_states.pt')
            train_preds, test_preds = self.predict_all(prob_scores=True, class_id=class_id)
            train_distribution = train_preds.mean(axis=0)
            test_distribution = test_preds.mean(axis=0)

            if mode == 'sep':
                file_name = run_path.parts[-1] + '-class_{} output distribution.png'.format(class_id)
                save_path = self.savefig_dir / file_name
                self.plot_class_distribution(train_distribution, test_distribution, class_id, save_path=save_path)
            elif mode == 'mean':
                sum_trains += train_distribution
                sum_tests += test_distribution

        if mode == 'mean':
            ave_train_dist = sum_trains / n_runs
            ave_test_dist = sum_tests / n_runs
            save_path = self.savefig_dir / 'mean_runs output distribution for class {}.png'.format(class_id)
            self.plot_class_distribution(ave_train_dist, ave_test_dist, class_id, save_path=save_path)


class FeatureExtractor(object):

    def __init__(self, model, model_dir, dataloader, mode='single', run_num=1):
        self.model = model
        self.model_dir = model_dir
        self.dataloader = dataloader
        self.mode = mode
        self.run_num = run_num
        self.batch_features = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_states(self, path=None):
        if path is None:
            path = (self.model_dir / '{}_frame'.format(self.model.n_frames) /
                    'run{}'.format(self.run_num) / 'best_states.pt')

        model_states = torch.load(path)['model']
        self.model.load_state_dict(model_states)
        self.model.to(self.device)
        self.model.eval()

    def register_hooks(self):
        def get_activation(name):
            def hook(module, inputs):
                self.batch_features[name] = inputs[0].detach()
            return hook
        self.model.classifier.register_forward_pre_hook((get_activation('classifier')))

    def extract(self):
        self.load_states()
        self.register_hooks()
        with torch.no_grad():
            all_features = []
            labels = []
            for data in self.dataloader:
                if self.mode == 'single':
                    x = data[0].to(self.device, non_blocking=True)
                    _ = self.model(x)
                    y = data[1].numpy()
                    all_features.append(self.batch_features['classifier'].cpu().numpy())
                    self.batch_features = {}
                    labels.append(y)
                else:
                    x_vision = data[0][0].to(self.device, non_blocking=True)
                    x_tactile = data[0][1].to(self.device, non_blocking=True)
                    _ = self.model(x_vision, x_tactile)
                    y = data[1].cpu().numpy()
                    all_features.append(self.batch_features['classifier'].cpu().numpy())
                    self.batch_features = {}
                    labels.append(y)

            all_features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(labels, axis=0)
            return all_features, labels

    def save_features(self, save_path=None):
        if save_path is None:
            save_path = (self.model_dir / '{}_frame'.format(self.model.n_frames) /
                         'run{}'.format(self.run_num)) / 'features and labels.npz'
        features, labels = self.extract()
        np.savez_compressed(save_path, features=features, labels=labels)


if __name__ == '__main__':
    import pathlib
    from models.mobilenet2 import MobileNetV2

    base_dir = pathlib.Path().resolve()
    data_dir = base_dir / 'data'
    N_FRAMES = 5

    block_depths = 1 * [16] + 2 * [24] + 3 * [32] + 4 * [64] + 3 * [96] + 3 * [160] + 1 * [320]
    strides = 1 * [1] + 1 * [2] + 1 * [1] + 1 * [2] + 2 * [1] + 1 * [2] + 3 * [1] + 3 * [1] + 3 * [1] + 1 * [1]
    expansion_ratios = 1 * [1] + (len(block_depths) - 1) * [6]
    dropouts = (len(block_depths) - 2) * [0.0] + 2 * [0.0]

    model = MobileNetV2(n_frames=N_FRAMES,
                        blocks_depths=block_depths,
                        blocks_strides=strides,
                        exp_ratios=expansion_ratios,
                        blocks_dropouts=dropouts,
                        preconv_depth=32,
                        in_channels=1)

    pretrained_dir = base_dir / 'pretrained_models'
    model_dir = pretrained_dir / 'tactilenet'

    analiser = ConfidenceAnalyser(model, model_dir, data_dir, n_frames=N_FRAMES, batch_size=512)
    analiser.get_output_hist(mode='sep')
    analiser.get_output_hist(mode='mean')
    for c in range(0, 27):
        analiser.get_class_distribution(class_id=c, mode='sep')
        analiser.get_class_distribution(class_id=c, mode='mean')
