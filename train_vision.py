import pathlib
import time
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.vision_dataloader import VisionDataset
from utils import metrics
from utils.extra import EarlyStopping, SmoothedCrossEnropyLoss, CPCE
from models.pretrained_models import PretrainedModel


class ModelTrainer(object):

    def __init__(
            self,
            model,
            checkpoint_dir,
            data_dir,
            n_frames=1,
            n_workers=0,
            do_augment=True,
            use_tensorboard=False):

        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_path = str(checkpoint_dir / 'states.pt')
        self.n_frames = n_frames
        self.n_workers = n_workers
        self.do_augment = do_augment
        self.use_tensorboard = use_tensorboard
        self.data_dir = data_dir

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.use_tensorboard is True:
            tensorboard_logdir = checkpoint_dir / 'logs'
            if tensorboard_logdir.is_dir():
                shutil.rmtree(tensorboard_logdir)
            tensorboard_logdir.mkdir(parents=True, exist_ok=True)
            self.summary_writer = SummaryWriter(tensorboard_logdir)

        self.augmentation_params = {'do_standardization': True,
                                    'do_flip_h': True,
                                    'do_flip_v': False,
                                    'flip_prob': 0.5,
                                    'add_random_noise': False,
                                    'do_erase': False}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None
        self.scheduler = None
        self.best_acc = 0.0
        self.offset_epoch = 0

        self.train_top1_acc = metrics.TopKAccuracy(k=1)
        self.train_top3_acc = metrics.TopKAccuracy(k=3)
        self.test_top1_acc = metrics.TopKAccuracy(k=1)
        self.test_top3_acc = metrics.TopKAccuracy(k=3)
        self.train_loss_recorder = metrics.LossRecorder()
        self.test_loss_recorder = metrics.LossRecorder()
        self.lr_recorder = metrics.BasicRecorder()

    def load_dataset(self):
        train_dataset = VisionDataset(self.data_dir,
                                      split='train',
                                      n_frames=self.n_frames,
                                      image_size=(200, 200),
                                      do_augment=self.do_augment,
                                      **self.augmentation_params)

        test_dataset = VisionDataset(self.data_dir,
                                     split='test',
                                     image_size=(200, 200),
                                     n_frames=self.n_frames,
                                     do_augment=False)

        return train_dataset, test_dataset

    def get_data_loader(self, dataset, batch_size, split, shuffle):
        if split == 'train':
            batch_size = batch_size
            n_workers = self.n_workers
            do_shuffle = shuffle
        else:
            batch_size = 128
            n_workers = self.n_workers
            do_shuffle = shuffle

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=do_shuffle,
                                                 num_workers=n_workers, pin_memory=True,
                                                 drop_last=False)

        return dataloader

    def register_metrics(self):
        self.train_top1_acc.reset_state()
        self.train_top3_acc.reset_state()
        self.test_top1_acc.reset_state()
        self.test_top3_acc.reset_state()
        self.train_loss_recorder.reset_state()
        self.test_loss_recorder.reset_state()

    def _write_to_tensorboard(self, main_tag, tag_scalar_dict, step):
        self.summary_writer.add_scalars(main_tag, tag_scalar_dict, step)

    def update_tensorboard(self, step):
        loss_dict = {'Train/Cross Entropy Loss': self.train_loss_recorder.log[step - 1],
                     'test/Cross Entropy Loss': self.test_loss_recorder.log[step - 1]}
        top1_acc_dict = {'Train/Top1 Accuracy': self.train_top1_acc.log[step - 1],
                         'test/Top1 Accuracy': self.test_top1_acc.log[step - 1]}
        top3_acc_dict = {'Train/Top3 Accuracy': self.train_top3_acc.log[step - 1],
                         'test/Top3 Accuracy': self.test_top3_acc.log[step - 1]}

        self._write_to_tensorboard('Loss', loss_dict, step)
        self._write_to_tensorboard('Top1 Accuracy', top1_acc_dict, step)
        self._write_to_tensorboard('Top3 Accuracy', top3_acc_dict, step)

    def save_checkpoint(self, epoch):
        metrics_state = {'train_loss': self.train_loss_recorder.log,
                         'train_top1': self.train_top1_acc.log,
                         'train_top3': self.train_top3_acc.log,
                         'test_loss': self.test_loss_recorder.log,
                         'test_top1': self.test_top1_acc.log,
                         'test_top3': self.test_top3_acc.log,
                         'learning rate': self.lr_recorder.log}

        # Checkpoint the model after each epoch
        states = {'model': self.model.state_dict(),
                  'optimizer': self.optimizer.state_dict(),
                  'epoch': epoch,
                  'metrics': metrics_state}
        if self.scheduler is not None:
            states['scheduler'] = self.scheduler.state_dict()

        torch.save(states, self.checkpoint_path)
        last_acc = self.test_top1_acc.log[-1]
        if last_acc > self.best_acc:
            self.best_acc = last_acc
            shutil.copy(self.checkpoint_path, self.checkpoint_dir / 'best_states.pt')

    def load_states(self):
        states = torch.load(self.checkpoint_path)
        model_states = states['model']
        # optimizer_state = states['optimizer']
        offset_epoch = states['epoch']
        metrics_state = states['metrics']

        self.model.load_state_dict(model_states)
        # self.optimizer.load_state_dict(optimizer_state)
        #
        # if self.scheduler is not None:
        #     self.scheduler.load_state_dict(states['scheduler'])

        self.train_loss_recorder.log = metrics_state['train_loss']
        self.train_top1_acc.log = metrics_state['train_top1']
        self.train_top3_acc.log = metrics_state['train_top3']
        self.test_loss_recorder.log = metrics_state['test_loss']
        self.test_top1_acc.log = metrics_state['test_top1']
        self.test_top3_acc.log = metrics_state['test_top3']
        self.lr_recorder.log = metrics_state['learning rate']

        self.best_acc = max(self.test_top1_acc.log)
        self.offset_epoch = offset_epoch

    def print_results(self, step):
        print('\nepoch {} train loss = {:.6f}'.format(step, self.train_loss_recorder.log[step - 1]))
        print('epoch {} test loss = {:.6f}'.format(step, self.test_loss_recorder.log[step - 1]))
        print('epoch {} train top1 accuracy = {:.6f}'.format(step, self.train_top1_acc.log[step - 1]))
        print('epoch {} train top3 accuracy = {:.6f}'.format(step, self.train_top3_acc.log[step - 1]))
        print('epoch {} test top1 accuracy = {:.6f}'.format(step, self.test_top1_acc.log[step - 1]))
        print('epoch {} test top3 accuracy = {:.6f}'.format(step, self.test_top3_acc.log[step - 1]))

    @staticmethod
    def get_criterion(beta=0.0):
        train_criterion = CPCE(beta=beta, add_softmax=True)
        test_criterion = SmoothedCrossEnropyLoss(smoothing=0.0, add_softmax=True)
        return train_criterion, test_criterion

    def get_optimizer(self, lr, optim_name='adam', weight_decay=0.0):
        if optim_name == 'adam':
            Optimizer = torch.optim.Adam
            kwargs = {}
        elif optim_name == 'sgd':
            Optimizer = torch.optim.SGD
            kwargs = {'nesterov': True, 'momentum': 0.9}
        else:
            Optimizer = torch.optim.Adam
            kwargs = {}

        optimizer = Optimizer(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=lr, weight_decay=weight_decay, **kwargs)

        return optimizer

    def warmup_stage(self, train_dataloader):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                module.requires_grad_(True)
            else:
                module.requires_grad_(False)

        criterion = torch.nn.CrossEntropyLoss()
        warmup_optimizer = self.get_optimizer(0.01, 'sgd')
        self.model.train()
        top1_acc = metrics.TopKAccuracy(k=1)
        top3_acc = metrics.TopKAccuracy(k=3)
        for i, data_batch in enumerate(train_dataloader):
            X_train = data_batch[0].to(self.device, non_blocking=True)
            y_train = data_batch[1].to(self.device, non_blocking=True)
            warmup_optimizer.zero_grad()
            preds = self.model(X_train)
            warmup_loss = criterion(preds, y_train)
            warmup_loss.backward()
            warmup_optimizer.step()
            top1_acc.update_state(preds, y_train)
            top3_acc.update_state(preds, y_train)

        top1_acc.reset_state()
        top3_acc.reset_state()
        for module in self.model.modules():
            module.requires_grad_(True)

        print('Warmup Stage is Finished...')
        print('Top 1 Accuracy After Warmup = {}'.format(top1_acc.log[0]))
        print('Top 3 Accuracy After Warmup = {}'.format(top3_acc.log[0]))

    def train_step(self, X, y, criterion, l2_reg=0.0, l1_reg=0.0):
        self.optimizer.zero_grad()
        preds = self.model(X)
        train_loss = criterion(preds, y)

        if l1_reg > 0 or l2_reg > 0:
            for params in self.model.parameters():
                if len(params.size()) > 3:
                    params = torch.flatten(params)
                    if l1_reg > 0:
                        norm = torch.sum(torch.abs(params))
                        train_loss += l1_reg * norm
                    if l2_reg > 0:
                        norm = torch.sum(params ** 2)
                        train_loss += l2_reg * norm

        train_loss.backward()
        self.optimizer.step()
        return train_loss, preds

    def test_step(self, X, y, loss_function):
        with torch.no_grad():
            preds = self.model(X)
            test_loss = loss_function(preds, y)
            return test_loss, preds

    def train_on_epoch(self, dataloader, criterion, l1_reg, l2_reg):
        self.model.train()
        for step, data_batch in enumerate(dataloader):
            X_train = data_batch[0].to(self.device, non_blocking=True)
            y_train = data_batch[1].to(self.device, non_blocking=True)

            batch_loss, batch_preds = self.train_step(X_train, y_train, criterion,
                                                      l2_reg=l2_reg, l1_reg=l1_reg)

            self.train_top1_acc.update_state(batch_preds, y_train)
            self.train_top3_acc.update_state(batch_preds, y_train)
            self.train_loss_recorder.update_state(batch_loss)

    def test_on_epoch(self, dataloader, criterion):
        self.model.eval()
        for step, data_batch in enumerate(dataloader):
            X_test, y_test = (data_batch[0].to(self.device, non_blocking=True),
                              data_batch[1].to(self.device, non_blocking=True))

            batch_loss, batch_preds = self.test_step(X_test, y_test, criterion)
            self.test_top1_acc.update_state(batch_preds, y_test)
            self.test_top3_acc.update_state(batch_preds, y_test)
            self.test_loss_recorder.update_state(batch_loss)

    def train_model(self,
                    epochs=10,
                    batch_size=32,
                    learning_rate=0.001,
                    l2_reg=0.0,
                    l1_reg=0.0,
                    warmup=False,
                    weight_decay=0.0,
                    beta=0.0,
                    resume_training=False,
                    use_early_stopping=False,
                    es_patience=10,
                    verbose=True):

        train_dataset, test_dataset = self.load_dataset()
        train_dataloader = self.get_data_loader(train_dataset,
                                                batch_size=batch_size,
                                                split='train',
                                                shuffle=True)
        test_dataloader = self.get_data_loader(test_dataset,
                                               batch_size=batch_size,
                                               split='test',
                                               shuffle=True)

        self.model.to(device=self.device)
        if resume_training is True:
            if pathlib.Path(self.checkpoint_path).is_file():
                self.load_states()

        train_criterion, test_criterion = self.get_criterion(beta=beta)
        early_stopping = EarlyStopping(es_patience)
        if warmup is True:
            self.warmup_stage(train_dataloader)

        self.optimizer = self.get_optimizer(lr=learning_rate, optim_name='sgd', weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        for epoch in range(self.offset_epoch + 1, epochs + self.offset_epoch + 1):
            start_time = time.time()
            self.train_on_epoch(train_dataloader, train_criterion, l1_reg=l1_reg, l2_reg=l2_reg)
            self.test_on_epoch(test_dataloader, test_criterion)
            # Must be called after train and test of an epoch
            self.register_metrics()

            # Adjust Learning Rate after each epoch
            if self.scheduler is not None:
                self.scheduler.step()
                new_lr = self.scheduler.get_last_lr()[0]
                self.lr_recorder.update_state(new_lr)
            else:
                self.lr_recorder.update_state(learning_rate)
            if self.use_tensorboard is True:
                self.update_tensorboard(epoch)
            if verbose is True:
                self.print_results(epoch)

            # Checkpointing
            self.save_checkpoint(epoch)

            # Use Early Stopping
            if use_early_stopping is True:
                STOP_FLAG = early_stopping.stop_early(self.test_top1_acc.log[epoch - 1])
                if STOP_FLAG is True:
                    print('\nEarly Stopping is forced at epoch', epoch)
                    print('Because test accuracy did not improve from {:.5f}'.format(self.best_acc))
                    break

            end_time = time.time()
            epoch_time = end_time - start_time
            print('Elapsed time for epoch {} = {:.2f} seconds'.format(epoch, epoch_time))


if __name__ == '__main__':

    base_dir = pathlib.Path()
    data_dir = base_dir / 'data'

    for N_FRAMES in range(1, 9):
        for run_number in range(1, 2):
            model = PretrainedModel(n_frames=N_FRAMES, model_name='mobilenet', pretrained=True, freeze=False)
            chk_dir = (base_dir / 'pretrained_models' / 'test' / '{}_frame'.format(
                N_FRAMES) / 'run{}'.format(run_number))

            trainer = ModelTrainer(model, chk_dir, data_dir,
                                   n_frames=N_FRAMES, n_workers=0,
                                   do_augment=True, use_tensorboard=True)

            trainer.train_model(epochs=3,
                                batch_size=24,
                                learning_rate=0.01,
                                weight_decay=0.0000,
                                l2_reg=0.0000,
                                l1_reg=0.0,
                                warmup=False,
                                beta=0.0,
                                es_patience=6,
                                resume_training=False,
                                use_early_stopping=True)

            print('Run {} is Complete!'.format(run_number))
        print('All {} Frame Models are Trained!'.format(N_FRAMES))
