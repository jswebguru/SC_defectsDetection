"""
@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
"""

#System Modules
from enum import Enum
import datetime
import os
import time

# Deep Learning Modules
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from sklearn.metrics import multilabel_confusion_matrix
import torch.nn.functional as F
from torchvision import models

# User Defined Modules
from configs.serde import *
from utils.stopping import EarlyStoppingCallback
import pdb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
epsilon = 1e-15



class Training:
    '''
    This class represents training (including validation) process.
    '''
    def __init__(self, cfg_path, stopping_patience, num_epochs=10, RESUME=False, torch_seed=None):
        '''
        :cfg_path (string): path of the experiment config file
        :torch_seed (int): Seed used for random generators in PyTorch functions
        :stopping_patience: Number of epochs that we had no improvement in the loss
         and then we should stop training.
        '''
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.RESUME = RESUME
        self.best_loss = float('inf')
        self.best_F1 = 0

        if RESUME == False:
            self.model_info = self.params['Network']
            self.model_info['seed'] = torch_seed or self.model_info['seed']
            self.epoch = 0
            self.num_epochs = num_epochs

            if 'trained_time' in self.model_info:
                self.raise_training_complete_exception()

            self.setup_cuda()
            self.writer = SummaryWriter(log_dir=os.path.join(self.params['tb_logs_path']))

    def setup_cuda(self, cuda_device_id=0):
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
            torch.cuda.manual_seed_all(self.model_info['seed'])
            torch.manual_seed(self.model_info['seed'])
        else:
            self.device = torch.device('cpu')

    def setup_model(self, model, optimiser, optimiser_params, loss_function, pos_weight=None):
        '''
        :param model: an object of our network
        :param optimiser: an object of our optimizer, e.g. torch.optim.SGD
        :param optimiser_params: is a dictionary containing parameters for the optimiser, e.g. {'lr':7e-3}
        '''
        # number of parameters of the model
        print(f'\nThe model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters!\n')

        # Tensor Board Graph
        self.add_tensorboard_graph(model)

        self.model = model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)
        self.loss_function = loss_function(pos_weight=pos_weight.to(self.device))

        if 'retrain' in self.model_info and self.model_info['retrain'] == True:
            self.load_pretrained_model()

        # Saves the model, optimiser,loss function name for writing to config file
        # self.model_info['model_name'] = model.__name__
        self.model_info['optimiser'] = optimiser.__name__
        self.model_info['loss_function'] = loss_function.__name__
        self.model_info['optimiser_params'] = optimiser_params
        self.params['Network'] = self.model_info
        write_config(self.params, self.cfg_path, sort_keys=True)


    def load_checkpoint(self, model, optimiser, optimiser_params, loss_function, pos_weight=None):

        checkpoint = torch.load(self.params['network_output_path'] + '/' + self.params['checkpoint_name'])
        self.device = None
        self.model_info = checkpoint['model_info']
        self.setup_cuda()
        self.model = model.to(self.device)
        self.optimiser = optimiser(self.model.parameters(), **optimiser_params)
        self.loss_function = loss_function(pos_weight=pos_weight.to(self.device))

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.num_epochs = checkpoint['num_epoch']
        self.loss_function = checkpoint['loss']
        self.best_loss = checkpoint['best_loss']
        self.best_F1 = checkpoint['best_F1']
        self.writer = SummaryWriter(log_dir=os.path.join(self.params['tb_logs_path']), purge_step=self.epoch + 1)

    def add_tensorboard_graph(self, model):
        '''Creates a tensor board graph for network visualisation'''
        dummy_input = torch.rand(1, 3, 300, 300)  # To show tensor sizes in graph
        self.writer.add_graph(model, dummy_input)


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def execute_training(self, train_loader, valid_loader=None, batch_size=1):
        '''
        Executes training by running training and validation at each epoch
        '''
        total_start_time = time.time()

        # reads param file again to include changes if any
        self.params = read_config(self.cfg_path)

        if self.RESUME == False:
            # Checks if already trained
            if 'trained_time' in self.model_info:
                self.raise_training_complete_exception

            # CODE FOR CONFIG FILE TO RECORD MODEL PARAMETERS
            self.model_info = self.params['Network']
            self.model_info['num_epochs'] = self.num_epochs or self.model_info['num_epochs']

        print('Starting time:' + str(datetime.datetime.now()) +'\n')

        for epoch in range(self.num_epochs - self.epoch):
            self.epoch += 1
            start_time = time.time()

            print('Training (intermediate metrics):')
            train_loss, train_acc, train_F1 = self.train_epoch(train_loader, batch_size)

            if valid_loader:
                print('\nValidation (intermediate metrics):')
                valid_loss, valid_acc, valid_F1 = self.valid_epoch(valid_loader, batch_size)

            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            total_mins, total_secs = self.epoch_time(total_start_time, end_time)

            # Writes to the tensorboard after number of steps specified.
            if valid_loader:
                self.calculate_tb_stats(train_loss, train_F1, valid_loss, valid_F1)
            else:
                self.calculate_tb_stats(train_loss, train_F1)

            # Saves information about training to config file
            self.model_info['num_steps'] = self.epoch
            self.model_info['trained_time'] = "{:%B %d, %Y, %H:%M:%S}".format(datetime.datetime.now())
            self.params['Network'] = self.model_info
            write_config(self.params, self.cfg_path, sort_keys=True)

            '''Saving the model'''
            if valid_loader:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    torch.save(self.model.state_dict(), self.params['network_output_path'] + '/' +
                               self.params['trained_model_name'])
            else:
                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    torch.save(self.model.state_dict(), self.params['network_output_path'] + '/' +
                               self.params['trained_model_name'])
            # Saving based on the F1 score
            if valid_F1 > self.best_F1:
                self.best_F1 = valid_F1
                torch.save(self.model.state_dict(), self.params['network_output_path'] + '/F1based_' +
                           self.params['trained_model_name'])

            # Saving every 20 epochs
            if (self.epoch) % self.params['network_save_freq'] == 0:
                torch.save(self.model.state_dict(), self.params['network_output_path'] + '/' +
                           'epoch{}_'.format(self.epoch) + self.params['trained_model_name'])

            # Save a checkpoint
            torch.save({'epoch': self.epoch, 'best_F1': self.best_F1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimiser.state_dict(),
                'loss': self.loss_function, 'num_epoch': self.num_epochs,
                'model_info': self.model_info, 'best_loss': self.best_loss},
                self.params['network_output_path'] + '/' + self.params['checkpoint_name'])

            # Print accuracy, F1, and loss after each epoch
            print('\n---------------------------------------------------------------')
            print(f'Epoch: {self.epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | '
                  f'Total Time so far: {total_mins}m {total_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F1: {train_F1:.3f}')
            if valid_loader:
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% |  Val. F1: {valid_F1:.3f}')
            print('---------------------------------------------------------------\n')


            #TODO: earlystoping goes here!
            # best_valid_loss = self.stopper.step(current_loss=valid_loss, best_loss=best_valid_loss)
            # stopping_flag = self.stopper.should_stop()
            # if stopping_flag == True:
            #     break


    def train_epoch(self, train_loader, batch_size):
        '''
        Train using one single iteration of all messages (epoch) in dataset
        '''
        print("Epoch [{}/{}]".format(self.epoch, self.model_info['num_epochs']))
        self.model.train()
        previous_idx = 0

        # initializing the loss list
        batch_loss = 0
        batch_count = 0

        # initializing the caches
        logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(train_loader) * batch_size, 2)))
        logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
        labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

        for idx, (image, label) in enumerate(train_loader):
            image = image.to(self.device)
            label = label.to(self.device)

            #Forward pass.
            self.optimiser.zero_grad()

            with torch.set_grad_enabled(True):

                output = self.model(image)
                label = label.float()
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                for i, batch in enumerate(output_sigmoided):
                    logits_with_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(output):
                    logits_no_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_cache[idx * batch_size + i] = batch

                # Loss
                loss = self.loss_function(output, label)
                batch_loss += loss.item()
                batch_count += 1

                #Backward and optimize
                loss.backward()
                self.optimiser.step()

                # Prints loss statistics after number of steps specified.
                if (idx + 1)%self.params['display_stats_freq'] == 0:
                    print('Epoch {:02} | Batch {:03}-{:03} | Train loss: {:.3f}'.
                          format(self.epoch, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    batch_loss = 0
                    batch_count = 0

        '''Metrics calculation (macro) over the whole set'''
        crack_confusion, inactive_confusion = multilabel_confusion_matrix(labels_cache.cpu(), logits_with_sigmoid_cache.cpu())
        # Crack class
        TN = crack_confusion[0, 0]
        FP = crack_confusion[0, 1]
        FN = crack_confusion[1, 0]
        TP = crack_confusion[1, 1]
        accuracy_Crack = (TP + TN) / (TP + TN + FP + FN + epsilon)
        F1_Crack = 2 * TP / (2 * TP + FN + FP + epsilon)
        # Inactive class
        TN_inactive = inactive_confusion[0, 0]
        FP_inactive = inactive_confusion[0, 1]
        FN_inactive = inactive_confusion[1, 0]
        TP_inactive = inactive_confusion[1, 1]
        accuracy_inactive = (TP_inactive + TN_inactive) / (TP_inactive + TN_inactive + FP_inactive + FN_inactive + epsilon)
        F1_inactive = 2 * TP_inactive / (2 * TP_inactive + FN_inactive + FP_inactive + epsilon)
        # Macro averaging
        epoch_accuracy = (accuracy_Crack + accuracy_inactive) / 2
        epoch_f1_score = (F1_Crack + F1_inactive) / 2
        loss = self.loss_function(logits_no_sigmoid_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        return epoch_loss, epoch_accuracy, epoch_f1_score



    def valid_epoch(self, valid_loader, batch_size):
        '''Test (validation) model after an epoch and calculate loss on test dataset'''
        print("Epoch [{}/{}]".format(self.epoch, self.model_info['num_epochs']))
        self.model.eval()
        previous_idx = 0

        with torch.no_grad():
            # initializing the loss list
            batch_loss = 0
            batch_count = 0

            # initializing the caches
            logits_with_sigmoid_cache = torch.from_numpy(np.zeros((len(valid_loader) * batch_size, 2)))
            logits_no_sigmoid_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))
            labels_cache = torch.from_numpy(np.zeros_like(logits_with_sigmoid_cache))

            for idx, (image, label) in enumerate(valid_loader):
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                label = label.float()
                output_sigmoided = F.sigmoid(output)
                output_sigmoided = (output_sigmoided > 0.5).float()

                # saving the logits and labels of this batch
                for i, batch in enumerate(output_sigmoided):
                    logits_with_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(output):
                    logits_no_sigmoid_cache[idx * batch_size + i] = batch
                for i, batch in enumerate(label):
                    labels_cache[idx * batch_size + i] = batch

                # Loss
                loss = self.loss_function(output, label)
                batch_loss += loss.item()
                batch_count += 1

                # Prints loss statistics after number of steps specified.
                if (idx + 1)%self.params['display_stats_freq'] == 0:
                    print('Epoch {:02} | Batch {:03}-{:03} | Val. loss: {:.3f}'.
                          format(self.epoch, previous_idx, idx, batch_loss / batch_count))
                    previous_idx = idx + 1
                    batch_loss = 0
                    batch_count = 0

        '''Metrics calculation (macro) over the whole set'''
        crack_confusion, inactive_confusion = multilabel_confusion_matrix(labels_cache.cpu(), logits_with_sigmoid_cache.cpu())
        # Crack class
        TN = crack_confusion[0, 0]
        FP = crack_confusion[0, 1]
        FN = crack_confusion[1, 0]
        TP = crack_confusion[1, 1]
        accuracy_Crack = (TP + TN) / (TP + TN + FP + FN + epsilon)
        F1_Crack = 2 * TP / (2 * TP + FN + FP + epsilon)
        # Inactive class
        TN_inactive = inactive_confusion[0, 0]
        FP_inactive = inactive_confusion[0, 1]
        FN_inactive = inactive_confusion[1, 0]
        TP_inactive = inactive_confusion[1, 1]
        accuracy_inactive = (TP_inactive + TN_inactive) / (TP_inactive + TN_inactive + FP_inactive + FN_inactive + epsilon)
        F1_inactive = 2 * TP_inactive / (2 * TP_inactive + FN_inactive + FP_inactive + epsilon)
        # Macro averaging
        epoch_accuracy = (accuracy_Crack + accuracy_inactive) / 2
        epoch_f1_score = (F1_Crack + F1_inactive) / 2
        loss = self.loss_function(logits_no_sigmoid_cache.to(self.device), labels_cache.to(self.device))
        epoch_loss = loss.item()

        self.model.train()
        return epoch_loss, epoch_accuracy, epoch_f1_score


    def calculate_tb_stats(self, train_loss, train_F1, valid_loss=None, valid_F1=None):
        '''Adds the statistics of metrics to the tensorboard'''

        # Adds the metrics to TensorBoard
        self.writer.add_scalar('Training' + '_Loss', train_loss, self.epoch)
        self.writer.add_scalar('Training' + '_F1', train_F1, self.epoch)
        if valid_loss:
            self.writer.add_scalar('Validation' + '_Loss', valid_loss, self.epoch)
            self.writer.add_scalar('Validation' + '_F1', valid_F1, self.epoch)


    def raise_training_complete_exception(self):
        raise Exception("Model has already been trained on {}. \n"
                        "1.To use this model as pre trained model and train again\n "
                        "create new experiment using create_retrain_experiment function.\n\n"
                        "2.To start fresh with same experiment name, delete the experiment  \n"
                        "using delete_experiment function and create experiment "
                        "               again.".format(self.model_info['trained_time']))


def load_pretrained_model():
    # Load a pre-trained model from config file
    # self.model.load_state_dict(torch.load(self.model_info['pretrain_model_path']))

    # Load a pre-trained model from Torchvision
    MODEL = models.resnet34(pretrained=True)
    # for param in MODEL.parameters():
    #     param.requires_grad = False
    MODEL.fc = nn.Sequential(
        nn.Linear(512, 2))
    for param in MODEL.fc.parameters():
        param.requires_grad = True

    return MODEL



class Prediction:
    '''
    This class represents prediction (testing) process similar to the Training class.
    '''
    def __init__(self, cfg_path):
        '''
        :cfg_path (string): path of the experiment config file
        '''
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


    def setup_model(self, model, model_file_name=None):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']

        # in case of pretrained model
        model = load_pretrained_model()

        self.model_p = model.to(self.device)

        # Loads model from model_file_name and default network_output_path
        # self.model_p.load_state_dict(torch.load(self.params['network_output_path'] + "/" + model_file_name))
        self.model_p.load_state_dict(torch.load(self.params['network_output_path'] + "/epoch60_" + model_file_name))


    def save_onnx(self, fn):
        m = self.model_p.cpu()
        m.eval()
        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self.model_p(x)
        torch.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})



class Mode(Enum):
    '''
    Class Enumerating the 3 modes of operation of the network.
    This is used while loading datasets
    '''

    TRAIN = 0
    VALID = 1
    TEST = 2
