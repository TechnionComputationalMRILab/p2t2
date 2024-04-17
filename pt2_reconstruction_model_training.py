import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
from torch.utils.tensorboard import SummaryWriter
import logging

from pt2_reconstruction_model_networks import FC_Model, Convolution_FC_Model
from pt2_reconstruction_model_utils import MSE_wasserstein_combo, wasserstein_distance

class PT2Net_Trainer():

    def __init__(self, args, device) -> None:
        # Set random seed for reproducibility
        self.seed = args.seed
        self.set_seed()
        self.device = device
        self.args = args

        # Get the model based on the specified arguments
        self.model = self.get_model(args, device)

        # Create a logger
        log_name = os.path.join(args.training_path, 'logs.log')
        logging.basicConfig(filename=log_name,
                            filemode='a',
                            format='%(asctime)s - d%(levelname)s:  %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(log_name)


    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_model(self, args, device):
        num_used_echoes = args.n_echoes
        input_channel = num_used_echoes * 2 if args.model_type in ['P2T2-FC'] else num_used_echoes
        num_channels = args.num_channels * 2 if args.model_type in ['P2T2-FC'] else args.num_channels
        print(f'in channels = {input_channel}, num layers = {num_channels}')
        if args.model_type == 'P2T2-conv-FC':
            model = Convolution_FC_Model(
                input_shape=[2, args.n_echoes],
                channels=[args.channel_size] * num_channels,
                output_channel=args.T2_log.num_samples
            ).to(device)

        elif args.model_type in ['P2T2-FC', 'MIML']:
            model = FC_Model(
                input_channel=input_channel,
                channels=[args.channel_size] * num_channels,
                output_channel=args.T2_log.num_samples
            ).to(device)

        else:
            raise NotImplementedError()

        return model

    def get_model_inputs(self, data):
        if self.args.model_type == 'P2T2-FC':
            inputs = torch.cat((data["s_norm"], data["TE"]), dim=-1).to(self.device)
        elif self.args.model_type == 'P2T2-conv-FC':
            inputs = torch.stack((data["s_norm"], data["TE"]), dim=1).to(self.device) 
        else:
            inputs = data["s_norm"].to(self.device)

        # inputs = inputs.type(torch.cuda.FloatTensor)
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.squeeze().to(self.device)

        pT2 = data["p_T2"].squeeze().to(self.device)
        # pT2 = pT2.type(torch.cuda.FloatTensor)
        pT2 = pT2.type(torch.FloatTensor)
        pT2 = pT2.to(self.device)

        return inputs, pT2

    def train_val_step(self, data, mode='train'):
        
        train_args = self.args.Trainer
        if mode=='train':
            self.optimizer.zero_grad()

        inputs, pT2 = self.get_model_inputs(data)
        outputs = self.model(inputs)
        
        if mode=='train':
            loss = self.pT2_loss_function(y_actual=pT2, y_pred=outputs, mse_weight=train_args.mse_weight)
            loss.backward()
            self.optimizer.step() 

            return loss.item()
        
        else:
            loss = self.val_pT2_loss_function(y_actual=pT2, y_pred=outputs, mse_weight=train_args.mse_weight)
            mse = self.val_mse(pT2, outputs)
            wass_dist = self.val_wasserstein(pT2, outputs)

            return loss.item(), mse.item(), wass_dist.item()

    def train_loop(self,train_dl, epoch):
        self.model.train()
        epoch_loss = 0
        step = 0
        for train_data in train_dl:
            step += 1
            batch_loss = self.train_val_step(train_data, mode='train')
            epoch_loss += batch_loss

        epoch_loss /= step
        self.writer.add_scalar(tag='Loss/train', scalar_value=epoch_loss, global_step=epoch + 1)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if self.logger is not None:
            self.logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    def eval_loop(self, val_dl, epoch):
        self.model.eval()
        val_loss = 0
        val_epoch_loss = {'pT2_mse':0., 'pT2_wass': 0.}
        step = 0
        for val_data in val_dl:
            step += 1
            val_batch_loss, val_batch_mse, val_batch_wass = self.train_val_step(val_data, mode='val')
            val_loss += val_batch_loss
            val_epoch_loss['pT2_mse'] += val_batch_mse
            val_epoch_loss['pT2_wass'] += val_batch_wass
            
        val_loss /= step
        self.writer.add_scalar(tag='Loss/val', scalar_value=val_loss, global_step=epoch + 1)
        for key in val_epoch_loss.keys():
            val_epoch_loss[key] /= step
            self.writer.add_scalar(tag=f'Val_Loss/{key}', scalar_value=val_epoch_loss[key], global_step=epoch + 1)
        print([f"{key} val loss = {val_epoch_loss[key]}" for key in val_epoch_loss.keys()])

        
        if val_loss < self.best_val_loss:
            # save new model
            torch.save(self.model.state_dict(), os.path.join(self.args.training_path, 'best_model.pt'))
            self.best_val_loss = val_loss
            print("save new best model")
            self.best_epoch = epoch
        
        print(f"epoch {epoch + 1}, val loss: {val_loss:.4f}")
        print(f"best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch + 1}")
        
        if self.logger is not None:
            self.logger.info(f"epoch {epoch + 1}, val loss: {val_loss:.4f}")
            self.logger.info(f"best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
            self.logger.info([f"val {key} loss = {val_epoch_loss[key]}" for key in val_epoch_loss.keys()])

    def save_checkpoint(self, epoch):
        checkpoints = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'last epoch': epoch
        }
        torch.save(checkpoints, os.path.join(self.args.training_path, 'checkpoints.pt'))

    def train_model(self, train_dl, val_dl):
        self.writer = SummaryWriter(self.args.training_path)
        self.logger.info('start train')

        train_args = self.args.Trainer
        num_epochs = train_args.num_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_args.lr)
        self.pT2_loss_function = MSE_wasserstein_combo(min_T2=self.args.T2_log.start, max_T2=self.args.T2_log.end, num_samples=self.args.T2_log.num_samples)
    
        self.val_pT2_loss_function = MSE_wasserstein_combo(min_T2=self.args.T2_log.start, max_T2=self.args.T2_log.end, num_samples=self.args.T2_log.num_samples)
        self.val_mse = torch.nn.MSELoss()
        self.val_wasserstein= wasserstein_distance(min_T2=self.args.T2_log.start, max_T2=self.args.T2_log.end, num_samples=self.args.T2_log.num_samples)

        val_every = train_args.val_step
        self.best_val_loss = 1e8
        self.best_epoch = 0
        last_epoch = 0


        if os.path.exists(os.path.join(self.args.training_path, 'checkpoints.pt')):
            checkpoints = torch.load(os.path.join(args.training_path, 'checkpoints.pt'))
            self.model.load_state_dict(checkpoints['model'])
            self.optimizer.load_state_dict(checkpoints['optimizer'])
            self.best_val_loss = checkpoints['best_val_loss']
            last_epoch = checkpoints['last epoch']
        
        for epoch in range(last_epoch, num_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{num_epochs}")

            # perform training loop -
            self.train_loop(train_dl, epoch)

            if (epoch + 1) % val_every == 0:
                # perform evaluation loop - 
                with torch.no_grad():
                    self.eval_loop(val_dl, epoch)

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch)

        if self.logger is not None:
            self.logger.info(f"finished train, best val loss: {self.best_val_loss:.6f}, at epoch {self.best_epoch}")
        print(f"finished train, best val loss: {self.best_val_loss:.6f}, at epoch {self.best_epoch}")    
        return None


