import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

os.environ["WANDB_API_KEY"] = "KEY"
os.environ["WANDB_MODE"] = 'offline'
from itertools import combinations

# Change the loss function to MAE
from loss import ClipLoss

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from eegdatasets_leaveone_v3 import EEGDataset
from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from util import wandb_logger

import csv
from torch import Tensor
import math

from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding_inverted
import numpy as np

from diffusers.utils import load_image
from IPython.display import display
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import *
import torch.optim.lr_scheduler as lr_scheduler
import argparse
import datetime
import itertools
from subject_layers.Embed import DataEmbedding
    
class Config:
    def __init__(self):
        self.task_name = 'classification'  # Example task name
        self.seq_len = 250                 # Sequence length
        self.pred_len = 250                # Prediction length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.embed = 'timeF'               # Time encoding method
        self.freq = 'h'                    # Time frequency
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Dimension of the feedforward network
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)


def Bmat2upper(mat):
    triu_indices = torch.triu_indices(7,7,offset=1)
    return mat[:,triu_indices[0],triu_indices[1]]

def Bupper2mat(upper):
    mat = torch.zeros(upper.shape[0],7,7,dtype=upper.dtype, device=upper.device)
    triu_indices = torch.triu_indices(7,7,offset=1)
    mat[:,triu_indices[0],triu_indices[1]] = upper
    mat[:,triu_indices[1],triu_indices[0]] = upper
    return mat

def mat2upper(mat):
    triu_indices = torch.triu_indices(7,7,offset=1)
    return mat[triu_indices[0],triu_indices[1]]

def upper2mat(upper):
    mat = torch.zeros(7,7,dtype=upper.dtype, device=upper.device)
    triu_indices = torch.triu_indices(7,7,offset=1)
    mat[triu_indices[0],triu_indices[1]] = upper
    mat[triu_indices[1],triu_indices[0]] = upper
    return mat

def Bcode2order(cogcode):
    map_arr = torch.tensor([0,5,4,3,2,1],dtype=torch.float,device=cogcode.device)
    y_mapped = map_arr[cogcode]
    return y_mapped

def Border2code(y_mapped):
    reverse_map = torch.tensor([0,5,4,3,2,1],device=y_mapped.device)
    inv_map = torch.zeros(6,dtype=torch.long,device=y_mapped.device)
    inv_map[reverse_map] = torch.arange(6).to(y_mapped.device)
    return inv_map[y_mapped]

def get_symm_gmat(gmat_all):
    B = gmat_all.shape[0] # [B, 21]
    out_all = []
    for i in range(B):
        gmat = gmat_all[i,:]
        result = []
        gmat = upper2mat(gmat)
        result.append(gmat)

        gmat1_1 = gmat.clone()
        gmat1_1 = gmat1_1[[0,1,3,2,4,5,6],:]
        gmat1_1 = gmat1_1[:,[0,1,3,2,4,5,6]]
        result.append(gmat1_1)
        gmat1_2 = gmat.clone()
        gmat1_2 = gmat1_2[[0,1,2,3,4,6,5],:]
        gmat1_2 = gmat1_2[:,[0,1,2,3,4,6,5]]
        result.append(gmat1_2)
        gmat1_3 = gmat1_1.clone()
        gmat1_3 = gmat1_3[[0,1,2,3,4,6,5],:]
        gmat1_3 = gmat1_3[:,[0,1,2,3,4,6,5]]
        result.append(gmat1_3)

        output = []
        for i in range(len(result)):
            output.append(mat2upper(result[i]))
        
        output = torch.stack(output, dim=0) #[4,21]
        out_all.append(output)
    out_all = torch.stack(out_all, dim=0) #[B,4,21]
    return out_all
    
def compute_class_weight(dl, num_classes=6, epsilon=1e-3):
    with torch.no_grad():
        gmat_all = []
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features, gmat) in enumerate(dl):
            gmat_all.append(gmat) #[B, 21]

        gmat_all = torch.stack(gmat_all, dim=0)
        flat = gmat_all.view(-1)
        counts = torch.bincount(flat, minlength=num_classes).float()
        weights = 1.0/(counts+epsilon)
        weights = weights / weights.sum() * num_classes
        return weights


def train_model(eegmodel, dataloader, optimizer, device, text_features_all, img_features_all, save_dir, epoch, batch_size=8, alpha=2, beta=1):
    eegmodel.train()
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    total_loss_penalty = 0
    total_loss_reg = 0
    total_loss_latent = 0

    correct = 0
    total = 0
    features_list = []  # List to store features
    save_features= True
    ridge_lambda = 0.1
    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    image_reconstructed = False  # Flag to track if the image has been reconstructed
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')

    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)
    for batch_idx, (eeg_data, labels, text, text_features, img, img_features, gmat, latent) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        # eeg_data = eeg_data.permute(0, 2, 1)
        img_features = img_features.to(device).float()
        labels = labels.to(device)
        
        gmat_symm = get_symm_gmat(gmat).to(device)
        gmat_symm = Bcode2order(gmat_symm)
        latent = latent.to(device)

        subject_id = 1
        subject_ids = torch.full((len(labels),), subject_id, dtype=torch.long).to(device)

        pred_latent, pred_s = eegmodel(eeg_data[:, :, :250], subject_ids)
        loss_reg = symm_reg_loss(pred_s, gmat_symm)
        loss_latent = F.l1_loss(pred_latent, latent)
        penalty = structure_penalty(pred_s,)

        loss = loss_latent + beta*loss_reg.mean() + alpha*penalty.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_loss_penalty += alpha*penalty.mean().item()
        total_loss_reg += beta*loss_reg.mean().item()
        total_loss_latent += loss_latent.item()


        with torch.no_grad():
            if not image_reconstructed:
                
                z = pred_latent
                x_rec = vae.decode(z).sample
                x_train = vae.decode(img_features).sample
                image_rec = image_processor.postprocess(x_rec, output_type='pil')
                image_train = image_processor.postprocess(x_train, output_type='pil')
                # Use label to create a unique file name
                for i, label in enumerate(labels.tolist()):
                    index_feat = i + batch_idx * batch_size                   
                    save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{index_feat}.png")                    
                    image_rec[i].save(save_path)
                                 
                    save_path2 = os.path.join(epoch_save_dir, f"train_image_{index_feat}.png")                    
                    image_train[i].save(save_path2)
                    image_reconstructed = True                    
        
        del img_features, eeg_data
        
    # torch.cuda.empty_cache()

    average_loss = total_loss / (batch_idx+1)
    average_penalty = total_loss_penalty / (batch_idx+1)
    avg_reg = total_loss_reg / (batch_idx+1)
    avg_latent = total_loss_latent / (batch_idx+1)

    return {'loss':average_loss, 'loss_latent':avg_latent, 'loss_reg':avg_reg, 'penalty':average_penalty}

def evaluate_model(eegmodel, dataloader, device, text_features_all, img_features_all, k, save_dir, epoch, batch_size=8, alpha=2, beta=1):
    eegmodel.eval()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    total_loss_penalty = 0
    total_loss_reg = 0
    total_loss_latent = 0

    mse_loss_fn = nn.MSELoss()
    mae_loss_fn = nn.L1Loss()
    ridge_lambda = 0.1
    accuracy = 0
    top5_acc = 0
    
    epoch_save_dir = os.path.join(save_dir, f'epoch_{epoch}')
    if not os.path.exists(epoch_save_dir):
        os.makedirs(epoch_save_dir)
    fg = True
    with torch.no_grad():
        for batch_idx, (eeg_data, labels, text, text_features, img, img_features,gmat, latent) in enumerate(dataloader):            
            eeg_data = eeg_data.to(device)
            # eeg_data = eeg_data.permute(0, 2, 1)
            img_features = img_features.to(device).float()
            labels = labels.to(device)
            gmat_symm = get_symm_gmat(gmat).to(device)
            gmat_symm = Bcode2order(gmat_symm)

            latent = latent.to(device)

            subject_id = 1
            subject_ids = torch.full((len(labels),), subject_id, dtype=torch.long).to(device)

            pred_latent, pred_s = eegmodel(eeg_data[:, :, :250], subject_ids)
            loss_reg = symm_reg_loss(pred_s, gmat_symm)
            loss_latent = F.l1_loss(pred_latent, latent)
            penalty = structure_penalty(pred_s,)

            loss = loss_latent + beta*loss_reg.mean() + alpha*penalty.mean()

            total_loss += loss.item()
            total_loss_penalty += alpha*penalty.mean().item()
            total_loss_reg += beta*loss_reg.mean().item()
            total_loss_latent += loss_latent.item()
            
            if epoch % 25 ==0:
                
                z = pred_latent
                x_rec = vae.decode(z).sample
                image_rec = image_processor.postprocess(x_rec, output_type='pil')

                # use i to create unique file name
                for i, label in enumerate(labels.tolist()):
                    index_feat = i + batch_idx * batch_size  
                    base_save_path = os.path.join(epoch_save_dir, f"reconstructed_image_{index_feat}.png")
                    save_path = base_save_path
                    # Save the image
                    image_rec[i].save(save_path)

                del img_features, eeg_data, image_rec, x_rec    
                continue
            del img_features, eeg_data
            
    # torch.cuda.empty_cache()
    average_loss = total_loss / (batch_idx + 1)
    average_penalty = total_loss_penalty / (batch_idx+1)
    avg_reg = total_loss_reg / (batch_idx+1)
    avg_latent = total_loss_latent / (batch_idx+1)

    return {'loss':average_loss, 'loss_latent':avg_latent, 'loss_reg':avg_reg, 'penalty':average_penalty}

def main_train_loop(sub, current_time, eeg_model, train_dataloader, test_dataloader, optimizer, device, 
                    text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, config, logger=None):
    # Introduce cosine annealing scheduler
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    logger = wandb_logger(config) if logger else None
    logger.watch(eeg_model,logger)
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch
    
    for epoch in range(config.epochs):

        # Add date-time prefix to save_dir
        train_save_dir = f'vae/{sub}/{current_time}_vae_train_imgs'
        train_info = train_model(eeg_model, train_dataloader, optimizer, device, 
                        text_features_train_all, img_features_train_all, 
                        save_dir=train_save_dir, epoch=epoch, batch_size=config.batch_size)
        
        if (epoch +1) % 50 == 0:                    
            # Get the current time and format it as a string (e.g., '2024-01-17_15-30-00')                  
            if config.insubject==True:       
                os.makedirs(f"./models/contrast/{config.encoder_type}/{sub}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/{config.encoder_type}/{sub}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)            
            else:                
                os.makedirs(f"./models/contrast/across/{config.encoder_type}/{current_time}", exist_ok=True)             
                file_path = f"./models/contrast/across/{config.encoder_type}/{current_time}/{epoch+1}.pth"
                torch.save(eeg_model.state_dict(), file_path)
            print(f"model saved in {file_path}!")

        # Update learning rate
        scheduler.step()
        
        # Evaluate the model
        # test_loss, test_accuracy, top5_acc = evaluate_model(eeg_model, img_model, test_dataloader, device, text_features_test_all, img_features_test_all,k=200)
                # Call evaluate_model function
                        # Get the current date and time, format as "YYYYMMDD_HHMM"

        # Add date-time prefix to save_dir
        test_save_dir = f'vae/{sub}/{current_time}_vae_imgs'
        eval_info = evaluate_model(eeg_model, test_dataloader, device, 
                        text_features_test_all, img_features_test_all, k=200, 
                        save_dir=test_save_dir, epoch=epoch, batch_size = config.batch_size)

        # {'loss':average_loss, 'loss_latent':avg_latent, 'loss_reg':avg_reg, 'penalty':average_penalty}
        # Append results for this epoch
        epoch_results = {
        "epoch": epoch + 1,
        "test_loss": eval_info['loss'],
        "test_loss_latent": eval_info['loss_latent'],
        "test_loss_reg": eval_info['loss_reg'],
        "test_loss_penalty": eval_info['penalty'],
        }

        results.append(epoch_results)
        #
        logger.log({
            "epoch": epoch + 1,
            "train_loss": train_info['loss'],
            "train_loss_latent": train_info['loss_latent'],
            "train_loss_reg": train_info['loss_reg'],
            "train_loss_penalty": train_info['penalty'],
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_info['loss']:.4f}, latent-reg-penalty: {train_info['loss_latent']:.4f}-{train_info['loss_reg']:.4f}-{train_info['penalty']:.4f}" + 
              f"  Test Loss: {eval_info['loss']:.4f}, latent-reg-penalty: {eval_info['loss_latent']:.4f}-{eval_info['loss_reg']:.4f}-{eval_info['penalty']:.4f}")
        torch.cuda.empty_cache()

    logger.finish()
    return results

clip_loss = ClipLoss()
image_processor = VaeImageProcessor()
# path = "stabilityai/stable-diffusion-xl-base-1.0"
# vae = AutoencoderKL.from_pretrained(path, subfolder='vae').to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float, variant="fp16")
if hasattr(pipe, 'vae'):
    for param in pipe.vae.parameters():
        param.requires_grad = False

vae = pipe.vae.to(device)
vae.requires_grad_(False)
vae.eval()

# from train_vae_cogcode2latent import Cog2LatentNet
# cogmodel = Cog2LatentNet()
# cogmodel.eval()
# # checkpoint = torch.load(f'/home/nncc/lyx/project/EEG_Image_decode-main/Generation/models/cogcode/cog2latentNet.pth')
# checkpoint = torch.load(f'/home/nncc/lyx/project/EEG_Image_decode-main/Generation/models/cogcode/cog2latentNet_cnn_full.pth')
# cogmodel.load_state_dict(checkpoint['model'], strict=True)
# cogmodel.to(device)

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='EEG Model Training Script')
    parser.add_argument('--data_path', type=str, default="/home/nncc/lyx/project/EEG_Image_decode-main/myEEG/splitv3/whiten/Preprocessed_data_250Hz", help='Path to data')
    parser.add_argument('--output_dir', type=str, default='./outputs/contrast', help='Directory to save output results')
    parser.add_argument('--project', type=str, default='train_pos_img_text_rep', help='Project name for logging')
    parser.add_argument('--entity', type=str, default="sustech_rethinkingbci", help='WandB entity name')
    parser.add_argument('--name', type=str, default="lr=3e-4_img_pos_pro_eeg", help='Experiment name')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--insubject', default=True, help='Flag to indicate within-subject training')
    parser.add_argument('--encoder_type', type=str, default='ATM_CogMulti',# default='encoder_low_level', 
                        choices=['EEGNetv4_Encoder', 'ATCNet_Encoder', 'EEGConformer_Encoder', 'EEGITNet_Encoder', 'ShallowFBCSPNet_Encoder', 'encoder_low_level'], 
                        help='Encoder type')
    parser.add_argument('--img_encoder', type=str, default='Proj_img', help='Image encoder type')
    parser.add_argument('--logger', default=True, help='Enable logging')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to run on (cpu or gpu)')
    parser.add_argument('--subjects', nargs='+', default=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10', 'sub-11', 'sub-12',
                                                          'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20', 'sub-21', 'sub-22', 'sub-23', 'sub-24'], 
                        help='List of subject IDs (default: sub-01 to sub-24)')
    
    # parser.add_argument('--subjects', nargs='+', default=['sub-01'], help='List of subject IDs')
    
    args = parser.parse_args()

    # Set device based on the argument
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(args.gpu)
    else:
        device = torch.device('cpu')
    
    data_path = args.data_path
    subjects = args.subjects
    # current_time = datetime.datetime.now().strftime("%m-%d_%H-%M")
    current_time = '0707'

    for sub in subjects:
        # Re-initialize the models for each subject
        eeg_model = globals()[args.encoder_type]()
        
        eeg_model.to(device)
        
        optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=args.lr)
        
        if args.insubject:
            train_dataset = EEGDataset(data_path, subjects=[sub], train=True)
            test_dataset = EEGDataset(data_path, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(data_path, exclude_subject=sub, train=True)
            test_dataset = EEGDataset(data_path, exclude_subject=sub, train=False)
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        text_features_train_all = train_dataset.text_features # Nonetype
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features


        results = main_train_loop(sub, current_time, eeg_model, train_loader, test_loader, optimizer, device, 
                                  text_features_train_all, text_features_test_all, img_features_train_all, img_features_test_all, 
                                  config=args, logger=args.logger)

        # Save results to a CSV file
        results_dir = os.path.join(args.output_dir, args.encoder_type, sub, current_time)
        os.makedirs(results_dir, exist_ok=True)
        
        if args.insubject:
            results_file = os.path.join(results_dir, f"{args.encoder_type}_{sub}.csv")
        else:
            results_file = os.path.join(results_dir, f"{args.encoder_type}_cross_exclude_{sub}.csv")

        with open(results_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            print(f'Results saved to {results_file}')

if __name__ == '__main__':
    main()
