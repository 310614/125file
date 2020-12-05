# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 15:39:10 2020

@author: Administrator
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
#import torch.nn.functional as F
import torch.optim as optim
#import matplotlib.pyplot as plt
from LeNet import LeNet
#from evaluation import rie, CC

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
# Set random seed
setup_seed(2020)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Optimization and Evaluation
def getloss(net, mea, img, device):
	mea, img = mea.to(device), img.to(device)
	num_sample = img.shape[0]
	img_p = net(mea)
	h_x = img_p.shape[2]
	w_x = img_p.shape[3]
	#TVloss
	count_h = (h_x - 1) * w_x
	count_w = h_x * (w_x -1)
	h_tv = torch.abs((img_p[:,:,1:,:]-img_p[:,:,:(h_x-1),:])).sum()
	w_tv = torch.abs((img_p[:,:,:,1:]-img_p[:,:,:,:(w_x-1)])).sum()
	TV_loss = 2 * (h_tv / count_h + w_tv / count_w) / num_sample

	#MSE
	MSE_loss = torch.sum((img - img_p) ** 2) / num_sample

	return MSE_loss + 10 * TV_loss



def param_optim(net, mea, img, optimizer, device):
	mea, img = mea.to(device), img.to(device)
	optimizer.zero_grad()
	#net.zero_grad()
	loss = getloss(net, mea, img, device)
	loss.backward()
	optimizer.step()

'''
def net_evaluate(net, loader, l, device):
	total_loss, total_error = 0, 0
	for num, (tm, ti) in enumerate(loader):
		t_l, t_e = t_evaluate(net, tm, ti, device)
		total_loss = total_loss + t_l
		total_error = total_error + t_e
	return total_loss / l, total_error / l


def t_evaluate(net, tm, ti, device):
	tm, ti = tm.to(device), ti.to(device)
	ti_p = net(tm)
	#Loss
	t_loss = torch.sum((ti - ti_p) ** 2)

	#Error
	error = torch.sum(torch.sum(torch.abs(ti - ti_p), 2), 2) / torch.sum(torch.sum(torch.abs(ti), 2), 2) 
	t_error = torch.sum(error)

	return t_loss.detach().cpu().numpy(), t_error.detach().cpu().numpy()
'''
'''
def net_evaluate(net, loader, l, device):
	total_loss = 0.0
	total_rie = 0.0
	total_cc = 0.0
	for step, (x, y) in enumerate(loader):
		x, y = x.to(device), y.to(device)
		y_fake = net(x)
		total_loss += torch.sum((y - y_fake) ** 2).detach().cpu().numpy()
		rie_ = rie(y, y_fake)
		cc_ = CC(y, y_fake)
		total_rie += rie_
		total_cc += cc_
	mean_loss = total_loss / l
	mean_rie = total_rie / l
	mean_cc = total_cc / l
	return mean_loss, mean_rie, mean_cc
'''

def net_evaluate(net, loader, l, device):
	total_loss = 0
	for num, (tm, ti) in enumerate(loader):
		t_l = t_evaluate(net, tm, ti, device)
		total_loss = total_loss + t_l
	return total_loss / l


def t_evaluate(net, tm, ti, device):
	#tm, ti = tm.to(device), ti.to(device)
	#ti_p = net(tm)
	#Loss
	t_loss = getloss(net, tm, ti, device)

	return t_loss.detach().cpu().numpy()

#Data preprocess
class Norm_mea():
	def __init__(self, mea_training):
		self.mean = np.mean(mea_training, axis=0)
		self.std = np.std(mea_training, axis=0)
		self.std[np.where(self.std==0.)] = self.std[np.where(self.std==0.)] + 0.00000000001
        
	def Norm(self, mea):
		tar_mea = (mea - self.mean) / self.std
		return tar_mea

def info2tensor(x):
	x_shape = [x.shape[i] for i in range(len(x.shape))]
	x_shape.insert(1,1)
	tar_x = torch.from_numpy(x).float()
	tar_x = torch.reshape(tar_x, x_shape)
	return tar_x

if __name__ == "__main__":
    #Data Load
    
    data_path = "../data/"
    output_path = "../output/Lenet/"

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    '''
    When training, first use lr = 2e-4  wd = 1e-3 to train 121 epochs.
    Then, set lr = 2e-5, wd = 5e-3 to train 82 epochs.
    '''
    Batch_size = 128
    learning_rate = 2e-4
    #learning_rate = 2e-5
    weight_decay = 1e-3
    num_epochs = 124
        
    sub_path = output_path
    if os.path.exists(sub_path) == False:
        os.mkdir(sub_path)
    
    img_training = np.load(data_path + "training_img.npy")
    img_training = np.concatenate((img_training, img_training, img_training, img_training), axis = 0)
    img_training = np.abs(img_training)
    mea_training = np.load(data_path + "training_mea.npy")
    img_validating = np.load(data_path + "validating_img.npy")
    img_validating = np.concatenate((img_validating, img_validating, img_validating, img_validating), axis = 0)
    img_validating = np.abs(img_validating)
    mea_validating = np.load(data_path + "validating_mea.npy")
    
    print("Data loading finished!")
    
    mea_whiten = Norm_mea(mea_training)
    mea_training = mea_whiten.Norm(mea_training)
    mea_validating = mea_whiten.Norm(mea_validating)
    
    mea_training_T = info2tensor(mea_training)
    mea_validating_T = info2tensor(mea_validating)
    
    img_training_T = info2tensor(img_training)
    img_validating_T = info2tensor(img_validating)
    
    print("Data norm and tensored finished!")

    #initialization
    net = LeNet()
    #net.load_state_dict(torch.load(output_path + "best_1.pth", map_location=torch.device('cpu')))
    
    optimizer = optim.Adam(net.parameters(), lr = learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    
    
    #batch
    training_set = Data.TensorDataset(mea_training_T, img_training_T)
    validating_set = Data.TensorDataset(mea_validating_T, img_validating_T)
    
    training_load = Data.DataLoader(dataset=training_set, batch_size=Batch_size,
    								shuffle=True, num_workers=2)

    validating_load = Data.DataLoader(dataset=validating_set, batch_size=Batch_size,
    								shuffle=False, num_workers=2)
    
    print("Dataloader prepration finished!")
    
    #GPU start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
   
    f = open(sub_path + "log.txt", 'w')
    
    validating_loss_mim = [1000] * 10
    #validating_loss_mim = [19.8, 19.23, 19.6, 19.48, 19.77, 19.35, 19.56, 19.61, 19.62, 19.85]
    
    
    for epoch in range(num_epochs):
    	epoch = epoch
    	print("Epoch:", epoch+1, "Training")
    	# Evaluate step
    	if (epoch % 3 == 0 and epoch > 0):
    		net.eval()

    		training_loss = net_evaluate(net, training_load, mea_training.shape[0], device)
    		validating_loss = net_evaluate(net, validating_load, mea_validating.shape[0], device)		

    		print("Training Loss:", training_loss)
    		print("Validating Loss:", validating_loss)
    		f.write('Epoch: ' + str(epoch+1) + "  Training Loss = " + str(training_loss) + '\n')
    		f.write('Epoch: ' + str(epoch+1) + "  Validating Loss = " + str(validating_loss) + '\n')
    		f.write('\n')
    
    
    		#Save step

    		scheduler.step(training_loss)
    		if validating_loss < np.max(validating_loss_mim):
    			save_index = np.argmax(validating_loss_mim)
    			torch.save(net.state_dict(), sub_path + "best_" + str(save_index) + ".pth")
    			validating_loss_mim[save_index] = validating_loss
    
    
    
    	net.train()
    
    	for num, (tm, ti) in enumerate(training_load):
    		param_optim(net, tm, ti, optimizer, device)
            
    f.write('\n')
    f.write('\n')
    
    '''
    learning_rate = 1e-4
    weight_decay = 1e-2
    num_epochs = 22  
    
    optimizer1 = optim.Adam(net.parameters(), lr = learning_rate, weight_decay=weight_decay)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.2, patience=3)
            
    for epoch in range(num_epochs):
    	epoch = epoch + 121
    	print("Epoch:", epoch+1, "Training")
    	# Evaluate step
    	if (epoch % 3 == 0 and epoch > 0):
    		net.eval()

    		training_loss = net_evaluate(net, training_load, mea_training.shape[0], device)
    		validating_loss = net_evaluate(net, validating_load, mea_validating.shape[0], device)		

    		print("Training Loss:", training_loss)
    		print("Validating Loss:", validating_loss)
    		f.write('Epoch: ' + str(epoch+1) + "  Training Loss = " + str(training_loss) + '\n')
    		f.write('Epoch: ' + str(epoch+1) + "  Validating Loss = " + str(validating_loss) + '\n')
    		f.write('\n')
    
    
    		#Save step

    		scheduler1.step(training_loss)
    		if validating_loss < np.max(validating_loss_mim):
    			save_index = np.argmax(validating_loss_mim)
    			torch.save(net.state_dict(), sub_path + "best1_" + str(save_index) + ".pth")
    			validating_loss_mim[save_index] = validating_loss
    
    
    
    	net.train()
    
    	for num, (tm, ti) in enumerate(training_load):
    		param_optim(net, tm, ti, optimizer1, device)
    '''