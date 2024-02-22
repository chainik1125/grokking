import sys
print(sys.version)



import subprocess
#Check if cuda installed
# Function to execute a shell command and return the output
def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, text=True)
        return output
    except subprocess.CalledProcessError as e:
        return e.output

# Running nvcc --version
nvcc_version = run_command("nvcc --version")
print("NVCC Version:\n", nvcc_version)

# Running nvidia-smi
nvidia_smi = run_command("nvidia-smi")
print("NVIDIA SMI:\n", nvidia_smi)


##################################
# standard imports
import os
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer
import time
import pickle
import tqdm
#from tqdm.auto import tqdm # for loop progress bar
import itertools as it
import numpy as np
import random
from matplotlib import pyplot as plt
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# machine learning imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader



##################Data params

#Data parameters

encoded=True #Whether you're going to load encoded or unencoded tensors
encoded_averaging='mean' #Whether you're going to feed the 30 snapshots or the average

# training_images=2000
# test_images=2000
L=12
#label_dic={2:'ergodic',0:'Anderson Localized',1:'MBL'}
label_dic={0:'Anderson Localized',1:'MBL',2:'Ergodic'}
rev_label_dic=dict((v, k) for k, v in label_dic.items())
erg_data=False
MBL_data=True #Do you draw from the MBL set?
AL_data=True #Do you draw from the AL set?
add=True
dtype=torch.float32#Doing this for now because I think you need to change the model defaults to use 64 in the default optimizers.
train_size=1850
test_size=100
batch_size=train_size
erg_W_lim=1
mbl_W_lim=0
al_W_lim=0

#######################Loading
if encoded:
    #Load MBL dataset
    tensor_to_open='../Data/encoded_tensor_L1_z_1000.p'
    tensor_to_open2='../Data/encoded_tensor_L1_x_1000.p'
    tensor_to_open3='../Data/encoded_tensor_L1_y_1000.p'
    folder=os.getcwd()

    # for file_path in os.listdir(folder):
    #     if tensor_to_open in file_path:
    with open(tensor_to_open, 'rb') as f:
        # print(file_path)
        tensors_mbl_z = pickle.load(f)
    with open(tensor_to_open2, 'rb') as f:
        # print(file_path)
        tensors_mbl_x=pickle.load(f)
    with open(tensor_to_open3, 'rb') as f:
        # print(file_path)
        tensors_mbl_y=pickle.load(f)
    #Load AL dataset
    tensor_to_open='../Data/encoded_tensor_A1_z_1000.p'
    tensor_to_open2='../Data/encoded_tensor_A1_x_1000.p'
    tensor_to_open3='../Data/encoded_tensor_A1_y_1000.p'
    folder=os.getcwd()

    # for file_path in os.listdir(folder):
    #     if tensor_to_open in file_path:
    with open(tensor_to_open, 'rb') as f:
        # print(file_path)
        tensors_al_z = pickle.load(f)
    with open(tensor_to_open2, 'rb') as f:
        # print(file_path)
        tensors_al_x = pickle.load(f)
    with open(tensor_to_open3, 'rb') as f:
        # print(file_path)
        tensors_al_y = pickle.load(f)
else:
    tensor_to_open='non_encoded_tensor_MBL_1_3000'
    folder=os.getcwd()

    # for file_path in os.listdir(folder):
    #     if tensor_to_open in file_path:
    with open(tensor_to_open, 'rb') as f:
        # print(file_path)
        tensors_mbl = pickle.load(f)

    tensor_to_open='non_encoded_tensor_AL_1_3000'
    folder=os.getcwd()

    # for file_path in os.listdir(folder):
    #     if tensor_to_open in file_path:
    with open(tensor_to_open, 'rb') as f:
        # print(file_path)
        tensors_al = pickle.load(f)



def average_converter(time_snapshot,L):
    int_averaged=time_snapshot.int32


    return None

def form_array_grid(tensor,encoded,L):
    if encoded:
        tensor_array=tensor.cpu().detach().numpy()
        #new_tensor=np.zeros((tensor_array.shape[0],L),dtype=float)
        #first un_encode binary string
        array_list=[]
        print(tensor_array.shape)
        for snap in tensor_array:
            new_array=np.zeros(L,dtype=float)
            for average in snap:
                average=int(average)
                bin_string=np.binary_repr(average,width=L)
                array=np.array([int(i) for i in bin_string])
                new_array+=array
            new_array=new_array/len(snap)
            array_list.append(new_array)
        print(len(array_list))
        new_tensor=np.array(array_list)
    elif encoded==False:
        tensor_array=tensor.cpu().detach().numpy()
        array_list=[]
        for snap in tensor_array:
            new_array=sum(snap)
            new_array/len(snap)
            array_list.append(new_array)
        new_tensor=np.array(array_list)


    return new_tensor

def plot_heatmap(plot_matrix,label,tensor_index):
    fig = go.Figure(data=go.Heatmap(
                    z=plot_matrix,
                    x=[f'Qubit {i}' for i in range(L)],  # Optional: X-axis labels
                    y=[f'S {i}' for i in range(plot_matrix.shape[0])])) # Optional: Y-axis labels

    # Update layout
    fig.update_layout(
        title=f'Averaged occupation, {label_dic[label]} Phase, Tensor {tensor_index}',
        xaxis_nticks=12,
        yaxis_nticks=20)

    # Show plot
    #fig.write_image(f'{label_dic[label]}.png')

    #Convert data to required format
def binary(x, bits):
    # Creating a mask in reverse order
    mask = 2**torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def convert_average(input_tensor,encoded_averaging,L):
    if encoded_averaging=='mean':
        int_tensor=input_tensor.int()
        dec=binary(x=int_tensor,bits=L)
        avg_tensor=torch.mean(dec.to(dtype),1)#Needs to be float for averaging...

    elif encoded_averaging=='var':
        int_tensor=input_tensor.int()
        dec=binary(x=int_tensor,bits=L)
        avg_tensor=torch.var(dec.to(dtype),1)#Needs to be float for averaging...

    else:
        avg_tensor=torch.mean(input_tensor,1)

    return avg_tensor

# enc=convert_average(X_test[0],True,L)
# print(enc.shape)



#if encoded and encoded_qubit_average:




print(f"Is the raw data encoded Tensors? {encoded}")
# I need to:
#1. Construct phase labels - pretty straighforward, just 0,1 or 2.
#2. Shuffle the data
#3.
if encoded==True:
    data=[]
    #0 - ergodic, 1, AL, 2, MBL
    if MBL_data:
        if erg_data:
            for i in range(len(tensors_mbl_z[:erg_W_lim])):
                print(i)
                for tensor in range(len(tensors_mbl[i])):
                    if encoded_averaging==False:
                        temp_triple=(tensor,rev_label_dic['Ergodic'],i)#ergodic
                    else:
                        temp_triple=(convert_average(tensor,encoded_averaging,L),rev_label_dic['Ergodic'],i)
                    data.append(temp_triple)

        for i in range(mbl_W_lim,1):#len(tensors_mbl_z):
            print(f'mbl index {i}')
            for tensor in range(len(tensors_mbl_z[i])):
                if encoded_averaging==False:
                    temp_triple=(tensor,rev_label_dic['MBL'],i)
                else:
                    if add:
                        tensor_x=convert_average(tensors_mbl_x[i][tensor],encoded_averaging,L)
                        tensor_y=convert_average(tensors_mbl_y[i][tensor],encoded_averaging,L)
                        tensor_z=convert_average(tensors_mbl_z[i][tensor],encoded_averaging,L)
                        tensor_xyz=torch.stack((tensor_z,tensor_x,tensor_y),dim=2)
                        #tensor_zy=torch.stack((tensor_z,tensor_y),dim=2)
                        #temp_triple=(tensor_zy,rev_label_dic['MBL'],i)#MBL
                        temp_triple=(tensor_xyz,rev_label_dic['MBL'],i)#MBL
                data.append(temp_triple)
    if AL_data:
        for i in range(al_W_lim,1):##Because I think the W=0 phase is ergodic! #len(tensors_al_z)
            print(f'al index {i}')
            for tensor in range(len(tensors_al_z[i])):
                if encoded_averaging==False:
                    temp_triple=(tensor,rev_label_dic['Anderson Localized'],i)#AL
                else:
                    if add:
                        tensor_x=convert_average(tensors_al_x[i][tensor],encoded_averaging,L)
                        tensor_y=convert_average(tensors_al_y[i][tensor],encoded_averaging,L)
                        tensor_z=convert_average(tensors_al_z[i][tensor],encoded_averaging,L)
                        tensor_xyz=torch.stack((tensor_z,tensor_x,tensor_y),dim=2)
                        #temp_triple=(tensor_zy,rev_label_dic['Anderson Localized'],i)
                        temp_triple=(tensor_xyz,rev_label_dic['Anderson Localized'],i)
                data.append(temp_triple)


    print(len(data))
    print(data[0][0].shape)
    random.shuffle(data)
    print(data[0][2])
    print(f"Number of snapshots in data: {len(data)}")
    print(f"Number of Ergodic snapshots in data: {len([1 for i in data if i[1]==rev_label_dic['Ergodic']])}")
    print(f"Number of Anderson Localized snapshots in data: {len([1 for i in data if i[1]==rev_label_dic['Anderson Localized']])}")
    print(f"Number of Many-Body Localized snapshots in data: {len([1 for i in data if i[1]==rev_label_dic['MBL']])}")

    # split data into input (array) and labels (phase and temp)
    # for now ignore temp labels
    my_X = torch.stack([i[0] for i in data], dim=0).to(dtype) # transform to torch tensor of FLOATS
    my_y = torch.Tensor([i[1] for i in data]).to(torch.long) # transform to torch tensor of INTEGERS
    my_y_W = torch.Tensor([i[2] for i in data]).to(dtype)
    my_X.float()
    print(my_X.dtype, my_y.dtype)
    print(my_y[:10])
    print("Created (encoded) AL/MBL/Ergodic Dataset")
# elif encoded==False:
#     data=full_tensors

# manually do split between training and testing data (only necessary if ising data)
#Do I need t =o put in something so that there's no overlap between test and train or is that taken care of automatically by the DataLoader? Taken care of by the indexing actually I think
#train_size, test_size, batch_size = 1000, 100, 1000 - defined in the general training parameters
a, b = train_size, test_size
train_data = TensorDataset(my_X[b:a+b], my_y[b:a+b]) # Choose training data of specified size
test_data = TensorDataset(my_X[:b], my_y[:b]) # test

print(f'test size: {test_size}')
# load data in batches for reduced memory usage in learning
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)
for b, (X_train, y_train) in enumerate(train_loader):
    print("batch:", b)
    print("input tensors shape (data): ", X_train.shape)
    print("output tensors shape (labels): ", y_train.shape)

#Module for scrambling
scramble_snapshot=False#For Ising
for b, (X_test, y_test) in enumerate(test_loader):
    if scramble_snapshot:
        X_test_a=np.array(X_test)
        X_test_perc=np.zeros((test_size,16,16))
        for t in range(test_size):
            preshuff=X_test_a[t,:,:].flatten()
            np.random.shuffle(preshuff)
            X_test_perc[t,:,:]=np.reshape(preshuff,(16,16))
        X_test=torch.Tensor(X_test_perc).to(dtype)
print("batch:", b)
print("input tensors shape (data): ", X_test.shape)

    #print("output tensors shape (labels): ", y_test.shape)


print(X_test.dtype)
#Statistics of data:
print(f"Anderson, MBL, Ergodic in train: {[sum(x.item()==i for x in y_train) for i in range(3)]}")
print(f"Anderson, MBL, Ergodic in test: {[sum(x.item()==i for x in y_test) for i in range(3)]}")

# print(y_train[:10])
# print(y_test[:10])


train_erg_indices=[i for i in range(len(y_train)) if y_train[i]==rev_label_dic['Ergodic']]
train_mbl_indices=[i for i in range(len(y_train)) if y_train[i]==rev_label_dic['MBL']]
train_al_indices=[i for i in range(len(y_train)) if y_train[i]==rev_label_dic['Anderson Localized']]


mbl_no=train_mbl_indices[np.random.randint(len(train_mbl_indices))]


# print(X_train.shape)
# print(y_train[:10])
# ind=3
# plot_heatmap(plot_matrix=X_train[ind],label=y_train[ind].item(),tensor_index=ind)


tmbl=X_train[mbl_no][:,:,0]
print(tmbl.shape)


#Remove plotting
#plot_heatmap(plot_matrix=tmbl,label=rev_label_dic['MBL'],tensor_index=mbl_no)

# if len(train_erg_indices)>0:
#     erg_no=train_erg_indices[np.random.randint(len(train_erg_indices))]
#     terg=X_train[erg_no][:,:,0]
#     plot_heatmap(plot_matrix=terg,label=rev_label_dic['Ergodic'],tensor_index=erg_no)

# if len(train_al_indices)>0:
#     al_no=train_al_indices[np.random.randint(len(train_al_indices))]
#     tal=X_train[al_no][:,:,0]
#     plot_heatmap(plot_matrix=tal,label=rev_label_dic['Anderson Localized'],tensor_index=al_no)





# general parameters
#dtype = torch.float32 # very important - already set
seed = 1 # fixed random seed for reproducibility
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # use GPU if available
print(device)

example = False # dataset is MNIST example if True
#ising = True # import ising dataset

# set functions for neural network
loss_fn = nn.CrossEntropyLoss   # 'MSELoss' or 'CrossEntropyLoss'
optimizer_fn = torch.optim.Adam     # 'Adam' or 'AdamW' or 'SGD'
activation = nn.ReLU    # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

# set parameters for neural network
# weight_decay = 1.0
# learning_rate = 1e-3 # lr
# weight_multiplier = 10.0 # initialization-scale
# input_dim = 16**2 # 28 x 28 array of pixel values in MNIST (16x16 in Ising data)
# output_dim = 2 # 10 different digits it could recognize (2 diff phases for Ising)
# hiddenlayers=[5]

# set torch data type and random seeds
torch.set_default_dtype(dtype)





class CNN_Rect(nn.Module):
    def __init__(self, input_dims, output_size, input_channels, conv_channels, hidden_widths,
                 activation=nn.ReLU(), optimizer=torch.optim.Adam,
                 learning_rate=0.001, weight_decay=0, multiplier=1, dropout_prob=0):

        super().__init__()

        # Ensure conv_channels has enough channels for three convolutional layers
        if len(conv_channels) < 3:
            raise ValueError("conv_channels must have at least three elements for three convolutional layers")

        # Convolution layers
        conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv_channels[0],
                          kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1],
                          kernel_size=3, stride=1, padding=1)
        conv3 = nn.Conv2d(in_channels=conv_channels[1], out_channels=conv_channels[2],
                          kernel_size=3, stride=1, padding=1)

        # Pooling layer
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Construct convolution and pool layers
        self.conv_layers = nn.ModuleList()
        for conv_layer in [conv1, conv2, conv3]:
            self.conv_layers.append(conv_layer)
            self.conv_layers.append(activation)
            self.conv_layers.append(pool)

        # Dropout for FC layers
        self.dropout = nn.Dropout(dropout_prob)

        # Calculate size after convolutions and pooling
        sample_input = torch.zeros(1, input_channels, *input_dims)
        for layer in self.conv_layers:
            sample_input = layer(sample_input)
        flattened_size = sample_input.numel()

        # Construct fully connected layers
        self.fc_layers = nn.ModuleList()
        input_size = flattened_size
        for size in hidden_widths:
            self.fc_layers.append(nn.Linear(input_size, size))
            input_size = size  # For the next layer
            self.fc_layers.append(activation)
            self.fc_layers.append(self.dropout)

        # Last layer without activation or dropout
        self.fc_layers.append(nn.Linear(input_size, output_size))

        # Multiply weights by overall factor
        if multiplier != 1:
            with torch.no_grad():
                for param in self.parameters():
                    param.data = multiplier * param.data

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Set optimizer
        self.optimizer = optimizer(params=self.parameters(),
                                   lr=learning_rate, weight_decay=weight_decay)

    def forward(self, X):
        # convolution and pooling
        for layer in self.conv_layers:
            X = layer(X)
        # fully connected layers
        X = X.view(X.size(0), -1)   # Flatten data for FC layers
        for layer in self.fc_layers:
            X = layer(X)
        return X




random.seed(seed)
for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
    set_seed(seed)
#Remember to change the output size if number of things changes!!
# initialize model and print details
lr=0.0001
wd=0.08
wm=10
model = CNN_Rect(input_dims=(150,L), output_size=2, input_channels=3, conv_channels=[4,8,16], hidden_widths=[200,100],
				activation=nn.ReLU(), optimizer=torch.optim.Adam,
				learning_rate=lr, weight_decay=wd, multiplier=wm, dropout_prob=0)
#model.double()
print(model)
first_param = next(model.parameters())
print(first_param.dtype)
print(X_test[0].device)
# Define loss function
criterion = nn.CrossEntropyLoss()
# Define optimizer for stochastic gradient descent (including learning rate and weight decay)
# use the one I defined as an attribute in the CNN class
optimizer = model.optimizer




start_time = timer()
first_time_training = True
epochs = 1000 # how many runs through entire training data
save_models=True
save_interval=200
# live_plot = create_live_plot()
# live_plot
if save_models:
    run_name = f"../Results/mbl_xyz_hw200-100_wd_{str(wd)}_mult_{str(wm)}_average_{encoded_averaging}_{int(time.time())}"
    os.mkdir(run_name)
    save_dict = {'model':model.state_dict(), 'train_data':train_data, 'test_data':test_data}
    torch.save(save_dict, run_name+'/init.pth')

if  first_time_training == True:
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
else:
    print("Starting additional training")
    epochs = 2900

#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
for i in range(epochs):
    train_correct = 0
    test_correct = 0

    # Run the training batches
    for batch, (X_train, y_train) in enumerate(train_loader):

        # Apply the current model to make prediction of training data

        # Flatten input tensors to two index object with shape (batch_size, input_dims) using .view()
        # Predict label probabilities (y_train) based on current model for input (X_train)
        y_pred = model(X_train.view(batch_size, 3, 150, L).to(device))#note- data dimension set to number of points, 1 only one channel, 16x16 for each data-point. Model transforms 2d array into 3d tensors with 4 channels
        # print(f"y_pred: {y_pred.dtype}")
        train_loss = criterion(y_pred, y_train.to(device))
        # print(f"train_loss: {train_loss.dtype}")
        # Tally the number of correct predictions per epoch
        predicted = torch.max(y_pred.data, 1)[1]
        train_correct += (predicted == y_train.to(device)).sum().item()
        #print(f"train_correct: {train_correct.dtype}")

        # Update parameters
        optimizer.zero_grad() # clears old gradients stored in previous step
        train_loss.backward() # calculates gradient of loss function using backpropagation (stochastic)
        # print(model.device)
        # print(f"model: {next(model.parameters()).dtype}")

        optimizer.step() # adjust parameters in direction of steepest descent

    # Update overall train loss (most recent batch) & accuracy (of all batches) for the epoch
    train_losses.append(train_loss.item())
    train_accuracy.append(train_correct/train_size)

    # Run the testing batches
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test.view(test_size, 3, 150, L).to(device))

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            test_correct += (predicted == y_test.to(device)).sum().item()


    # Update test loss & accuracy for the epoch

    test_loss = criterion(y_val, y_test.to(device))
    test_losses.append(test_loss.item())
    test_accuracy.append(test_correct/test_size)
    if save_models and i%save_interval==0:
        save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'epoch': i,
            }
        torch.save(save_dict, run_name+f"/{i}.pth")


    if i % 200 == 0 or i == epochs-1:
        # Print interim result
        print(f"epoch: {i}, train loss: {train_loss.item():.4f}, accuracy: {train_accuracy[-1]:.4f}")
        print(12*" " + f"test loss: {test_loss.item():.4f}, accuracy: {test_accuracy[-1]:.4f}" )
        print(60*"*")
        #print(f"Saved model to {root/run_name/f'{i}.pth'}")
        #plot_metrics(train_losses, test_losses, train_accuracy, test_accuracy, i)
# print(f'\nDuration: {timer() - start_time:.0f} seconds') # print the time elapsed


filename_grokkingcurve_losses=f'losses_mbl_xyz_wd_{str(wd)}_mult_{str(wm)}_lr_{str(lr)}avg_{encoded_averaging}_{int(time.time())}.p'
data = ["data[1] is of form [[seed],train_loss, test_loss], [train_acc, test_acc]", [[seed],[train_losses, test_losses], [train_accuracy, test_accuracy]]]
# print(data)
# print(data[0][:10])
with open(filename_grokkingcurve_losses, "wb") as handle:
    pickle.dump(data, handle)
print("data saved")
print(filename_grokkingcurve_losses)
# print(data[1])



# with open(str(root)+filename_grokkingcurve_losses, 'rb') as handle:
#     data1 = pickle.load(handle)

# with open("MagzeroNoScrambleGrokData_size_100_mult_20_decay_0.08_prob_0.p", "rb") as handle:
#         data = pickle.load(handle)

# print(data1[0])
# print(len(data1[1][0][1]))

#Plot the data

# [train_losses,test_losses]=data1[1][1]
# [train_accuracy,test_accuracy]=data1[1][2]

# Plot losses and accuracies of train and test data as epochs go on
fig, ax = plt.subplots(1,2)
# loss plots
ax[0].semilogx(train_losses, c='r',label="train")
ax[0].semilogx(test_losses, c='b',label="test")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")

# accuracy plots
ax[1].semilogx(train_accuracy, c='r',label="train")
ax[1].semilogx(test_accuracy, c='b', label="test")
#plt.hlines(y=39.5, colors='aqua', linestyles='-', lw=2, label='Single Short Line')#xmin=100, xmax=175
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

fig.suptitle(f"AL/MBL x mixed Train size = {str(train_size)}, weight multiplier = {str(int(wm))}, weight decay = {str(wd)}, dropout prob=0,"+
                            "\n CNN: Convo layers: [16, 32, 64], Hidden FC layers: [30,30]")

fig.tight_layout()
fig.savefig(f"mbl_xyz_wd_{str(wd)}_lr_{str(lr)}_wm_{str(wm)}_{int(time.time())}.pdf")



# data = ["data[1] is of form [train_loss, test_loss], [train_acc, test_acc]", [[train_losses, test_losses], [train_accuracy, test_accuracy]]]
# with open(str(root)+"/MBL04012024hidden200100snapshots2000.p", "wb") as handle:
#     pickle.dump(data, handle)
# print("data saved")



