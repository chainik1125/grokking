
#import kaleido
import os
import sys

print(int(sys.argv[1]))


from timeit import default_timer as timer
import time
import pickle
from tqdm.auto import tqdm # for loop progress bar
import itertools as it
import numpy as np
import random
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# machine learning imports

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader



data_seed=int(sys.argv[2])
sgd_seed=0
init_seed=0


weight_decay=int(sys.argv[3])/100

grok_str=str(sys.argv[4])
print(f" seed: {data_seed}")
print(f" wd: {weight_decay}")
print(f" grok_str: {grok_str}")

if grok_str=='True':
    grok=True
elif grok_str=='False':
    grok=False


train_size=5
test_size=1000
grok_locations=[0,100]#

if grok:
    learning_rate=10**-4
    weight_decay=weight_decay
    weight_multiplier=10

else:
    learning_rate=10**-4
    weight_decay=0.01
    weight_multiplier=1
    

root=f'../../ClusterResults/Ising_data_seed_{data_seed}_train_{train_size}_grok_{grok}_wd_{str(weight_decay)}time_{int(time.time())}'
os.mkdir(root)
print(str(root))



dtype = torch.float32 # very important
# seed = seed # fixed random seed for reproducibility
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # use GPU if available
example = False # dataset is MNIST example if True
ising = True # import ising dataset

# set functions for neural network
loss_fn = nn.CrossEntropyLoss   # 'MSELoss' or 'CrossEntropyLoss'
optimizer_fn = torch.optim.Adam     # 'Adam' or 'AdamW' or 'SGD'
activation = nn.ReLU    # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

# set parameters for neural network
weight_decay = weight_decay
learning_rate = learning_rate # lr
weight_multiplier = weight_multiplier # initialization-scale
dropout_prob=0
input_dim = 16**2 # 28 x 28 array of pixel values in MNIST (16x16 in Ising data)
output_dim = 2 # 10 different digits it could recognize (2 diff phases for Ising)
hiddenlayers=[20]
conv_channels=[2,4]

# set torch data type and random seeds
torch.set_default_dtype(dtype)


# basic Convolutional Neural Network for input datapoints that are 2D tensors (matrices)
# size of input is (num_datapts x input_channels x input_dim x input_dim)  and requires input_dim % 4 = 0
# Conv, Pool, Conv, Pool, Fully Connected, Fully Connected, ...,  Output
# zero padding is included to ensure same dimensions pre and post convolution
class CNN(nn.Module):
	def __init__(self, input_dim, output_size, input_channels, conv_channels, hidden_widths,
				activation=nn.ReLU(), optimizer=torch.optim.Adam,
				learning_rate=0.001, weight_decay=0, multiplier=1, dropout_prob=0.01):

		super().__init__()

		# transforms input from input_channels x input_dim x input_dim
		# to out_channels x input_dim x input_dim
		conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv_channels[0],
									kernel_size=3, stride=1, padding=1)
		conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1],
									kernel_size=3, stride=1, padding=1)
		# divide shape of input_dim by two after applying each convolution
		# since this is applied twice, make sure that input_dim is divisible by 4!!!
		pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Construct convolution and pool layers
		self.conv_layers = nn.ModuleList()
		for conv_layer in [conv1, conv2]:
			self.conv_layers.append(conv_layer)
			self.conv_layers.append(activation)
			self.conv_layers.append(pool)

		# dropout to apply in FC layers
		self.dropout = nn.Dropout(dropout_prob)

		# construct fully connected layers
		self.fc_layers = nn.ModuleList()
		# flattened size after two convolutions and two poolings of data
		input_size = (input_dim//4)**2 * conv_channels[1]
		for size in hidden_widths:
			self.fc_layers.append(nn.Linear(input_size, size))
			input_size = size  # For the next layer
			self.fc_layers.append(activation)
			self.fc_layers.append(self.dropout)
		# add last layer without activation or dropout
		self.fc_layers.append(nn.Linear(input_size, output_size))

		# multiply weights by overall factor
		if multiplier != 1:
			with torch.no_grad():
				for param in self.parameters():
					param.data = multiplier * param.data

		# use GPU if available
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

		# set optimizer
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


random.seed(data_seed)
for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
    set_seed(data_seed)



if ising:
    with open("../../Data/IsingML_L16_traintest.pickle", "rb") as handle:
        data = pickle.load(handle)
        print(f'Data length: {len(data[1])}')
        print(data[0])
        data = data[1]
    # shuffle data list
    random.shuffle(data)
    # split data into input (array) and labels (phase and temp)
    inputs, phase_labels, temp_labels = zip(*data)
    # for now ignore temp labels
    my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
    my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
    my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
    print(my_X.dtype, my_y.dtype)
    print("Created Ising Dataset")
else:
    print("Need to input your own training and test data")
	



# manually do split between training and testing data (only necessary if ising data)
# otherwise use torch.utils.data.Subset to get subset of MNIST data
train_size, test_size, batch_size = train_size, test_size, train_size
a, b = train_size, test_size
train_data = TensorDataset(my_X[b:a+b], my_y[b:a+b]) # Choose training data of specified size
test_data = TensorDataset(my_X[:b], my_y[:b]) # test
scramble_snapshot=False

# load data in batches for reduced memory usage in learning
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)
for b, (X_train, y_train) in enumerate(train_loader):
    print("batch:", b)
    print("input tensors shape (data): ", X_train.shape)
    print("output tensors shape (labels): ", y_train.shape)

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

    print("output tensors shape (labels): ", y_test.shape)



desc='data[1] is [[data_seed,sgd_seed,init_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]'
data_save1=[[data_seed,sgd_seed,init_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]
data_save=[desc,data_save1]

with open(str(root)+f"/train_test_data_seed{str(data_seed)}.p", "wb") as handle:
    pickle.dump(data_save, handle)
print("data saved")

random.seed(init_seed)
for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
    set_seed(init_seed)

# initialize model and print details
model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
				activation=nn.ReLU(), optimizer=torch.optim.Adam,
				learning_rate=learning_rate, weight_decay=weight_decay, multiplier=weight_multiplier, dropout_prob=dropout_prob)
print(model)
# Define loss function
criterion = nn.CrossEntropyLoss()
# Define optimizer for stochastic gradient descent (including learning rate and weight decay)
# use the one I defined as an attribute in the CNN class
optimizer = model.optimizer



start_time = timer()
first_time_training = True
epochs = 100000 # how many runs through entire training data
save_models=True
save_interval=100

random.seed(sgd_seed)
for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
    set_seed(sgd_seed)
if save_models:
    run_name = f"models_cnn_{conv_channels}_hw{hiddenlayers}_time{int(time.time())}"
    os.mkdir(root+'/'+run_name)
    save_dict = {'model':model.state_dict(), 'train_data':train_data, 'test_data':test_data}
    torch.save(save_dict, root+'/'+run_name+'/'+'/init.pth')

if  first_time_training == True:
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
else:
    print("Starting additional training")
    epochs = 2900
first=True
for i in tqdm(range(epochs)):
    train_correct = 0
    test_correct = 0

    # Run the training batches
    for batch, (X_train, y_train) in enumerate(train_loader):

        # Apply the current model to make prediction of training data

        # Flatten input tensors to two index object with shape (batch_size, input_dims) using .view()
        # Predict label probabilities (y_train) based on current model for input (X_train)
        y_pred = model(X_train.view(batch_size, 1, 16, 16).to(device))#note- data dimension set to number of points, 1 only one channel, 16x16 for each data-point. Model transforms 2d array into 3d tensors with 4 channels
        train_loss = criterion(y_pred, y_train.to(device))

        # Tally the number of correct predictions per epoch
        predicted = torch.max(y_pred.data, 1)[1]
        train_correct += (predicted == y_train.to(device)).sum().item()

        # Update parameters
        optimizer.zero_grad() # clears old gradients stored in previous step
        train_loss.backward() # calculates gradient of loss function using backpropagation (stochastic)
        optimizer.step() # adjust parameters in direction of steepest descent

    # Update overall train loss (most recent batch) & accuracy (of all batches) for the epoch
    train_losses.append(train_loss.item())
    train_accuracy.append(train_correct/train_size)

    # Run the testing batches
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = model(X_test.view(test_size, 1, 16, 16).to(device))

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
        torch.save(save_dict, root+'/'+run_name+'/'+f"{i}.pth")
        if i%1000==0:
            print(f"Saved model to {root+'/'+run_name+'/'+f'{i}.pth'}")

    if i % 200 == 0 or i == epochs-1:
        # Print interim result
        print(f"epoch: {i}, train loss: {train_loss.item():.4f}, accuracy: {train_accuracy[-1]:.4f}")
        print(12*" " + f"test loss: {test_loss.item():.4f}, accuracy: {test_accuracy[-1]:.4f}" )
        print(60*"*")

print(f'\nDuration: {timer() - start_time:.0f} seconds') # print the time elapsed

data = ["data[1] is of form [train_loss, test_loss], [train_acc, test_acc]", [[train_losses, test_losses], [train_accuracy, test_accuracy]]]
with open(str(root)+'/'+f"Isinglosses_hw{hiddenlayers}_cnn_{conv_channels}_data_seed_{data_seed}_{int(time.time())}.p", "wb") as handle:
    pickle.dump(data, handle)
print("data saved")




run_folder=str(root)
#run_folder='models_seed0_cnn_[2, 4]_hw[30]_time1706744598'

def stitch_losscurve(run_folder):
    train_losses=[]
    test_losses=[]
    train_accs=[]
    test_accs=[]

    onlyfiles = [f for f in os.listdir(run_folder) if 'losses' in f]
    def sort_by_time(filestring):
        print((filestring.split(f'data_seed_{data_seed}_')[-1]))
        return int((filestring.split(f'data_seed_{data_seed}_')[-1]).split('.')[0])

    onlyfiles.sort(key=sort_by_time)
    print(onlyfiles)


    for run in onlyfiles:
        #with open(str(root)+'/'+str(run_name)+'/'+run, "rb") as f:
        with open(str(root)+'/'+run, "rb") as f:
            losses_data=pickle.load(f)
        print(losses_data[0])


        train_losses=train_losses+losses_data[1][0][0]
        test_losses=test_losses+losses_data[1][0][1]
        train_accs=train_accs+losses_data[1][1][0]
        test_accs=test_accs+losses_data[1][1][1]
    return [train_losses,test_losses],[train_accs,test_accs]

[train_losses1,test_losses1],[train_accs,test_accs]=stitch_losscurve(run_folder)

fig,axs=plt.subplots(2)
axs[0].title.set_text('Losses')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].plot(list(range(epochs)),train_losses1,label='Train')
axs[0].plot(list(range(epochs)),test_losses1,label='Test')

axs[1].title.set_text('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].plot(list(range(epochs)),train_accs,label='Train')
axs[1].plot(list(range(epochs)),test_accs,label='Test')
fig.savefig(str(root)+'/Losscurve.png')





fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(
        x=list(range(len(train_losses1))),
        y=train_losses1,
        name='Train losses'
    ),
    row=1,
    col=1)

fig.add_trace(
    go.Scatter(
        x=list(range(len(test_losses1))),
        y=test_losses1,
        name='Test losses'
    ),
    row=1,
    col=1)

fig.add_trace(
    go.Scatter(
        x=list(range(len(train_accs))),
        y=train_accs,
        name='Train Accuracy'
    ),
    row=1,
    col=2)

fig.add_trace(
    go.Scatter(
        x=list(range(len(test_accs))),
        y=test_accs,
        name='Test Accuracy'

    ),
    row=1,
    col=2)

# Update layout
fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_yaxes(title_text="Accuracy", row=1, col=2,type='log')
fig.update_yaxes(type='log', row=1, col=1)
fig.update_xaxes(title_text='Epochs', row=1, col=2)

fig.update_layout(
    title=f'CNN conv_layers{str(conv_channels)}, MLP {str(hiddenlayers)}, lr={str(learning_rate)}, wd={str(weight_decay)}, wm={str(weight_multiplier)}, train size={train_size}, test size={test_size}, one batch, W=6,7,8',
    xaxis_title='Epochs',
    # yaxis_title='Test accuracy',
    showlegend=True,
)

fig.write_image(root+"/Losses_plotly.png")


run_folder=str(root)
onlyfiles = [f for f in os.listdir(run_folder) if 'models' in f]
print(int(onlyfiles[0].split('time')[-1]))
def sort_by_time(filestring):
    return int((filestring.split('time')[-1]))

onlyfiles.sort(key=sort_by_time)


model_folder=str(root)+'/'+onlyfiles[-1]

#Defines the image - input to the function
example_images=X_test.view(1000, 1, 16, 16).to(device)
example_image=example_images[0]

#necessary imports

from os import listdir
from os.path import isfile, join


#OK - let's iterate over ensembles of the training data

def generate_training_set():
    ising=True
    if ising:
        with open("../../Data/IsingML_L16_traintest.pickle", "rb") as handle:
            data = pickle.load(handle)
            #print(f'Data length: {len(data[1])}')
            #print(data[0])
            data = data[1]
        # shuffle data list
        random.shuffle(data)
        # split data into input (array) and labels (phase and temp)
        inputs, phase_labels, temp_labels = zip(*data)
        # for now ignore temp labels
        my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
        my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
        my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
        #print(my_X.dtype, my_y.dtype)
        #print("Created Ising Dataset")
    else:
        print("Need to input your own training and test data")


    train_size, test_size, batch_size = 100, 1000, 100
    a, b = train_size, test_size
    train_data = TensorDataset(my_X[b:a+b], my_y[b:a+b]) # Choose training data of specified size
    test_data = TensorDataset(my_X[:b], my_y[:b]) # test
    scramble_snapshot=False

    # load data in batches for reduced memory usage in learning
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)
    # for b, (X_train, y_train) in enumerate(train_loader):
        #print("batch:", b)
        #print("input tensors shape (data): ", X_train.shape)
        #print("output tensors shape (labels): ", y_train.shape)

    for b, (X_test, y_test) in enumerate(test_loader):
        if scramble_snapshot:
            X_test_a=np.array(X_test)
            X_test_perc=np.zeros((test_size,16,16))
            for t in range(test_size):
                preshuff=X_test_a[t,:,:].flatten()
                np.random.shuffle(preshuff)
                X_test_perc[t,:,:]=np.reshape(preshuff,(16,16))
            X_test=torch.Tensor(X_test_perc).to(dtype)
        #print("batch:", b)
        #print("input tensors shape (data): ", X_test.shape)
        #print("output tensors shape (labels): ", y_test.shape)

    return test_loader

#Urgh - need a function that will let me find the epoch of a given file
def get_epoch(x):
    #assume it's of the form <number>.pth
    return int(x.split('.')[0])

def accuracy(test_loader):
    epoch=-1
    pathtodata=model_folder
    print(model_folder)

    onlyfiles = [f for f in listdir(pathtodata) if isfile(join(pathtodata, f)) if f!='init.pth']
    onlyfiles=sorted(onlyfiles,key=get_epoch)


    full_run_data = torch.load(pathtodata+f'/{onlyfiles[epoch]}')
    print(pathtodata)
    print(f'File name: {onlyfiles[epoch]}')
    #test_model=model
    test_model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
                    activation=nn.ReLU(), optimizer=torch.optim.Adam,
                    learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)

    test_model.to(device)
    test_model.load_state_dict(full_run_data['model'])
    test_correct=0
    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):

            # Apply the model
            y_val = test_model(X_test.view(test_size, 1, 16, 16).to(device))

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1]
            test_correct += (predicted == y_test.to(device)).sum().item()
            acc=test_correct/test_size
    return acc


test_loader=generate_training_set()
# accuracy=accuracy(test_loader)
# print(accuracy)

ensemble_acc=[]
print(len(ensemble_acc))
for i in tqdm(range(1,5), desc="Processing List", unit="i"):
    test_loader=generate_training_set()
    acc=accuracy(test_loader)
    ensemble_acc.append(acc)


# print(f'Model folder: {grok_folder}')
# print(ensemble_acc)

#onlyfiles = [f for f in listdir(grok_folder) if isfile(join(grok_folder, f)) if f!='init.pth']


fig,axs=plt.subplots(1)
axs.scatter(list(range(len(ensemble_acc))),ensemble_acc)
axs.set_title('Accuracy on random test sets')
axs.set_xlabel('Test set')
axs.set_ylabel('Accuracy')
fig.savefig(str(root)+'/Sample_acc.png')




#I think this allows you to not have to define a new function
from os import listdir
from os.path import isfile, join



#File locations
#1.1 FULL ISING NO SCRAMBLE...

#grok_folder=str(root)+'/FullIsing_noscramble_time1700670593'
#1. Full Ising, No Scramble
#grok_folder=str(root)+'/FullIsing_noscramble_time1700587569'
#3. Magzero No Scramble
#grok_folder=str(root)+'/magzero_noscramble_time1700665129'

#grok_folder=str(root)+'/grok_shuffle1699644166'
#grok_folder=str(root)+'/grok_magzero1699637978'

#grok_folder=run_name#This plays the role of pathtodata
#print(grok_folder)


#grok_1699316359 - 20k images



def hook_fn2(module,input,output,activation_list):
    activation_list.append(output)

def get_model_neuron_activations(epochindex,pathtodata,image_number,image_set):
    model_activations=[]
    temp_hook_dict={}

    #1. define the model:
    #1a. Get the saved_files
    onlyfiles = [f for f in listdir(pathtodata) if isfile(join(pathtodata, f)) if f!='init.pth']



    full_run_data = torch.load(pathtodata+f'/{onlyfiles[epochindex]}')

    onlyfiles=sorted(onlyfiles,key=get_epoch)


    full_run_data = torch.load(pathtodata+f'/{onlyfiles[epochindex]}')
    print(f'File name: {onlyfiles[epochindex]}')
    epoch=int(onlyfiles[epochindex].split('.')[0])


    test_model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
                    activation=nn.ReLU(), optimizer=torch.optim.Adam,
                    learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)

    test_model.to(device)
    test_model.load_state_dict(full_run_data['model'])
    #Define the hook
    def hook_fn(module, input, output):
        model_activations.append(output)

    count=0
    for layer in test_model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
            count+=1
            handle = layer.register_forward_hook(hook_fn)
            #print(type(layer).__name__)
            temp_hook_dict[count]=handle


    # 2. Forward pass the example image through the model
    with torch.no_grad():
        #print(example_image.shape)
        output = test_model(image_set)


    # Detach the hook
    for hook in temp_hook_dict.values():
        hook.remove()


    # 3. Get the activations from the list  for the specified layer


    #print([len(x) for x in model_activations])

    detached_activations=[]
    #print(model_activations[1].shape)
    for act in model_activations:
        #print(f'len(act): {len(act)}')
        if image_number==None:#averages over all images
            flat=torch.flatten(sum([act[x] for x in range(len(act))])/len(act))
        elif image_number=='dist':
            flat=act
        elif image_number=='var':
            mean=torch.flatten(sum([act[x] for x in range(len(act))])/len(act))
            x2=torch.flatten(sum([act[x]**2 for x in range(len(act))])/len(act))
            # print(len(act))
            # print(act[0].shape)
            # print(act[0])
            # print(act[0]**2)
            # print(len(mean))
            # print(len(x2))
            flat=(x2-mean**2)**(1/2)
            flat=torch.flatten(flat)
            #print(flat)
        else:
            flat=torch.flatten(act[image_number])
        #print(type(flat))
        #print(flat.shape)
        if image_number=='dist':
            detached_act=flat.detach().cpu().numpy()
        else:
            detached_act=flat.detach().cpu().numpy()
        detached_activations.append(detached_act)

    model_activations.clear() #Need this because otherwise the hook will repopulate the list!

    #detached_activations_array=np.stack(detached_activations,axis=0)#Convert to an array to keep the vectorization-->You don't want to do this because each layer has a different shape

    return detached_activations,epoch

# sorted_by=None
# test=get_model_neuron_activations(epoch=0,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# #test20=get_model_neuron_activations(epoch=1,pathtodata=grok_folder,image_number='var') # For some reason this epoch has a negative variance. Does that indicate you're calcualting the variance wrong?
# #test100=get_model_neuron_activations(epoch=5,pathtodata=grok_folder,image_number='var',image_set=example_images)
# test1000=get_model_neuron_activations(epoch=50,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test5000=get_model_neuron_activations(epoch=250,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test30000=get_model_neuron_activations(epoch=600,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test30000=get_model_neuron_activations(epoch=600,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test40000=get_model_neuron_activations(epoch=800,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test3=get_model_neuron_activations(epoch=-10,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test4=get_model_neuron_activations(epoch=-2,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test5=get_model_neuron_activations(epoch=-3,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
# test2=get_model_neuron_activations(epoch=-1,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)


#plotting

def plot_activation_hist(activation_list,epoch):
    freqs=[]
    bin_cs=[]

    fig,ax= plt.subplots(1,len(activation_list),figsize=(20,5))
    for act in activation_list:
        hist=np.histogram(act)
        freq = hist[0]
        bin_edges=hist[1]
        bin_centers=[(bin_edges[i]+bin_edges[i+1])/2 for i in range(0,len(bin_edges)-1)]
        freqs.append(freq)
        bin_cs.append(bin_centers)
    count=0
    for i in range(len(freqs)):
        count+=1
        ax[i].title.set_text(f"Layer {count} Epoch {epoch}")
        ax[i].set_xlabel("Activation (averaged over training images)")
        ax[i].set_ylabel("Frequency")
        ax[i].plot(bin_cs[i],freqs[i],color='r')
        #plt.scatter(bin_cs2[i],freqs2[i],color='b',label='First model (Ep. 100)')
        # ax[i].legend()
    #plt.show()


# plot_activation_hist(activation_list=test,epoch=1)
# #plot_activation_hist(activation_list=test20,epoch=20) <-- Negative variance - am I calculating var right?
# #plot_activation_hist(activation_list=test100,epoch=100)
# plot_activation_hist(activation_list=test1000,epoch=1000)
# plot_activation_hist(activation_list=test5000,epoch=5000)
# plot_activation_hist(activation_list=test30000,epoch=30000)
# plot_activation_hist(activation_list=test40000,epoch=40000)
# plot_activation_hist(activation_list=test3,epoch=48000)
# plot_activation_hist(activation_list=test5,epoch=49960)
# plot_activation_hist(activation_list=test4,epoch=49980)
# plot_activation_hist(activation_list=test2,epoch=50000)




def plot_activation_hist_one(epochlist,sortby,train_accuracies,test_accuracies):
    activation_test,epoch=get_model_neuron_activations(epochindex=0,pathtodata=model_folder,image_number=sortby,image_set=example_images)
    fig,axs=plt.subplots(len(epochlist),len(activation_test)+1,figsize=(20,15))


    for i in epochlist:
        print(f'i {i}')
        activation_list,epoch1=get_model_neuron_activations(epochindex=int(i/100),pathtodata=model_folder,image_number=sortby,image_set=example_images)
        freqs=[]
        bin_cs=[]

        print(len(freqs))
        axs[epochlist.index(i)][0].title.set_text(f"Grok curve - Accuracy")
        #ymin, ymax = axs[epochlist.index(i)][0].get_ylim()
        axs[epochlist.index(i)][0].vlines(x=epoch1+1,ymin=0.4,ymax=0.99,linestyles='dashed')
        axs[epochlist.index(i)][0].semilogx(train_accuracies[3:], c='r',label="train")
        axs[epochlist.index(i)][0].semilogx(test_accuracies[3:], c='b', label="test")
        axs[epochlist.index(i)][0].set_xlabel("Epoch")
        axs[epochlist.index(i)][0].set_ylabel("Accuracy")
        axs[epochlist.index(i)][0].legend()

        for act in activation_list:
            hist=np.histogram(act)
            freq = hist[0]
            bin_edges=hist[1]
            bin_centers=[(bin_edges[k]+bin_edges[k+1])/2 for k in range(0,len(bin_edges)-1)]
            freqs.append(freq)
            bin_cs.append(bin_centers)
        count=0

        for j in range(len(freqs)):
            count+=1
            print(epochlist.index(i),j)
            axs[epochlist.index(i)][j+1].title.set_text(f"Layer {count} Epoch {epoch1}")
            axs[epochlist.index(i)][j+1].set_xlabel("Activation (averaged over training images)")
            axs[epochlist.index(i)][j+1].set_ylabel("Frequency")
            axs[epochlist.index(i)][j+1].plot(bin_cs[j],freqs[j],color='r')
            #axs[epochlist.index(i)][j].plot([1,2,3,4,5],[1,2,3,4,5])
        plt.tight_layout()
    #plt.show()
    fig.savefig(str(root)+"/Barry_activation_hist.pdf")
            #axs[epochlist.index(i)][j].scatter(bin_cs2[i],freqs2[i],color='b',label='First model (Ep. 100)')
            # ax[i].legend()
print(f'Model folder: {model_folder}')
plot_activation_hist_one(epochlist=grok_locations,sortby='var',train_accuracies=train_accs,test_accuracies=test_accs)






#Interp




#Single neuron properties

#magnetization

def magnetization(spin_grid):
    return sum(sum(spin_grid))

#1. Energy vs. activation

def compute_energy(spin_grid, J=1):
    """
    Compute the energy of a 2D Ising model.

    Parameters:
    spin_grid (2D numpy array): The lattice of spins, each element should be +1 or -1.
    J (float): Interaction energy. Defaults to 1.

    Returns:
    float: Total energy of the configuration.
    """
    energy = 0
    rows, cols = spin_grid.shape

    for i in range(rows):
        for j in range(cols):
            # Periodic boundary conditions
            right_neighbor = spin_grid[i, (j + 1) % cols]
            bottom_neighbor = spin_grid[(i + 1) % rows, j]

            # Sum over nearest neighbors
            energy += -J * spin_grid[i, j] * (right_neighbor + bottom_neighbor)

    return energy/(2*J*spin_grid.shape[0]*spin_grid.shape[1])

# Example usage
# spin_grid = np.array(X_test[2].detach().cpu().numpy())
# print("Energy of the configuration:", compute_energy(spin_grid))
# print(round(compute_energy(spin_grid),2))


# plt.spy(X_test[2] + 1 )

#2. Connected cluster


#http://dragly.org/2013/03/25/working-with-percolation-clusters-in-python/
#That's one implementation, but doesn't account for PBC's - so I'll
# from pylab import *
# import scipy.ndimage

# lw, num = scipy.ndimage.label((X_test[0]+1)/2)
# print(lw,num)
# print(lw.shape)
# plt.spy(X_test[0]+1)
# plt.show()

#Let me try the networkx path

import networkx as nx
import itertools



#1. First need to find the neigbours of a given point

def find_neighbours(image, position):
    neighbour_positions=[]
    linear_size=image.shape[0]

    neighbour_positions=[((position[0]+i)%linear_size,(position[1]+j)%linear_size) for i,j in itertools.product([-1,0,1],repeat=2)]
    value=image[position[0],position[1]]
    matches=[neighbour_positions[i] for i in range(len(neighbour_positions)) if value==image[neighbour_positions[i][0],neighbour_positions[i][1]]]
    return matches


#2. Then find the connectivity matrix
def form_connectivity_matrix(image):
    dim=1
    for i in image.shape:
        dim=i*dim
        #print(f'dim {dim}')
    cm=np.zeros((dim,dim))
    linear_dim=image.shape[0]
    #print(f'linear dim {linear_dim}')
    for i in range(cm.shape[0]):
        #first convert i to a coordinate
        imagepos=[int(i/16),i%16]
        neighbours=find_neighbours(image=image,position=imagepos)
        for n in neighbours:
            #first you need to convert the neighbour positions into array indices
            neighbourindex=n[0]*16+n[1]
            cm[i,neighbourindex]=1
            cm[neighbourindex,i]=1
    #In principle you should go through each row and column but I think just one should be sufficient.
    for i in range(cm.shape[0]):
        cm[i,i]=0
    return cm


#3. Create a networkx graph from that matrix
def create_graph(c_matrix):
    linear_dim=c_matrix.shape[0]
    G=nx.Graph()
    G.add_nodes_from([i for i in range(linear_dim)])
    edges=np.transpose(np.nonzero(c_matrix))
    G.add_edges_from(edges)
    return G
#4. Use networkx to extract the connected component
def get_ccs(image):
    conn_matrix=form_connectivity_matrix(image)
    graph=create_graph(conn_matrix)
    ccs=max(nx.connected_components(graph), key=len)
    largest_cc = len(max(nx.connected_components(graph), key=len))
    return largest_cc


#Functions to pick out a neuron and get the image distribution associated with that neuron

#Let's just try to look at a sample of images for the highly stimulated neurons:

#print(grok_folder)
#grok_folder=str(root)+'/grok_magzero1699637978'
epind=-1
test2,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='var',image_set=example_images)#Note that this gives me average ordered
testdist,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='dist',image_set=example_images)





#Let's just do the dumb thing and try to go through the most stimulated neurons and feed in images and see what stimulates it most and least

neuron_indices=[]
for i in range(len(test2)):
    t1=np.argsort(test2[i])
    neuron_indices.append(t1)

# print(neuron_indices[-2])
# print(test2[-2][6])

def get_neuron(neuron_activation_list,layer,nlargest):
    #Sort by size
    layer_act=neuron_activation_list[layer]
    if len(layer_act.shape)>2:
        layer_act_flat=[layer_act[i].flatten() for i in range(len(layer_act))]
    else:
        layer_act_flat=layer_act
    size_sorted_indices=np.argsort(layer_act_flat)
    return size_sorted_indices[nlargest],layer

testn,testl=get_neuron(neuron_activation_list=test2,layer=2,nlargest=-1)
print(testn,testl)

def get_image_indices(image_activations,images,neuron_index,layer):
    #flatten conv layers - can turn off if you need it
    layer_act=image_activations[layer]
    if len(layer_act.shape)>2:
        layer_act_flat=np.array([layer_act[i].flatten() for i in range(len(layer_act))])
        #print(np.array(test2).shape)
    else:
        layer_act_flat=layer_act
    print(layer_act_flat.shape)
    images_for_neuron=layer_act_flat[:,neuron_index]
    image_indices=np.argsort(images_for_neuron)


    return images_for_neuron,image_indices

testi,testg=get_image_indices(image_activations=testdist,images=X_test,neuron_index=testn,layer=2)









def single_neuron_vec(image_activations_dist):
    neuron_vector=[]
    average_activation=[]
    variance=[]
    energy=[]
    # Quantities calculated over all images
    for layer in range(len(image_activations_dist)):
        average_activation.append(sum(image_activations_dist,axis=0)/image_activations_dist.shape[0])
        variance.append(np.sum(np.square(image_activations_dist),axis=0)/len(image_activations_dist.shape[0])-np.square(average_activation))
    return

def image_values(images,func):
    image_quantity=[]
    for i in range(images.shape[0]):#Assumes axis 0 is the image axis
        value=func(images[i,:,:])
        image_quantity.append(value)
    return np.array(image_quantity)

list_of_image_functions=[compute_energy,get_ccs]
#Let's form an array of image values:

# energies=[compute_energy(i) for i in X_test]
# largest_cc=[get_ccs(i) for i in X_test]


print(f"grok_folder: {model_folder}")
#epind=int(19900/100)
epind=-1
test2,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='var',image_set=example_images)#Note that this gives me average ordered
testdist,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='dist',image_set=example_images)


def visualize_images(image_list,image_indices,image_activations,nlargest):
    rowcount=10
    depthcount=3
    image_positions=[int(x) for x in np.linspace(depthcount,len(image_list)-1,rowcount)]
    image_indices=[[image_indices[x-i] for i in range(depthcount)] for x in image_positions]
    labels=[y_test[x] for x in image_positions]
    #print(image_indices)
    image_acts=[image_activations[x] for x in image_indices]
    #print(image_acts)
    #Now just visualize the images

    fig, ax = plt.subplots(depthcount,rowcount,figsize=(20,5))
    
    for j in range(depthcount):
        for i in range(rowcount):
            a=round(image_acts[i][j],2)
            ax[j,i].spy(X_test[image_indices[i][j]] + 1 )
            energy=compute_energy(X_test[image_indices[i][j]])
            energy=round(energy.item(),3)
            ax[j,i].set_title(f'Act: {str(a)}')#, , l{y_test[image_indices[i]]},#f'Act: {str(a)}'
            ax[j,i].set_xlabel(f'T-Tc: {str(np.round((my_y_temp[image_indices[i][j]]-2.69).numpy(),2))}, En: {energy}')#mag: {sum(torch.flatten(X_test[image_indices[i]]))}
            ax[j,i].set_yticklabels([])
            ax[j,i].set_xticklabels([])
            if nlargest<0:
                number=-nlargest
                text='th most activated neuron'
            else:
                number=nlargest+1
                text='th least activated neuron'
    fig.suptitle(f'{number}{text}')
    fig.tight_layout()
    fig.savefig(str(root)+f'/{number}{text}_epoch_{str(int(epind*20))}_quiltgraphs')
    fig.subplots_adjust(top=0.88)


    #plt.spy(configs[50]+1)
    #plt.show()

def visualize_image_values(image_values,image_activations,func_labels,nlargest):
    fig, ax=plt.subplots(1,len(image_values),figsize=(25,10))
    
    if len(image_values)==1:
        ax.scatter(image_values,image_activations)
        ax.set_title('Something')
        ax.set_xlabel('Values')
        ax.set_ylabel('Activations')
    else:
        for i in range(len(image_values)):
            ax[i].scatter(image_values[i],image_activations)
            ax[i].set_title(func_labels[i])
            ax[i].set_xlabel(func_labels[i])
            ax[i].set_ylabel('Activations')
    if nlargest<0:
        number=-nlargest
        text='th most activated neuron'
    else:
        number=nlargest+1
        text='th least activated neuron'
    fig.suptitle(f'{number}{text}'+' Image-Activation graphs')
    fig.tight_layout()
    fig.savefig(str(root)+f'/{number}{text}_epoch_{str(int(epind*20))}_corrgraphs')
    #plt.show()


    return None

testn,testl=get_neuron(neuron_activation_list=test2,layer=2,nlargest=-1)
def image_graphs(funcs,image_set,image_activations,image_indices,func_labels,nlargest):
    all_image_vals=[]
    new_acts=[image_activations[image_indices[i]]for i in range(len(image_indices))]
    for func in funcs:
        image_vals=[func(image_set[image_indices[i]]) for i in range(len(image_indices))]
        all_image_vals.append(image_vals)
    viz_all=visualize_image_values(all_image_vals,new_acts,func_labels,nlargest=nlargest)
    return image_vals

#testv1=visualize_images(image_list=X_test,image_indices=testg,image_activations=testi,nlargest=-1)
# testv2=visualize_images(image_list=X_test,image_indices=testg,image_activations=testi)


np.array(X_test[2].detach().cpu().numpy())

def get_quilt(image_activations,images,neuron_index,layer,global_funcs,func_labels):
    testn,testl=get_neuron(neuron_activation_list=test2,layer=layer,nlargest=neuron_index)
    print(testn)
    image_activations,image_indices=get_image_indices(image_activations=image_activations,images=images,neuron_index=testn,layer=layer)
    viz=visualize_images(image_list=images,image_indices=image_indices,image_activations=image_activations,nlargest=neuron_index)
    viz_all=image_graphs(funcs=global_funcs,image_set=images,image_activations=image_activations,image_indices=image_indices,func_labels=func_labels,nlargest=neuron_index)



get_quilt(image_activations=testdist,images=X_test,neuron_index=-1,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#get_quilt(image_activations=testdist,images=X_test,neuron_index=-3,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#get_quilt(image_activations=testdist,images=X_test,neuron_index=-4,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#get_quilt(image_activations=testdist,images=X_test,neuron_index=-5,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])


get_quilt(image_activations=testdist,images=X_test,neuron_index=-2,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
get_quilt(image_activations=testdist,images=X_test,neuron_index=-3,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
get_quilt(image_activations=testdist,images=X_test,neuron_index=-4,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
get_quilt(image_activations=testdist,images=X_test,neuron_index=-5,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
get_quilt(image_activations=testdist,images=X_test,neuron_index=-10,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#get_quilt(image_activations=testdist,images=X_test,neuron_index=-20,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#get_quilt(image_activations=testdist,images=X_test,neuron_index=0,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])

#Save user defined variables
import dill



# Function to filter user-defined variables
def filter_user_defined_variables(globals_dict):
    user_defined_vars = {}
    for name, value in globals_dict.items():
        # Check if the variable is not a built-in or imported module/function
        if not name.startswith('_') and not hasattr(value, '__module__'):
            user_defined_vars[name] = value
    return user_defined_vars

# Filter the user-defined variables
user_vars = filter_user_defined_variables(globals())

# Save only user-defined variablese
filename = str(root)+'/variables.p'
with open(filename, 'wb') as file:
    dill.dump(user_vars, file)
