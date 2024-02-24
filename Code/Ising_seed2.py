import os
import sys
import dill
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
import copy
#import kaleido

################Seeds
from data_objects import seed_average_onerun






def create_ising_dataset(data_seed,train_size,test_size):
    random.seed(data_seed)
    for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
        set_seed(data_seed)
    
    with open("../Data/IsingML_L16_traintest.pickle", "rb") as handle:
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



    # desc='data[1] is [[data_seed,sgd_seed,init_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]'
    # data_save1=[[data_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]
    # data_save=[desc,data_save1]
    return train_loader,test_loader

def create_model(init_seed):
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
    return model

def check_weights_update(initial_weights, updated_weights):
    return not torch.equal(initial_weights, updated_weights)

def train(epochs,initial_model,save_interval,train_loader,test_loader,sgd_seed,batch_size,one_run_object):
    start_time = timer()
    first_time_training = True
    epochs = epochs # how many runs through entire training data
    save_models=True
    save_interval=save_interval
    # fig,axs=plt.subplots(2,3)
    model=initial_model
    m1=model.state_dict()['fc_layers.0.weight']
    #plt.hist(np.ndarray.flatten(model.fc_layers[0].weight.data.detach().numpy()))
    
    initial_fc_weights=model.fc_layers[0].weight.data.clone()
    # axs[0,0].hist(np.ndarray.flatten(m1.detach().numpy()))
    # axs[1,0].hist(np.ndarray.flatten(initial_fc_weights.detach().numpy()))
    # print(initial_fc_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = model.optimizer
    random.seed(sgd_seed)
    for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
        set_seed(sgd_seed)
        
    if save_models:
        save_dict0 = {'model':model.state_dict()}#'train_data':train_data, 'test_data':test_data <-- I don't need this because I'll have this saved in the data. 
        one_run_object.models[0]=save_dict0
        #torch.save(save_dict, root+'/'+run_name+'/'+'/0.pth')

    if  first_time_training == True:
        train_losses = []
        test_losses = []
        train_accuracy = []
        test_accuracy = []
    else:
        print("Starting additional training")
        epochs = 2900
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
            # initial_fc_weights = model.fc_layers[0].weight.data.clone()
            # Update parameters
            
            optimizer.zero_grad() # clears old gradients stored in previous step
            train_loss.backward() # calculates gradient of loss function using backpropagation (stochastic)
            optimizer.step() # adjust parameters in direction of steepest descent
            # m2=model.state_dict()['fc_layers.0.weight']
            # if i>2000:
            #     print(model.fc_layers[0].weight.data)
                #plt.hist(np.ndarray.flatten(model.fc_layers[0].weight.data.detach().numpy()))
                # axs[0,1].hist(np.ndarray.flatten(m2.detach().numpy()))
                # axs[1,1].hist(np.ndarray.flatten(model.fc_layers[0].weight.data.detach().numpy()))

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
            # updated_fc_weights = model.fc_layers[0].weight.data
            # if check_weights_update(initial_fc_weights, updated_fc_weights):
            #     print("fc weights have been updated.")
            # else:
            #     print("fc weights have not been updated.")
            
            # print(save_dict0['model']['fc_layers.3.weight']==model.state_dict()['fc_layers.3.weight'])
            save_dict = {
                    'model': copy.deepcopy(model.state_dict()),
                    'optimizer': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'epoch': i,
                }
            one_run_object.models[i]=save_dict
            
            # if i>0:
            #     print(torch.equal(one_run_object.models[i]['model']['fc_layers.3.weight'],one_run_object.models[i-100]['model']['fc_layers.3.weight']))
            # if i>1000:
            #         fig,axs=plt.subplots(1,2)
            #         axs[0].hist(np.ndarray.flatten(one_run_object.models[i]['model']['fc_layers.0.weight'].detach().numpy()))
            #         axs[1].hist(np.ndarray.flatten(one_run_object.models[0]['model']['fc_layers.0.weight'].detach().numpy()))
            #         plt.show()
            #         exit()

        if i % 1000 == 0 or i == epochs-1:
            # Print interim result
            print(f"epoch: {i}, train loss: {train_loss.item():.4f}, accuracy: {train_accuracy[-1]:.4f}")
            print(12*" " + f"test loss: {test_loss.item():.4f}, accuracy: {test_accuracy[-1]:.4f}" )
            print(60*"*")

    print(f'\nDuration: {timer() - start_time:.0f} seconds') # print the time elapsed

    one_run_object.train_losses=train_losses
    one_run_object.test_losses=test_losses
    one_run_object.train_accuracies=train_accuracy
    one_run_object.test_accuracies=test_accuracy


    # data = ["data[1] is of form [train_loss, test_loss], [train_acc, test_acc]", [[train_losses, test_losses], [train_accuracy, test_accuracy]]]
    # with open(str(root)+'/'+f"Isinglosses_hw{hiddenlayers}_cnn_{conv_channels}_data_seed_{data_seed}_{int(time.time())}.p", "wb") as handle:
    #     pickle.dump(data, handle)
    # print("data saved")

    


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















#I don't want to do the analysis here, this script is only meant to generate the objects which are storing the data which will then be analyzed.
#run_folder='models_seed0_cnn_[2, 4]_hw[30]_time1706744598'

# def stitch_losscurve(run_folder):
#     train_losses=[]
#     test_losses=[]
#     train_accs=[]
#     test_accs=[]

#     onlyfiles = [f for f in os.listdir(run_folder) if 'losses' in f]
#     def sort_by_time(filestring):
#         print((filestring.split(f'data_seed_{data_seed}_')[-1]))
#         return int((filestring.split(f'data_seed_{data_seed}_')[-1]).split('.')[0])

#     onlyfiles.sort(key=sort_by_time)
#     print(onlyfiles)


#     for run in onlyfiles:
#         #with open(str(root)+'/'+str(run_name)+'/'+run, "rb") as f:
#         with open(str(root)+'/'+run, "rb") as f:
#             losses_data=pickle.load(f)
#         print(losses_data[0])


#         train_losses=train_losses+losses_data[1][0][0]
#         test_losses=test_losses+losses_data[1][0][1]
#         train_accs=train_accs+losses_data[1][1][0]
#         test_accs=test_accs+losses_data[1][1][1]
#     return [train_losses,test_losses],[train_accs,test_accs]

# [train_losses1,test_losses1],[train_accs,test_accs]=stitch_losscurve(run_folder)


# fig,axs=plt.subplots(2)
# axs[0].title.set_text('Losses')
# axs[0].set_xlabel('Epoch')
# axs[0].set_ylabel('Loss')
# axs[0].plot(list(range(epochs)),train_losses1,label='Train')
# axs[0].plot(list(range(epochs)),test_losses1,label='Test')

# axs[1].title.set_text('Accuracy')
# axs[1].set_xlabel('Epoch')
# axs[1].set_ylabel('Accuracy')
# axs[1].plot(list(range(epochs)),train_accs,label='Train')
# axs[1].plot(list(range(epochs)),test_accs,label='Test')
# fig.savefig(str(root)+'/Losscurve.png')

if __name__ == '__main__':

    cluster_array=0
         
    data_seed_start=int(sys.argv[1+cluster_array])
    data_seed_end=int(sys.argv[2+cluster_array])
    data_seeds=[i for i in range(data_seed_start,data_seed_end)]
    print(f' data_seeds={data_seeds}')

    sgd_seed_start=int(sys.argv[3+cluster_array])
    sgd_seed_end=int(sys.argv[4+cluster_array])
    sgd_seeds=[i for i in range(sgd_seed_start,sgd_seed_end)]

    init_seed_start=int(sys.argv[5+cluster_array])
    init_seed_end=int(sys.argv[6+cluster_array])
    init_seeds=[i for i in range(init_seed_start,init_seed_end)]

    weight_decay=int(sys.argv[7+cluster_array])/100
    grok_str=str(sys.argv[8+cluster_array])
    train_size=int(str(sys.argv[9+cluster_array]))
    print(f'train_size: {train_size}')

    print(f" seeds: {data_seeds}, sgd_seeds: {sgd_seeds},init_seeds {init_seeds}")
    print(f" wd: {weight_decay}")
    print(f" grok_str: {grok_str}")

    if grok_str=='True':
        grok=True
    elif grok_str=='False':
        grok=False

    #################Params
    train_size=train_size
    test_size=1000
    #grok_locations=[0,100]#

    if grok:
        learning_rate=10**-4
        weight_decay=weight_decay
        weight_multiplier=10

    else:
        learning_rate=10**-4
        weight_decay=0.01
        weight_multiplier=1
    

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


    #Train params
    epochs=50000
    save_interval=100
    # set torch data type and random seeds
    torch.set_default_dtype(dtype)


    
    root=f'../NewCluster/grok_{grok_str}_time_{int(time.time())}'
    os.mkdir(root)
    print(str(root))
    for i in range(len(data_seeds)):
        data_seed=data_seeds[i]
        sgd_seed=sgd_seeds[i]
        init_seed=init_seeds[i]
        params_dic={'weight_decay':weight_decay,'weight_multiplier':weight_multiplier,'learning_rate':learning_rate,'hidden_layers':hiddenlayers,'conv_channels':conv_channels,'train_size':train_size,'test_size':test_size,'dropout_p':dropout_prob}
        save_object=seed_average_onerun(data_seed=data_seed,sgd_seed=sgd_seed,init_seed=init_seed,params_dic=params_dic)
        save_object.start_time=int(time.time())
        train_loader,test_loader=create_ising_dataset(data_seed=data_seed,train_size=train_size,test_size=test_size)
        save_object.train_loader=train_loader
        save_object.test_loader=test_loader
        model=create_model(init_seed=init_seed)
        train(epochs=epochs,initial_model=model,save_interval=save_interval,train_loader=train_loader,test_loader=test_loader,sgd_seed=sgd_seed,batch_size=train_size,one_run_object=save_object)
        
        save_name=f'data_seed_{data_seed}_time_{int(time.time())}'
        run_folder=str(root)
        with open(str(root)+"/"+save_name, "wb") as dill_file:
            dill.dump(save_object, dill_file)    
    # fig,axs=plt.subplots(1,2)
    # axs[0].hist(np.ndarray.flatten(save_object.models[0]['model']['fc_layers.0.weight'].detach().numpy()))
    # axs[1].hist(np.ndarray.flatten(save_object.models[1900]['model']['fc_layers.0.weight'].detach().numpy()))
    # plt.show()
    exit()





