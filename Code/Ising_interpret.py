conv_channels=[2,4]
hiddenlayers=[20]
import os
from timeit import default_timer as timer
import time
import pickle
#from tqdm.auto import tqdm # for loop progress bar
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
import dill


# with open('/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/ClusterResults/Ising_data_seed_0_train_5_grok_True_wd_0.2time_1707074792/variables.p','rb') as f:
#     all_var=dill.load(f)

# print(all_var.keys())




device='cpu'




grok_model_folder_name='../NewResults/Ising_seed_0_train_5_grok_True_wd_0.08time_1707022296/models_cnn_[2, 4]_hw[20]_time1707022297'
# with open(grok_model_folder_name,'rb') as f:
#     grok_model_folder=pickle.loaf(f)


nogrok_model_folder_name='../NewResults/Ising_seed_0_train_5_grok_False_fixed_wd_0.001time_1707023134/models_cnn_[2, 4]_hw[20]_time1707023134'


# with open(nogrok_model_folder_name,'rb') as f:
#     nogrok_model_folder=pickle.loaf(f)


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

model=CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
                activation=nn.ReLU(), optimizer=torch.optim.Adam,
                learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)




def perform_svd_on_kernels(conv_layer):
    """
    Performs Singular Value Decomposition (SVD) on the kernels of a given Conv2d layer.
    
    Parameters:
    - conv_layer: An instance of torch.nn.Conv2d
    
    Returns:
    - U, S, V: Matrices from the SVD decomposition.
    """
    # Ensure the layer is an instance of nn.Conv2d
    if not isinstance(conv_layer, nn.Conv2d):
        raise ValueError("The layer must be an instance of nn.Conv2d")
    
    # Detach the kernel weights from the computation graph and convert to numpy
    kernels = conv_layer.weight.detach().cpu().numpy()
    
    # Reshape the kernels into a 2D matrix for SVD
    out_channels, in_channels, kernel_height, kernel_width = kernels.shape
    kernels_reshaped = kernels.reshape(out_channels, in_channels * kernel_height * kernel_width)
    
    # Perform SVD
    U, S, V = np.linalg.svd(kernels_reshaped, full_matrices=False)
    return U, S, V

# Example usage:
# Create a Conv2d layer
# conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3)

# # Perform SVD on the kernels of the Conv2d layer
# U, S, V = perform_svd_on_kernels(conv_layer)





def extract_singular_values(folder_name,epoch):
	onlyfiles = [f for f in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, f)) if f!='init.pth']
	full_run_data = torch.load(folder_name+f'/{onlyfiles[epoch]}')

	model=CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
                activation=nn.ReLU(), optimizer=torch.optim.Adam,
                learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)
	model.to(device)
	model.load_state_dict(full_run_data['model'])


	model.to(device)


	weight_matrices=[]
	for name,param in model.named_parameters():
		if 'weight' in name:
			weight_matrices.append(param)
	
	u2,s2,v2=torch.svd(weight_matrices[-2]-torch.mean(weight_matrices[-2]))#mean-centering
	print(weight_matrices[-2].shape)

	conv_layer=weight_matrices[1]
	kernels = conv_layer.detach().cpu().numpy()
    
    # Reshape the kernels into a 2D matrix for SVD
	out_channels, in_channels, kernel_height, kernel_width = kernels.shape
	kernels_reshaped = kernels.reshape(out_channels, in_channels * kernel_height * kernel_width)
    
    # Perform SVD
	uk, sk, vk = np.linalg.svd(kernels_reshaped, full_matrices=False)
	# return U, S, V
	
    #Can I find the inner product of the 
	return u2.detach().numpy(),s2.detach().numpy(), v2.detach().numpy(),weight_matrices,uk,sk,vk,kernels_reshaped
        
#plt.scatter(range(len(s1)),s1.detach().numpy(),label='Fourth Layer')

u_grok,sv_grok,v_grok,grok_weights,uk_grok,sk_grok,vk_grok,reshaped_grok=extract_singular_values(folder_name=grok_model_folder_name,epoch=-1)
u_nogrok,sv_nogrok,v_nogrok,nogrok_weights,uk_nogrok,sk_nogrok,vk_nogrok,reshaped_nogrok=extract_singular_values(folder_name=nogrok_model_folder_name,epoch=-1)

grok_mean=torch.mean(grok_weights[-1])

nogrok_mean=torch.mean(nogrok_weights[-2])
u_rand,sv_rand,v_rand=torch.svd(grok_mean.item()*torch.rand(grok_weights[-2].shape))
norms = grok_weights[-2].norm(p=2, dim=1, keepdim=True)
grok_normed_weights=grok_weights[-2]/norms
print(f'normed weights {grok_normed_weights.shape}')
print(f' weights {grok_weights[-2].shape}')

projected_matrix_grok=torch.matmul(grok_normed_weights, torch.tensor(v_grok[:, :2]))

norms = nogrok_weights[-2].norm(p=2, dim=1, keepdim=True)
nogrok_normed_weights=nogrok_weights[-2]/norms

projected_matrix_nogrok=torch.matmul(nogrok_normed_weights, torch.tensor(v_nogrok[:, :2])) 



print(grok_mean,nogrok_mean)
print(f'projectors equal: {torch.allclose(projected_matrix_grok,projected_matrix_nogrok)}')

# print(torch.nonzero(projected_matrix_grok.view(-1).data).squeeze())
# print(torch.nonzero(projected_matrix_nogrok.view(-1).data).squeeze())
print(torch.sum(torch.isclose(projected_matrix_grok,torch.zeros(projected_matrix_grok.shape),atol=10**-4).view(-1).data.squeeze()))
print(torch.sum(torch.isclose(projected_matrix_nogrok,torch.zeros(projected_matrix_nogrok.shape),atol=10**-4).view(-1).data.squeeze()))



cosine_similarity_left=np.sum(np.multiply(u_grok,u_nogrok),axis=1)
cosine_similarity_right=np.sum(np.multiply(v_grok,v_nogrok),axis=1)
print(cosine_similarity_right.shape)
# print(u_grok.shape)
# print(cosine_similarity_left.shape)
# print(np.sum([[1,2,3],[4,5,6]],axis=1).shape)
#Plot the SVD'd matrix:
# fig=make_subplots(rows=1,cols=2,subplot_titles=['Grok','No Grok'])

# fig.add_trace(go.Heatmap(z=np.abs(v_grok)/(torch.norm(v_grok,dtype=torch.float)),
# 						# x=len(v_grok[:,0]),
# 						# y=len(v_grok[0,:]),
# 			 		),
# 					row=1,
# 					col=1)

# fig.add_trace(go.Heatmap(z=np.abs(v_nogrok)/(torch.norm(v_nogrok,dtype=torch.float)),
# 						# x=len(v_nogrok[:,0]),
# 						# y=len(v_nogrok[0,:])			 
# 			 		),
# 					row=1,
# 					col=2)
# #fig.show()


fig=make_subplots(rows=1,cols=2,subplot_titles=['Singular values MLP','SV CNN','SV MLP No Grok','SV CNN No Grok'])#Will later put in cosine similarity

fig.add_trace(go.Scatter(x=np.array(range(len(sv_grok))),
				y=sv_grok,
				name='Grok'),
				row=1,
				col=1)

fig.add_trace(go.Scatter(x=np.array(range(len(sk_grok))),
				y=sk_grok,
				name='Grok'),
				row=1,
				col=2)
fig.add_trace(go.Scatter(x=np.array(range(len(sv_nogrok))),
				y=sv_nogrok,
				name='No grok'),
				row=1,
				col=1)
fig.add_trace(go.Scatter(x=np.array(range(len(sk_nogrok))),
				y=sk_nogrok,
				name='No grok'),
				row=1,
				col=2)
# fig.add_trace(go.Scatter(x=np.array(range(len(sv_rand))),
# 				y=sv_rand,
# 				name='Random'),
# 				row=1,
# 				col=1)


# fig.add_trace(go.Scatter(x=np.array(range(len(cosine_similarity_left))),
# 				y=cosine_similarity_left,
# 				name='Cos Left'),
# 				row=1,
# 				col=2)
# fig.add_trace(go.Scatter(x=np.array(range(len(cosine_similarity_right))),
# 				y=cosine_similarity_right,
# 				name='Cos Right'),
# 				row=1,
# 				col=3)
fig.update_yaxes(type="log", row=1, col=1)

#fig.show()


#fig=make_subplots(rows=1,cols=2,subplot_titles=['Grok First Two PCA','No Grok First Two PCA'],shared_xaxes=True,shared_yaxes=True)
numpy_projected_grok=projected_matrix_grok.detach().numpy()
numpy_projected_nogrok=projected_matrix_nogrok.detach().numpy()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assuming you have your numpy_projected_grok and numpy_projected_nogrok arrays ready
norms = torch.tensor(reshaped_grok).norm(p=2, dim=1, keepdim=True)
grok_normed_weights_k=reshaped_grok/norms

projected_matrix_grok_k=torch.matmul(np.transpose(grok_normed_weights_k), torch.tensor(vk_grok[:, :2]))
projected_matrix_grok_k.detach().numpy()

norms_ng = torch.tensor(reshaped_nogrok).norm(p=2, dim=1, keepdim=True)
nogrok_normed_weights_k=reshaped_nogrok/norms_ng

projected_matrix_nogrok_k=torch.matmul(np.transpose(nogrok_normed_weights_k), torch.tensor(vk_nogrok[:, :2]))
projected_matrix_nogrok_k.detach().numpy()

##################Try to implement PCA
layer=2
cnn_kernel=grok_weights[layer].detach().numpy()
cnn_kernel2=nogrok_weights[layer].detach().numpy()
print(len(cnn_kernel.shape))
print(cnn_kernel.shape)

u_g,s_g,v_g=np.linalg.svd(cnn_kernel,compute_uv=True)
u_ng,s_ng,v_ng=np.linalg.svd(cnn_kernel2,compute_uv=True)

def pois(sv):
	sv_space=[sv[i]-sv[i-1] for i in range(1,len(sv))]
	rs=[np.min(sv_space[i]+sv_space[i])/np.max(sv[i]+sv[i+1]) for i in range(len(sv)-1)]
	return rs

a=pois(s_g)
b=pois(s_ng)

# plt.scatter(list(range(len(a))),a)
# plt.scatter(list(range(len(b))),b)
# plt.show()
# exit()

print(a,b)


print(u_g.shape)
print(s_g.shape)
axis_norm = np.linalg.norm(cnn_kernel, axis=len(cnn_kernel.shape)-1, keepdims=True)#This is L2 norm, not sure if this or the L1 norm is more appropriate
cnn_kernel_normed=cnn_kernel/axis_norm
print(axis_norm)

axis_norm2 = np.linalg.norm(cnn_kernel2, axis=len(cnn_kernel.shape)-1, keepdims=True)#This is L2 norm, not sure if this or the L1 norm is more appropriate
print(axis_norm2)
cnn_kernel_normed2=cnn_kernel2/axis_norm2


fig=make_subplots(rows=1,cols=2,subplot_titles=['Grok','No Grok'])

fig.add_trace(go.Heatmap(z=np.abs(cnn_kernel)/np.max(cnn_kernel),
						# x=len(v_grok[:,0]),
						# y=len(v_grok[0,:]),
			 		),
					row=1,
					col=1)

fig.add_trace(go.Heatmap(z=np.abs(cnn_kernel2)/np.max(cnn_kernel),
						# x=len(v_nogrok[:,0]),
						# y=len(v_nogrok[0,:])			 
			 		),
					row=1,
					col=2)
fig.show()


# t0=np.matmul(cnn_kernel,v_g[:,:,:,:2])
# t1=np.flip(np.sort(np.ndarray.flatten(t0[:,:,:,0])))
# t2=np.flip(np.sort(np.ndarray.flatten(t0[:,:,:,1])))

if len(cnn_kernel.shape)==4:
	t0=np.matmul(cnn_kernel,v_ng[:,:,:,:2])
	t1=np.flip(np.sort(np.ndarray.flatten(t0[:,:,:,0])))
	t2=np.flip(np.sort(np.ndarray.flatten(t0[:,:,:,1])))

	t01=np.matmul(cnn_kernel2,v_ng[:,:2])
	t11=np.flip(np.sort(np.ndarray.flatten(t01[:,:,:,0])))
	t21=np.flip(np.sort(np.ndarray.flatten(t01[:,:,:,1])))

	t02=np.matmul(cnn_kernel_normed,v_g[:,:,:,:2])
	t12=np.flip(np.sort(np.ndarray.flatten(t02[:,:,:,0])))
	t22=np.flip(np.sort(np.ndarray.flatten(t02[:,:,:,1])))

	t03=np.matmul(cnn_kernel_normed2,v_ng[:,:,:,:2])
	t13=np.flip(np.sort(np.ndarray.flatten(t03[:,:,:,0])))
	t23=np.flip(np.sort(np.ndarray.flatten(t03[:,:,:,1])))
elif len(cnn_kernel.shape)==2:
	t0=np.matmul(cnn_kernel,v_ng[:,:2])
	t1=np.flip(np.sort(np.ndarray.flatten(t0[:,0])))
	t2=np.flip(np.sort(np.ndarray.flatten(t0[:,1])))

	t01=np.matmul(cnn_kernel2,v_ng[:,:2])
	t11=np.flip(np.sort(np.ndarray.flatten(t01[:,0])))
	t21=np.flip(np.sort(np.ndarray.flatten(t01[:,1])))

	t02=np.matmul(cnn_kernel_normed,v_g[:,:2])
	t12=np.flip(np.sort(np.ndarray.flatten(t02[:,0])))
	t22=np.flip(np.sort(np.ndarray.flatten(t02[:,1])))

	t03=np.matmul(cnn_kernel_normed2,v_ng[:,:2])
	t13=np.flip(np.sort(np.ndarray.flatten(t03[:,0])))
	t23=np.flip(np.sort(np.ndarray.flatten(t03[:,1])))

fig=make_subplots(rows=2,cols=2,subplot_titles=[f'Unnormed Grok Layer {layer}',f'Unnormed Learn Layer {layer}',f'Normed Grok Layer {layer}',f'Normed Learn Layer {layer}'])
fig.add_trace(go.Scatter(x=t1, y=t2, name='Grok', mode='markers', marker=dict(size=10, color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=t11, y=t21, name='No Grok', mode='markers', marker=dict(size=10, color='blue')), row=1, col=2)
fig.add_trace(go.Scatter(x=t12, y=t22, name='Grok', mode='markers',marker=dict(size=10,color='red')), row=2, col=1)
fig.add_trace(go.Scatter(x=t13, y=t23, name='No Grok', mode='markers',marker=dict(size=10,color='blue')), row=2, col=2)


#fig.show()



########Try to use paper code


cnn_kernel=grok_weights[0].detach().numpy()
cnn_kernel2=nogrok_weights[0].detach().numpy()
print(cnn_kernel.shape)
sg=np.linalg.svd(cnn_kernel,compute_uv=False)
s_ng=np.linalg.svd(cnn_kernel2,compute_uv=False)

fig=make_subplots(rows=1,cols=len(grok_weights),subplot_titles=[f'Layer {i}' for i in range(len(grok_weights))])
for i in range(len(grok_weights)):
	sg=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(grok_weights[i].detach().numpy(),compute_uv=False))))
	s_ng=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(nogrok_weights[i].detach().numpy(),compute_uv=False))))
	print(sg.shape)
	fig.add_trace(go.Scatter(x=list(range(len(sg))),y=sg, name='Grok',mode='lines',line=dict(width=2,color='red')), row=1, col=i+1)
	fig.add_trace(go.Scatter(x=list(range(len(s_ng))),y=s_ng, name='No grok',mode='lines',line=dict(width=2,color='blue',dash='dash')), row=1, col=i+1)
	fig.update_yaxes(type="log", row=1, col=i+1)
	fig.update_xaxes(title_text="Order", row=1, col=i+1)
	fig.update_yaxes(title_text="SV", row=1, col=i+1)

#fig.show()


# def SingularValues(kernel, input_shape):
# 	transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
# 	return np.linalg.svd(transforms, compute_uv=False)
# print('test')
# s_t=SingularValues(kernel=cnn_kernel,input_shape=(16,16))
# print(len(s_t[0]))
# exit()

# Create subplots with shared axes (if necessary, based on your previous questions)
fig = make_subplots(rows=2, cols=2, subplot_titles=['Grok MLP PCA','No Grok MLP PCA','Grok CNN PCA','No Grok CNN PCA'],shared_xaxes=True, shared_yaxes=True)

# Add traces
fig.add_trace(go.Scatter(x=numpy_projected_grok[:,0], y=numpy_projected_grok[:,1], name='Grok', mode='markers',marker=dict(size=10)), row=1, col=1)
fig.add_trace(go.Scatter(x=numpy_projected_nogrok[:,0], y=numpy_projected_nogrok[:,1], name='No Grok', mode='markers',marker=dict(size=10)), row=1, col=2)
fig.add_trace(go.Scatter(x=projected_matrix_grok_k[:,0], y=projected_matrix_grok_k[:,1], name='Grok CNN', mode='markers',marker=dict(size=10)), row=2, col=1)
fig.add_trace(go.Scatter(x=projected_matrix_nogrok_k[:,0], y=projected_matrix_nogrok_k[:,1], name='No Grok CNN', mode='markers',marker=dict(size=10)), row=2, col=2)
# Update x and y axis labels correctly for each subplot
fig.update_xaxes(title_text="Principal Component 1", row=1, col=1)
fig.update_xaxes(title_text="Principal Component 1", row=1, col=2)
fig.update_xaxes(title_text="Principal Component 1", row=2, col=1)
fig.update_xaxes(title_text="Principal Component 1", row=2, col=2)
fig.update_yaxes(title_text="Principal Component 2", row=1, col=1)
fig.update_yaxes(title_text="Principal Component 2", row=2, col=1)
#fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)



# Show figure
fig.show()
# fig=make_subplots(rows=2,cols=1)#subplot_titles=2*[f'Layer {i+1}' for i in range(len(grok_weights))#Will later put in cosine similarity


# for col in range(2,3):
#     #with torch.no_grad():
#     weights=grok_weights[col].flatten()
    
#     fig.add_trace(go.Histogram(x=grok_weights[col].detach().numpy(),nbinsx=50),
#                     row=1,
#                     col=1)
#     weights=nogrok_weights[col].flatten()
#     fig.add_trace(go.Histogram(x=weights[col].detach().numpy(),nbinsx=50,
#                     name='No Grok'),
#                     row=2,
#                     col=1)





# fig.show()
print('stuff')

fig, axs=plt.subplots(2,len(grok_weights))

for i in range(len(grok_weights)):
	print(i)

	axs[0,i].hist(grok_weights[i].detach().flatten().numpy()/np.max(grok_weights[i].detach().numpy()))
	axs[0,i].set_title('Grok')
	axs[0,i].grid(True)
	axs[1,i].hist(nogrok_weights[i].detach().flatten().numpy()/np.max(nogrok_weights[i].detach().numpy()))
	axs[1,i].set_title('No Grok')
	
	axs[1,i].grid(True)
plt.show()