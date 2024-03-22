
import os
import sys
import dill


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

class seed_average_onerun():
    def __init__(self,data_seed,sgd_seed,init_seed,params_dic):
        self.data_seed=data_seed
        self.sgd_seed=sgd_seed
        self.init_seed=init_seed
        self.params_dic=params_dic#Weight decay, learning rate etc...
        # params_dic={'weight_decay':weight_decay,'weight_multiplier':weight_multiplier,'learning_rate':learning_rate,'hidden_layers':hiddenlayers,'conv_channels':conv_channels,'train_size':train_size,'test_size':test_size,'dropout_p':dropout_prob}
        self.models={} #You'll save your models here
        self.train_losses=None
        self.test_losses=None
        self.train_accuracies=None
        self.test_accuracies=None
        self.train_loader=None
        self.test_loader=None
        self.start_time=None
        self.weights=None
        self.weightshistfig=None
        self.losscurvesfig=None
        self.svdata=None
        self.svdfig=None
        self.pcadata=None#I think I'll probably have a dictionary with all of the objects I need to calculate these
        self.pcafig=None
        self.neuroncorrs=None #Will be a dictionary with epoch, neuron indices as the objects.
        self.trainargs=None
    #Now I want to write scripts for the analysis function.
    
    def make_loss_curves(self):
        fig = make_subplots(rows=1, cols=2)
        train_losses=self.train_losses
        test_losses=self.test_losses
        train_accuracies=self.train_accuracies
        test_accuracies=self.test_accuracies

        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_losses))),
                y=train_losses,
                name='Train losses'
            ),
            row=1,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(test_losses))),
                y=test_losses,
                name='Test losses'
            ),
            row=1,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_accuracies))),
                y=train_accuracies,
                name='Train Accuracy'
            ),
            row=1,
            col=2)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(test_accuracies))),
                y=test_accuracies,
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
            title=f'CNN seeds {self.data_seed,self.sgd_seed,self.init_seed}',#conv_layers{str(single_data_obj.params_dic['conv_channels'])}',# MLP {str(single_data_obj.params_dic['hidden_layers'])}, lr={str(single_data_obj.params_dic['learning_rate'])}, wd={str(single_data_obj.params_dic['weight_decay'])}, wm={str(single_data_obj.params_dic['weight_multiplier'])}, train size={str(single_data_obj.params_dic['train_size'])}, test size={str(single_data_obj.params_dic['test_size'])}',
            xaxis_title='Epochs',
            # yaxis_title='Test accuracy',
            showlegend=True,
        )

        return fig#.write_image(root+"/Losses_plotly.png")
    def model_epochs(self):
        return list(self.models.keys())
    def make_weights_histogram(self,non_grokked_object,epoch):
        #last_epoch=max(self.models.keys())
        weights_grok=[]
        grok_state_dic=self.models[epoch]['model']
        
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        weights_nogrok=[]
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[grok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
        titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]


        fig=make_subplots(rows=2,cols=len(weights_grok)+1,subplot_titles=titles)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)


        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        for i in range(len(weights_grok)):
            flattened_gw=torch.flatten(weights_grok[i]).detach().numpy()
            flattened_ngw=torch.flatten(weights_nogrok[i]).detach().numpy()
            showleg=False
            if i==0:
                showleg=True
            # fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+2)
            # fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=i+2)

            fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            fig.update_xaxes(title_text="Weight", row=1, col=i+1)
            fig.update_yaxes(title_text="Freq", row=1, col=i+1)
            fig.update_xaxes(title_text="Weight", row=2, col=i+1)
            fig.update_yaxes(title_text="Freq", row=2, col=i+1)
        
        return fig
    
    def make_weights_histogram2(self,non_grokked_object,epoch,fig):
        #last_epoch=max(self.models.keys())
        
        grok_state_dic=self.models[epoch]['model']
        
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.test_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

        if epoch==self.model_epochs()[0]:
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        for i in range(len(weights_grok)):
            flattened_gw=torch.flatten(weights_grok[i]).detach().numpy()
            flattened_ngw=torch.flatten(weights_nogrok[i]).detach().numpy()
            showleg=False
            if i==0:
                showleg=True
            fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+2)
            fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=i+2)

            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            if epoch==self.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=i+1)
                fig.update_yaxes(title_text="Freq", row=1, col=i+1)
                fig.update_xaxes(title_text="Weight", row=2, col=i+1)
                fig.update_yaxes(title_text="Freq", row=2, col=i+1)
        


    def weights_histogram_epochs(self,non_grokked_object):
        epochs=self.model_epochs()
        if (self.model_epochs()!=non_grokked_object.model_epochs()):
            print('Grok and Non-Grokked epochs are not the same!')
            print(f'Grokked epochs: {self.model_epochs()}')
            print(f'NG epochs: {non_grokked_object.model_epochs()}')
        
        print(epochs[0])
        fig=self.make_weights_histogram(non_grokked_object,epoch=epochs[0])
        def frame_data(epoch):
            print(f'epoch {epoch}')
            grok_state_dic=self.models[epoch]['model']
            weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
            
            
            nogrok_state_dic=non_grokked_object.models[epoch]['model']
            weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

            frame_list=[
            go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies),
            go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies),
            go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1]),
            go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies),
            go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies),
            go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1])]
            

            for i in range(len(weights_grok)):
            #     flattened_gw=torch.flatten(weights_grok[i]).detach().numpy()
            #     flattened_ngw=torch.flatten(weights_nogrok[i]).detach().numpy()
                frame_list.append(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'))  # Placeholder histogram
                frame_list.append(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'))  # Placeholder histogram

            return go.Frame(data=frame_list,name=f'Epoch_{epoch}')
        
        frames=[]
        for epoch in epochs:
            frames.append(frame_data(epoch))
        fig.frames=frames

        #Now you need to add the slider

        sliders = [dict(
            active=0,
            currentvalue=dict(font=dict(size=12), prefix='Epoch: ', visible=True),
            pad=dict(t=50),
            steps=[dict(method='update',
                        args=[{'visible': [False] * len(fig.data)},
                            {'title': f'Epoch {epoch}'}],  # This second dictionary in the args list is for layout updates.
                        label=f'Epoch {epoch}') for epoch in epochs]
        )]

        # Initially setting all data to invisible, then making the first set visible
        for i, _ in enumerate(fig.data):
            fig.data[i].visible = False
        if fig.data:  # Check if there is any data
            fig.data[0].visible = True  # Making the first trace visible

        fig.update_layout(sliders=sliders,title_text='slide plz')

    #     fig.update_layout(
    #     # height=600,
    #     # width=800,
    #     title_text="Animated Graph with Time Step Control",
    #     sliders=sliders
    # )

        # To animate, Plotly requires that the first frame is explicitly set

        fig.show()
    

    def weights_histogram_epochs2(self,non_grokked_object):
        epochs=self.model_epochs()[0::10]
        ng_epochs=non_grokked_object.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        grok_state_dic=self.models[epochs[0]]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_grok))]
        fig=make_subplots(rows=2,cols=len(weights_grok)+1,subplot_titles=titles)
        for epoch in epochs:
            self.make_weights_histogram2(non_grokked_object,epoch,fig)
        
        total_plots=6+2*(len(weights_grok))
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        fig.show()

    def svd_one_epoch(self,non_grokked_object,epoch,fig):
        grok_state_dic=self.models[epoch]['model']
        grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]

        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        if fig==None:
            fig=make_subplots(rows=1,cols=len(grok_weights)+1,subplot_titles=['Grok Accuracy']+['No Grok Accuracy']+[f'Layer {i}' for i in range(len(grok_weights)-1)])
        
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies, name='Grok Train',mode='lines',line=dict(width=2,color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies, name='Grok Test',mode='lines',line=dict(width=2,color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies, name='No Grok Train',mode='lines',line=dict(width=2,color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies, name='No Grok Test',mode='lines',line=dict(width=2,color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        for i in range(len(grok_weights)-1):
            sg=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(grok_weights[i].detach().numpy(),compute_uv=False))))
            s_ng=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(nogrok_weights[i].detach().numpy(),compute_uv=False))))
            fig.add_trace(go.Scatter(x=list(range(len(sg))),y=sg/sg[0], name='Grok',mode='lines',line=dict(width=2,color='red'),showlegend=False), row=1, col=i+3)
            fig.add_trace(go.Scatter(x=list(range(len(s_ng))),y=s_ng/s_ng[0], name='No grok',mode='lines',line=dict(width=2,color='blue',dash='dash'),showlegend=False), row=1, col=i+3)
            #fig.update_yaxes(type="log", row=1, col=i+1)
            fig.update_xaxes(title_text="Order", row=1, col=i+3)
            fig.update_yaxes(title_text="SV", row=1, col=i+3)
        return fig
    
    def svd_epochs(self,non_grokked_object):
        epochs=self.model_epochs()[0::3]
        ng_epochs=non_grokked_object.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        grok_state_dic=self.models[epochs[0]]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok Accuracy']+[f'Layer {i}' for i in range(len(weights_grok))]
        fig=make_subplots(rows=1,cols=len(weights_grok)+1,subplot_titles=titles)
        for epoch in epochs:
            self.svd_one_epoch(non_grokked_object,epoch,fig)
        
        total_plots=6+2*(len(weights_grok)-1)
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        return fig
    
    def pca_one_epoch(self,non_grokked_object,epoch,fig):
        grok_state_dic=self.models[epoch]['model']
        grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]

        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        u_grok,s_grok,v_grok=torch.svd(grok_weights[-2])
        u_nogrok,s_nogrok,v_nogrok=torch.svd(nogrok_weights[-2])

        norms = grok_weights[-2].norm(p=2, dim=1, keepdim=True)
        grok_normed_weights=grok_weights[-2]/norms
        ng_norms=nogrok_weights[-2].norm(p=2,dim=1,keepdim=True)
        nogrok_normed_weights=nogrok_weights[-2]/ng_norms

        grok_proj_matrix=torch.matmul(grok_weights[-2], torch.tensor(v_grok[:, :2]))
        grok_proj_matrix_normed=torch.matmul(grok_normed_weights, torch.tensor(v_grok[:, :2]))

        nogrok_proj_matrix=torch.matmul(nogrok_weights[-2], torch.tensor(v_nogrok[:, :2]))
        nogrok_proj_matrix_normed=torch.matmul(nogrok_normed_weights, torch.tensor(v_nogrok[:, :2]))

        # np.flip(np.sort(np.ndarray.flatten(t0[:,:,:,0])))



        if fig==None:
            fig = make_subplots(rows=2, cols=3, subplot_titles=['Grok Accuracy','Grok MLP','Grok Normed MLP','No Grok Accuracy','No Grok MLP', 'No Grok Normed MLP'])#shared_xaxes=True, shared_yaxes=True

        #Add traces for the loss curves
        
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies, name='Grok Train',mode='lines',line=dict(width=2,color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies, name='Grok Test',mode='lines',line=dict(width=2,color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies, name='No Grok Train',mode='lines',line=dict(width=2,color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies, name='No Grok Test',mode='lines',line=dict(width=2,color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)
        # Add traces
        fig.add_trace(go.Scatter(x=grok_proj_matrix[:,0].detach().numpy(), y=grok_proj_matrix[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=nogrok_proj_matrix[:,0].detach().numpy(), y=nogrok_proj_matrix[:,1].detach().numpy(), name='No Grok', mode='markers',marker=dict(size=10,color='blue'),showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=grok_proj_matrix_normed[:,0].detach().numpy(), y=grok_proj_matrix_normed[:,1].detach().numpy(), name='Grok CNN', mode='markers',marker=dict(size=10,color='red'),showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=nogrok_proj_matrix_normed[:,0].detach().numpy(), y=nogrok_proj_matrix_normed[:,1].detach().numpy(), name='No Grok CNN', mode='markers',marker=dict(size=10,color='blue'),showlegend=False), row=2, col=3)
        # Update x and y axis labels correctly for each subplot
        fig.update_xaxes(title_text="PC 1", row=1, col=1)
        fig.update_xaxes(title_text="PC 1", row=1, col=2)
        fig.update_xaxes(title_text="PC 1", row=2, col=1)
        fig.update_xaxes(title_text="PC 1", row=2, col=2)
        fig.update_yaxes(title_text="PC 2", row=1, col=1)
        fig.update_yaxes(title_text="PC 2", row=2, col=1)
        #fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)
        return fig
    
    def pca_epochs(self,non_grokked_object):
        epochs=self.model_epochs()[0::3]
        ng_epochs=non_grokked_object.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        # grok_state_dic=self.models[epochs[0]]['model']
        # weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok Accuracy','Grok MLP','Grok Normed MLP','No Grok Accuracy','No Grok MLP', 'No Grok Normed MLP']
        fig=make_subplots(rows=2,cols=3,subplot_titles=titles)
        for epoch in epochs:
            self.pca_one_epoch(non_grokked_object,epoch,fig)
        
        total_plots=6+(2*2)
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        return fig


#Object for storing all the runs within a given param
class seed_run_container():
    def __init__(self):
        # params_dic={'weight_decay':weight_decay,'weight_multiplier':weight_multiplier,'learning_rate':learning_rate,'hidden_layers':hiddenlayers,'conv_channels':conv_channels,'train_size':train_size,'test_size':test_size,'dropout_p':dropout_prob}
        self.params_dic={}#Rememeber that I want an object that holds different seeds of the same params
        self.runs_dic={}#
    
    def aggregate_runs(self,folder):
        onlyfiles = [f for f in os.listdir(folder) if 'time_' in f]
        #extract the trainargs object and the params_dic
        for runfile in onlyfiles:
            print(runfile)
            if os.path.getsize(str(folder)+'/'+runfile) > 0:  # Checks if the file is not empty
                with open(str(folder)+'/'+runfile, 'rb') as file:
                    try:
                        runobj = dill.load(file)
                        # Proceed with using runobj
                    except EOFError:
                        print("Failed to load the object. The file may be corrupted or incomplete.")
            else:
                print("The file is empty.")

            
            with open(str(folder)+'/'+runfile,'rb') as file:
                runobj=dill.load(file)
            keys_to_ignore=['data_seed','sgd_seed','init_seed']
            
            filtered_dict1 = {k: v for k, v in runobj.params_dic.items() if k not in keys_to_ignore}
            filtered_dict2 = {k: v for k, v in self.params_dic.items() if k not in keys_to_ignore}


            if filtered_dict1==filtered_dict2:
                key0=(runobj.data_seed,runobj.sgd_seed,runobj.init_seed)
                if key0 in self.runs_dic.keys():
                    if runobj==self.runs_dic[key0]:
                        pass
                    else:
                        print('duplicate non-identical runs!')
                else:
                    self.runs_dic[(runobj.data_seed,runobj.sgd_seed,runobj.init_seed)]=runobj
            
    def create_average_run(self,fixed_seeds):
        # self.models={} #You'll save your models here
        # self.train_losses=None
        # self.test_losses=None
        # self.train_accuracies=None
        # self.test_accuracies=None
        averaged_attributes=['train_losses','test_losses','train_accuracies','test_accuracies']
        if fixed_seeds==None:
            avg_run=seed_average_onerun(data_seed=None,sgd_seed=None,init_seed=None,params_dic=self.params_dic)
            
            for seed_key in self.runs_dic.keys():
                run_obj=self.runs_dic[seed_key]
                for attribute in averaged_attributes:
                    pass

        return None
    
    def make_weights_histogram2(self,non_grokked_container,epoch,fig):
        #last_epoch=max(self.models.keys())
        avg_weights_grok=np.array([])
        avg_weights_nogrok=np.array([])
        first=True
        for seed_key in self.runs_dic.keys():
            grokked_run=self.runs_dic[seed_key]
            grok_state_dic=grokked_run.models[epoch]['model']

            non_grokked_object=non_grokked_container.runs_dic[seed_key]
            nogrok_state_dic=non_grokked_object.models[epoch]['model']
            grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
            nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
            if first:
                avg_weights_grok=grok_weights
                avg_weights_nogrok=nogrok_weights
            else:
                avg_weights_grok=[avg_weights_grok[i]+grok_weights[i] for i in range(len(avg_weights_grok))]
                avg_weights_nogrok=[avg_weights_nogrok[i]+nogrok_weights[i] for i in range(len(avg_weights_nogrok))]
            
            
        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]
        avg_weights_grok=[avg_weights_grok[i]/(len(self.runs_dic.keys())) for i in range(len(avg_weights_grok))]
        avg_weights_nogrok=[avg_weights_nogrok[i]/(len(self.runs_dic.keys())) for i in range(len(avg_weights_nogrok))]
        avg_test_accuracies_grok=np.mean(np.array([np.array(self.runs_dic[seed_key].test_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_train_accuracies_grok=np.mean(np.array([np.array(self.runs_dic[seed_key].train_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_test_accuracies_nogrok=np.mean(np.array([np.array(non_grokked_container.runs_dic[seed_key].test_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_train_accuracies_nogrok=np.mean(np.array([np.array(non_grokked_container.runs_dic[seed_key].train_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        print(len(avg_test_accuracies_grok))


        fig.add_trace(go.Scatter(x=list(range(len(avg_train_accuracies_grok))),y=avg_train_accuracies_grok,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(avg_test_accuracies_grok))),y=avg_test_accuracies_grok,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(avg_test_accuracies_grok), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(avg_train_accuracies_nogrok))),y=avg_train_accuracies_nogrok,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(avg_test_accuracies_nogrok))),y=avg_test_accuracies_nogrok,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(avg_test_accuracies_nogrok), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)
        run0=list(self.runs_dic.values())[0]
        if epoch==run0.model_epochs()[0]:
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        for i in range(len(avg_weights_grok)):
            flattened_gw=torch.flatten(avg_weights_grok[i]).detach().numpy()
            flattened_ngw=torch.flatten(avg_weights_nogrok[i]).detach().numpy()
            showleg=False
            if i==0:
                showleg=True
            fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+2)
            fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=i+2)

            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            if epoch==run0.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=i+1)
                fig.update_yaxes(title_text="Freq", row=1, col=i+1)
                fig.update_xaxes(title_text="Weight", row=2, col=i+1)
                fig.update_yaxes(title_text="Freq", row=2, col=i+1)

    #Now I want to write scripts for the analysis function.


    def weights_histogram_epochs2(self,non_grokked_container):
        run0=list(self.runs_dic.values())[0]
        epochs=run0.model_epochs()[0::20]
        run0_ng=list(non_grokked_container.runs_dic.values())[0]
        ng_epochs=run0_ng.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        grok_state_dic=run0.models[epochs[0]]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_grok))]
        fig=make_subplots(rows=2,cols=len(weights_grok)+1,subplot_titles=titles)
        for epoch in epochs:
            self.make_weights_histogram2(non_grokked_container,epoch,fig)
        
        total_plots=6+2*(len(weights_grok))
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        fig.show()


# # run_folder=str(root)
# # onlyfiles = [f for f in os.listdir(run_folder) if 'models' in f]
# # print(int(onlyfiles[0].split('time')[-1]))
# # def sort_by_time(filestring):
# #     return int((filestring.split('time')[-1]))

# # onlyfiles.sort(key=sort_by_time)

    
#     model_folder=str(root)+'/'+onlyfiles[-1]

#     #Defines the image - input to the function
#     example_images=X_test.view(1000, 1, 16, 16).to(device)
#     example_image=example_images[0]

#     #necessary imports

#     from os import listdir
#     from os.path import isfile, join


#     #OK - let's iterate over ensembles of the training data

#     def generate_training_set():
#         ising=True
#         if ising:
#             with open("../../Data/IsingML_L16_traintest.pickle", "rb") as handle:
#                 data = pickle.load(handle)
#                 #print(f'Data length: {len(data[1])}')
#                 #print(data[0])
#                 data = data[1]
#             # shuffle data list
#             random.shuffle(data)
#             # split data into input (array) and labels (phase and temp)
#             inputs, phase_labels, temp_labels = zip(*data)
#             # for now ignore temp labels
#             my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
#             my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
#             my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
#             #print(my_X.dtype, my_y.dtype)
#             #print("Created Ising Dataset")
#         else:
#             print("Need to input your own training and test data")


#         train_size, test_size, batch_size = 100, 1000, 100
#         a, b = train_size, test_size
#         train_data = TensorDataset(my_X[b:a+b], my_y[b:a+b]) # Choose training data of specified size
#         test_data = TensorDataset(my_X[:b], my_y[:b]) # test
#         scramble_snapshot=False

#         # load data in batches for reduced memory usage in learning
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
#         test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)
#         # for b, (X_train, y_train) in enumerate(train_loader):
#             #print("batch:", b)
#             #print("input tensors shape (data): ", X_train.shape)
#             #print("output tensors shape (labels): ", y_train.shape)

#         for b, (X_test, y_test) in enumerate(test_loader):
#             if scramble_snapshot:
#                 X_test_a=np.array(X_test)
#                 X_test_perc=np.zeros((test_size,16,16))
#                 for t in range(test_size):
#                     preshuff=X_test_a[t,:,:].flatten()
#                     np.random.shuffle(preshuff)
#                     X_test_perc[t,:,:]=np.reshape(preshuff,(16,16))
#                 X_test=torch.Tensor(X_test_perc).to(dtype)
#             #print("batch:", b)
#             #print("input tensors shape (data): ", X_test.shape)
#             #print("output tensors shape (labels): ", y_test.shape)

#         return test_loader

#     #Urgh - need a function that will let me find the epoch of a given file
#     def get_epoch(x):
#         #assume it's of the form <number>.pth
#         return int(x.split('.')[0])

#     def accuracy(test_loader):
#         epoch=-1
#         pathtodata=model_folder
#         print(model_folder)

#         onlyfiles = [f for f in listdir(pathtodata) if isfile(join(pathtodata, f)) if f!='init.pth']
#         onlyfiles=sorted(onlyfiles,key=get_epoch)


#         full_run_data = torch.load(pathtodata+f'/{onlyfiles[epoch]}')
#         print(pathtodata)
#         print(f'File name: {onlyfiles[epoch]}')
#         #test_model=model
#         test_model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
#                         activation=nn.ReLU(), optimizer=torch.optim.Adam,
#                         learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)

#         test_model.to(device)
#         test_model.load_state_dict(full_run_data['model'])
#         test_correct=0
#         with torch.no_grad():
#             for batch, (X_test, y_test) in enumerate(test_loader):

#                 # Apply the model
#                 y_val = test_model(X_test.view(test_size, 1, 16, 16).to(device))

#                 # Tally the number of correct predictions
#                 predicted = torch.max(y_val.data, 1)[1]
#                 test_correct += (predicted == y_test.to(device)).sum().item()
#                 acc=test_correct/test_size
#         return acc


#     test_loader=generate_training_set()
#     # accuracy=accuracy(test_loader)
#     # print(accuracy)

#     ensemble_acc=[]
#     print(len(ensemble_acc))
#     for i in tqdm(range(1,5), desc="Processing List", unit="i"):
#         test_loader=generate_training_set()
#         acc=accuracy(test_loader)
#         ensemble_acc.append(acc)


#     # print(f'Model folder: {grok_folder}')
#     # print(ensemble_acc)

#     #onlyfiles = [f for f in listdir(grok_folder) if isfile(join(grok_folder, f)) if f!='init.pth']


#     fig,axs=plt.subplots(1)
#     axs.scatter(list(range(len(ensemble_acc))),ensemble_acc)
#     axs.set_title('Accuracy on random test sets')
#     axs.set_xlabel('Test set')
#     axs.set_ylabel('Accuracy')
#     fig.savefig(str(root)+'/Sample_acc.png')




#     #I think this allows you to not have to define a new function
#     from os import listdir
#     from os.path import isfile, join



#     #File locations
#     #1.1 FULL ISING NO SCRAMBLE...

#     #grok_folder=str(root)+'/FullIsing_noscramble_time1700670593'
#     #1. Full Ising, No Scramble
#     #grok_folder=str(root)+'/FullIsing_noscramble_time1700587569'
#     #3. Magzero No Scramble
#     #grok_folder=str(root)+'/magzero_noscramble_time1700665129'

#     #grok_folder=str(root)+'/grok_shuffle1699644166'
#     #grok_folder=str(root)+'/grok_magzero1699637978'

#     #grok_folder=run_name#This plays the role of pathtodata
#     #print(grok_folder)


#     #grok_1699316359 - 20k images



#     def hook_fn2(module,input,output,activation_list):
#         activation_list.append(output)

#     def get_model_neuron_activations(epochindex,pathtodata,image_number,image_set):
#         model_activations=[]
#         temp_hook_dict={}

#         #1. define the model:
#         #1a. Get the saved_files
#         onlyfiles = [f for f in listdir(pathtodata) if isfile(join(pathtodata, f)) if f!='init.pth']



#         full_run_data = torch.load(pathtodata+f'/{onlyfiles[epochindex]}')

#         onlyfiles=sorted(onlyfiles,key=get_epoch)


#         full_run_data = torch.load(pathtodata+f'/{onlyfiles[epochindex]}')
#         print(f'File name: {onlyfiles[epochindex]}')
#         epoch=int(onlyfiles[epochindex].split('.')[0])


#         test_model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
#                         activation=nn.ReLU(), optimizer=torch.optim.Adam,
#                         learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)

#         test_model.to(device)
#         test_model.load_state_dict(full_run_data['model'])
#         #Define the hook
#         def hook_fn(module, input, output):
#             model_activations.append(output)

#         count=0
#         for layer in test_model.modules():
#             if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
#                 count+=1
#                 handle = layer.register_forward_hook(hook_fn)
#                 #print(type(layer).__name__)
#                 temp_hook_dict[count]=handle


#         # 2. Forward pass the example image through the model
#         with torch.no_grad():
#             #print(example_image.shape)
#             output = test_model(image_set)


#         # Detach the hook
#         for hook in temp_hook_dict.values():
#             hook.remove()


#         # 3. Get the activations from the list  for the specified layer


#         #print([len(x) for x in model_activations])

#         detached_activations=[]
#         #print(model_activations[1].shape)
#         for act in model_activations:
#             #print(f'len(act): {len(act)}')
#             if image_number==None:#averages over all images
#                 flat=torch.flatten(sum([act[x] for x in range(len(act))])/len(act))
#             elif image_number=='dist':
#                 flat=act
#             elif image_number=='var':
#                 mean=torch.flatten(sum([act[x] for x in range(len(act))])/len(act))
#                 x2=torch.flatten(sum([act[x]**2 for x in range(len(act))])/len(act))
#                 # print(len(act))
#                 # print(act[0].shape)
#                 # print(act[0])
#                 # print(act[0]**2)
#                 # print(len(mean))
#                 # print(len(x2))
#                 flat=(x2-mean**2)**(1/2)
#                 flat=torch.flatten(flat)
#                 #print(flat)
#             else:
#                 flat=torch.flatten(act[image_number])
#             #print(type(flat))
#             #print(flat.shape)
#             if image_number=='dist':
#                 detached_act=flat.detach().cpu().numpy()
#             else:
#                 detached_act=flat.detach().cpu().numpy()
#             detached_activations.append(detached_act)

#         model_activations.clear() #Need this because otherwise the hook will repopulate the list!

#         #detached_activations_array=np.stack(detached_activations,axis=0)#Convert to an array to keep the vectorization-->You don't want to do this because each layer has a different shape

#         return detached_activations,epoch

#     # sorted_by=None
#     # test=get_model_neuron_activations(epoch=0,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # #test20=get_model_neuron_activations(epoch=1,pathtodata=grok_folder,image_number='var') # For some reason this epoch has a negative variance. Does that indicate you're calcualting the variance wrong?
#     # #test100=get_model_neuron_activations(epoch=5,pathtodata=grok_folder,image_number='var',image_set=example_images)
#     # test1000=get_model_neuron_activations(epoch=50,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test5000=get_model_neuron_activations(epoch=250,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test30000=get_model_neuron_activations(epoch=600,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test30000=get_model_neuron_activations(epoch=600,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test40000=get_model_neuron_activations(epoch=800,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test3=get_model_neuron_activations(epoch=-10,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test4=get_model_neuron_activations(epoch=-2,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test5=get_model_neuron_activations(epoch=-3,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)
#     # test2=get_model_neuron_activations(epoch=-1,pathtodata=grok_folder,image_number=sorted_by,image_set=example_images)


#     #plotting

#     def plot_activation_hist(activation_list,epoch):
#         freqs=[]
#         bin_cs=[]

#         fig,ax= plt.subplots(1,len(activation_list),figsize=(20,5))
#         for act in activation_list:
#             hist=np.histogram(act)
#             freq = hist[0]
#             bin_edges=hist[1]
#             bin_centers=[(bin_edges[i]+bin_edges[i+1])/2 for i in range(0,len(bin_edges)-1)]
#             freqs.append(freq)
#             bin_cs.append(bin_centers)
#         count=0
#         for i in range(len(freqs)):
#             count+=1
#             ax[i].title.set_text(f"Layer {count} Epoch {epoch}")
#             ax[i].set_xlabel("Activation (averaged over training images)")
#             ax[i].set_ylabel("Frequency")
#             ax[i].plot(bin_cs[i],freqs[i],color='r')
#             #plt.scatter(bin_cs2[i],freqs2[i],color='b',label='First model (Ep. 100)')
#             # ax[i].legend()
#         #plt.show()


#     # plot_activation_hist(activation_list=test,epoch=1)
#     # #plot_activation_hist(activation_list=test20,epoch=20) <-- Negative variance - am I calculating var right?
#     # #plot_activation_hist(activation_list=test100,epoch=100)
#     # plot_activation_hist(activation_list=test1000,epoch=1000)
#     # plot_activation_hist(activation_list=test5000,epoch=5000)
#     # plot_activation_hist(activation_list=test30000,epoch=30000)
#     # plot_activation_hist(activation_list=test40000,epoch=40000)
#     # plot_activation_hist(activation_list=test3,epoch=48000)
#     # plot_activation_hist(activation_list=test5,epoch=49960)
#     # plot_activation_hist(activation_list=test4,epoch=49980)
#     # plot_activation_hist(activation_list=test2,epoch=50000)




#     def plot_activation_hist_one(epochlist,sortby,train_accuracies,test_accuracies):
#         activation_test,epoch=get_model_neuron_activations(epochindex=0,pathtodata=model_folder,image_number=sortby,image_set=example_images)
#         fig,axs=plt.subplots(len(epochlist),len(activation_test)+1,figsize=(20,15))


#         for i in epochlist:
#             print(f'i {i}')
#             activation_list,epoch1=get_model_neuron_activations(epochindex=int(i/100),pathtodata=model_folder,image_number=sortby,image_set=example_images)
#             freqs=[]
#             bin_cs=[]

#             print(len(freqs))
#             axs[epochlist.index(i)][0].title.set_text(f"Grok curve - Accuracy")
#             #ymin, ymax = axs[epochlist.index(i)][0].get_ylim()
#             axs[epochlist.index(i)][0].vlines(x=epoch1+1,ymin=0.4,ymax=0.99,linestyles='dashed')
#             axs[epochlist.index(i)][0].semilogx(train_accuracies[3:], c='r',label="train")
#             axs[epochlist.index(i)][0].semilogx(test_accuracies[3:], c='b', label="test")
#             axs[epochlist.index(i)][0].set_xlabel("Epoch")
#             axs[epochlist.index(i)][0].set_ylabel("Accuracy")
#             axs[epochlist.index(i)][0].legend()

#             for act in activation_list:
#                 hist=np.histogram(act)
#                 freq = hist[0]
#                 bin_edges=hist[1]
#                 bin_centers=[(bin_edges[k]+bin_edges[k+1])/2 for k in range(0,len(bin_edges)-1)]
#                 freqs.append(freq)
#                 bin_cs.append(bin_centers)
#             count=0

#             for j in range(len(freqs)):
#                 count+=1
#                 print(epochlist.index(i),j)
#                 axs[epochlist.index(i)][j+1].title.set_text(f"Layer {count} Epoch {epoch1}")
#                 axs[epochlist.index(i)][j+1].set_xlabel("Activation (averaged over training images)")
#                 axs[epochlist.index(i)][j+1].set_ylabel("Frequency")
#                 axs[epochlist.index(i)][j+1].plot(bin_cs[j],freqs[j],color='r')
#                 #axs[epochlist.index(i)][j].plot([1,2,3,4,5],[1,2,3,4,5])
#             plt.tight_layout()
#         #plt.show()
#         fig.savefig(str(root)+"/Barry_activation_hist.pdf")
#                 #axs[epochlist.index(i)][j].scatter(bin_cs2[i],freqs2[i],color='b',label='First model (Ep. 100)')
#                 # ax[i].legend()
#     print(f'Model folder: {model_folder}')
#     plot_activation_hist_one(epochlist=grok_locations,sortby='var',train_accuracies=train_accs,test_accuracies=test_accs)






#     #Interp




#     #Single neuron properties

#     #magnetization

#     def magnetization(spin_grid):
#         return sum(sum(spin_grid))

#     #1. Energy vs. activation

#     def compute_energy(spin_grid, J=1):
#         """
#         Compute the energy of a 2D Ising model.

#         Parameters:
#         spin_grid (2D numpy array): The lattice of spins, each element should be +1 or -1.
#         J (float): Interaction energy. Defaults to 1.

#         Returns:
#         float: Total energy of the configuration.
#         """
#         energy = 0
#         rows, cols = spin_grid.shape

#         for i in range(rows):
#             for j in range(cols):
#                 # Periodic boundary conditions
#                 right_neighbor = spin_grid[i, (j + 1) % cols]
#                 bottom_neighbor = spin_grid[(i + 1) % rows, j]

#                 # Sum over nearest neighbors
#                 energy += -J * spin_grid[i, j] * (right_neighbor + bottom_neighbor)

#         return energy/(2*J*spin_grid.shape[0]*spin_grid.shape[1])

#     # Example usage
#     # spin_grid = np.array(X_test[2].detach().cpu().numpy())
#     # print("Energy of the configuration:", compute_energy(spin_grid))
#     # print(round(compute_energy(spin_grid),2))


#     # plt.spy(X_test[2] + 1 )

#     #2. Connected cluster


#     #http://dragly.org/2013/03/25/working-with-percolation-clusters-in-python/
#     #That's one implementation, but doesn't account for PBC's - so I'll
#     # from pylab import *
#     # import scipy.ndimage

#     # lw, num = scipy.ndimage.label((X_test[0]+1)/2)
#     # print(lw,num)
#     # print(lw.shape)
#     # plt.spy(X_test[0]+1)
#     # plt.show()

#     #Let me try the networkx path

#     import networkx as nx
#     import itertools



#     #1. First need to find the neigbours of a given point

#     def find_neighbours(image, position):
#         neighbour_positions=[]
#         linear_size=image.shape[0]

#         neighbour_positions=[((position[0]+i)%linear_size,(position[1]+j)%linear_size) for i,j in itertools.product([-1,0,1],repeat=2)]
#         value=image[position[0],position[1]]
#         matches=[neighbour_positions[i] for i in range(len(neighbour_positions)) if value==image[neighbour_positions[i][0],neighbour_positions[i][1]]]
#         return matches


#     #2. Then find the connectivity matrix
#     def form_connectivity_matrix(image):
#         dim=1
#         for i in image.shape:
#             dim=i*dim
#             #print(f'dim {dim}')
#         cm=np.zeros((dim,dim))
#         linear_dim=image.shape[0]
#         #print(f'linear dim {linear_dim}')
#         for i in range(cm.shape[0]):
#             #first convert i to a coordinate
#             imagepos=[int(i/16),i%16]
#             neighbours=find_neighbours(image=image,position=imagepos)
#             for n in neighbours:
#                 #first you need to convert the neighbour positions into array indices
#                 neighbourindex=n[0]*16+n[1]
#                 cm[i,neighbourindex]=1
#                 cm[neighbourindex,i]=1
#         #In principle you should go through each row and column but I think just one should be sufficient.
#         for i in range(cm.shape[0]):
#             cm[i,i]=0
#         return cm


#     #3. Create a networkx graph from that matrix
#     def create_graph(c_matrix):
#         linear_dim=c_matrix.shape[0]
#         G=nx.Graph()
#         G.add_nodes_from([i for i in range(linear_dim)])
#         edges=np.transpose(np.nonzero(c_matrix))
#         G.add_edges_from(edges)
#         return G
#     #4. Use networkx to extract the connected component
#     def get_ccs(image):
#         conn_matrix=form_connectivity_matrix(image)
#         graph=create_graph(conn_matrix)
#         ccs=max(nx.connected_components(graph), key=len)
#         largest_cc = len(max(nx.connected_components(graph), key=len))
#         return largest_cc


#     #Functions to pick out a neuron and get the image distribution associated with that neuron

#     #Let's just try to look at a sample of images for the highly stimulated neurons:

#     #print(grok_folder)
#     #grok_folder=str(root)+'/grok_magzero1699637978'
#     epind=-1
#     test2,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='var',image_set=example_images)#Note that this gives me average ordered
#     testdist,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='dist',image_set=example_images)





#     #Let's just do the dumb thing and try to go through the most stimulated neurons and feed in images and see what stimulates it most and least

#     neuron_indices=[]
#     for i in range(len(test2)):
#         t1=np.argsort(test2[i])
#         neuron_indices.append(t1)

#     # print(neuron_indices[-2])
#     # print(test2[-2][6])

#     def get_neuron(neuron_activation_list,layer,nlargest):
#         #Sort by size
#         layer_act=neuron_activation_list[layer]
#         if len(layer_act.shape)>2:
#             layer_act_flat=[layer_act[i].flatten() for i in range(len(layer_act))]
#         else:
#             layer_act_flat=layer_act
#         size_sorted_indices=np.argsort(layer_act_flat)
#         return size_sorted_indices[nlargest],layer

#     testn,testl=get_neuron(neuron_activation_list=test2,layer=2,nlargest=-1)
#     print(testn,testl)

#     def get_image_indices(image_activations,images,neuron_index,layer):
#         #flatten conv layers - can turn off if you need it
#         layer_act=image_activations[layer]
#         if len(layer_act.shape)>2:
#             layer_act_flat=np.array([layer_act[i].flatten() for i in range(len(layer_act))])
#             #print(np.array(test2).shape)
#         else:
#             layer_act_flat=layer_act
#         print(layer_act_flat.shape)
#         images_for_neuron=layer_act_flat[:,neuron_index]
#         image_indices=np.argsort(images_for_neuron)


#         return images_for_neuron,image_indices

#     testi,testg=get_image_indices(image_activations=testdist,images=X_test,neuron_index=testn,layer=2)









#     def single_neuron_vec(image_activations_dist):
#         neuron_vector=[]
#         average_activation=[]
#         variance=[]
#         energy=[]
#         # Quantities calculated over all images
#         for layer in range(len(image_activations_dist)):
#             average_activation.append(sum(image_activations_dist,axis=0)/image_activations_dist.shape[0])
#             variance.append(np.sum(np.square(image_activations_dist),axis=0)/len(image_activations_dist.shape[0])-np.square(average_activation))
#         return

#     def image_values(images,func):
#         image_quantity=[]
#         for i in range(images.shape[0]):#Assumes axis 0 is the image axis
#             value=func(images[i,:,:])
#             image_quantity.append(value)
#         return np.array(image_quantity)

#     list_of_image_functions=[compute_energy,get_ccs]
#     #Let's form an array of image values:

#     # energies=[compute_energy(i) for i in X_test]
#     # largest_cc=[get_ccs(i) for i in X_test]


#     print(f"grok_folder: {model_folder}")
#     #epind=int(19900/100)
#     epind=-1
#     test2,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='var',image_set=example_images)#Note that this gives me average ordered
#     testdist,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='dist',image_set=example_images)


#     def visualize_images(image_list,image_indices,image_activations,nlargest):
#         rowcount=10
#         depthcount=3
#         image_positions=[int(x) for x in np.linspace(depthcount,len(image_list)-1,rowcount)]
#         image_indices=[[image_indices[x-i] for i in range(depthcount)] for x in image_positions]
#         labels=[y_test[x] for x in image_positions]
#         #print(image_indices)
#         image_acts=[image_activations[x] for x in image_indices]
#         #print(image_acts)
#         #Now just visualize the images

#         fig, ax = plt.subplots(depthcount,rowcount,figsize=(20,5))
        
#         for j in range(depthcount):
#             for i in range(rowcount):
#                 a=round(image_acts[i][j],2)
#                 ax[j,i].spy(X_test[image_indices[i][j]] + 1 )
#                 energy=compute_energy(X_test[image_indices[i][j]])
#                 energy=round(energy.item(),3)
#                 ax[j,i].set_title(f'Act: {str(a)}')#, , l{y_test[image_indices[i]]},#f'Act: {str(a)}'
#                 ax[j,i].set_xlabel(f'T-Tc: {str(np.round((my_y_temp[image_indices[i][j]]-2.69).numpy(),2))}, En: {energy}')#mag: {sum(torch.flatten(X_test[image_indices[i]]))}
#                 ax[j,i].set_yticklabels([])
#                 ax[j,i].set_xticklabels([])
#                 if nlargest<0:
#                     number=-nlargest
#                     text='th most activated neuron'
#                 else:
#                     number=nlargest+1
#                     text='th least activated neuron'
#         fig.suptitle(f'{number}{text}')
#         fig.tight_layout()
#         fig.savefig(str(root)+f'/{number}{text}_epoch_{str(int(epind*20))}_quiltgraphs')
#         fig.subplots_adjust(top=0.88)


#         #plt.spy(configs[50]+1)
#         #plt.show()

#     def visualize_image_values(image_values,image_activations,func_labels,nlargest):
#         fig, ax=plt.subplots(1,len(image_values),figsize=(25,10))
        
#         if len(image_values)==1:
#             ax.scatter(image_values,image_activations)
#             ax.set_title('Something')
#             ax.set_xlabel('Values')
#             ax.set_ylabel('Activations')
#         else:
#             for i in range(len(image_values)):
#                 ax[i].scatter(image_values[i],image_activations)
#                 ax[i].set_title(func_labels[i])
#                 ax[i].set_xlabel(func_labels[i])
#                 ax[i].set_ylabel('Activations')
#         if nlargest<0:
#             number=-nlargest
#             text='th most activated neuron'
#         else:
#             number=nlargest+1
#             text='th least activated neuron'
#         fig.suptitle(f'{number}{text}'+' Image-Activation graphs')
#         fig.tight_layout()
#         fig.savefig(str(root)+f'/{number}{text}_epoch_{str(int(epind*20))}_corrgraphs')
#         #plt.show()


#         return None

#     testn,testl=get_neuron(neuron_activation_list=test2,layer=2,nlargest=-1)
#     def image_graphs(funcs,image_set,image_activations,image_indices,func_labels,nlargest):
#         all_image_vals=[]
#         new_acts=[image_activations[image_indices[i]]for i in range(len(image_indices))]
#         for func in funcs:
#             image_vals=[func(image_set[image_indices[i]]) for i in range(len(image_indices))]
#             all_image_vals.append(image_vals)
#         viz_all=visualize_image_values(all_image_vals,new_acts,func_labels,nlargest=nlargest)
#         return image_vals

#     #testv1=visualize_images(image_list=X_test,image_indices=testg,image_activations=testi,nlargest=-1)
#     # testv2=visualize_images(image_list=X_test,image_indices=testg,image_activations=testi)


#     np.array(X_test[2].detach().cpu().numpy())

#     def get_quilt(image_activations,images,neuron_index,layer,global_funcs,func_labels):
#         testn,testl=get_neuron(neuron_activation_list=test2,layer=layer,nlargest=neuron_index)
#         print(testn)
#         image_activations,image_indices=get_image_indices(image_activations=image_activations,images=images,neuron_index=testn,layer=layer)
#         viz=visualize_images(image_list=images,image_indices=image_indices,image_activations=image_activations,nlargest=neuron_index)
#         viz_all=image_graphs(funcs=global_funcs,image_set=images,image_activations=image_activations,image_indices=image_indices,func_labels=func_labels,nlargest=neuron_index)



#     get_quilt(image_activations=testdist,images=X_test,neuron_index=-1,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     #get_quilt(image_activations=testdist,images=X_test,neuron_index=-3,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     #get_quilt(image_activations=testdist,images=X_test,neuron_index=-4,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     #get_quilt(image_activations=testdist,images=X_test,neuron_index=-5,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])


#     get_quilt(image_activations=testdist,images=X_test,neuron_index=-2,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     get_quilt(image_activations=testdist,images=X_test,neuron_index=-3,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     get_quilt(image_activations=testdist,images=X_test,neuron_index=-4,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     get_quilt(image_activations=testdist,images=X_test,neuron_index=-5,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     get_quilt(image_activations=testdist,images=X_test,neuron_index=-10,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     #get_quilt(image_activations=testdist,images=X_test,neuron_index=-20,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
#     #get_quilt(image_activations=testdist,images=X_test,neuron_index=0,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])

#     #Save user defined variables