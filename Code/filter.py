from data_objects import *

params_dic={'weight_decay':0.08,'weight_multiplier':10,'learning_rate':0.0001,'hidden_layers':[20],'conv_channels':[2,4],'train_size':100,'test_size':1000,'dropout_p':0}
# run_grok=seed_run_container()
# run_grok.params_dic=params_dic
# foldername='../NewCluster/grok_True_time_1710987028'
# run_grok.aggregate_runs(folder=foldername)
# print(f'Runs dic g: {run_grok.runs_dic}')
# filename=f'grok_container_time_{int(time.time())}'+'.dill'
# with open(filename,'wb') as file:
#     dill.dump(run_grok,file)

add=True
add_filename='nogrok_container_time_1711050018.dill'
params_dic_ng={'weight_decay':0.01,'weight_multiplier':1,'learning_rate':0.0001,'hidden_layers':[20],'conv_channels':[2,4],'train_size':100,'test_size':1000,'dropout_p':0}
if add:
    with open(add_filename,'rb') as file:
        run_ng=dill.load(file)
else:
    run_ng=seed_run_container()
run_ng.params_dic=params_dic_ng
foldername='../NewCluster/grok_False_time_1711050060'
filename=f'nogrok_container_time_{int(time.time())}'+'.dill'
run_ng.aggregate_runs(folder=foldername)
print(f'Runs dic ng: {run_ng.runs_dic}')
with open(filename,'wb') as file:
    dill.dump(run_ng,file)