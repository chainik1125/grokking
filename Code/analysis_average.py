from data_objects import *




data_object_file_name='grok_container_time_1711050013.dill'
non_grokked_file_name='nogrok_container_time_1711051645.dill'


with open(data_object_file_name, 'rb') as in_strm:
    avg_run = dill.load(in_strm)
with open(non_grokked_file_name, 'rb') as in_strm:
    avg_run_ng = dill.load(in_strm)
for run_key in avg_run.runs_dic.keys():
    avg_run.runs_dic[run_key].weights_histogram_epochs2(avg_run_ng.runs_dic[run_key])
exit()
# single_run.losscurvesfig=single_run.make_loss_curves()
# single_run.losscurvesfig.show()
lwg=4
titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(lwg)]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(lwg)]
fig=make_subplots(rows=2,cols=lwg+1,subplot_titles=titles)
#avg_run.make_weights_histogram2(non_grokked_container=avg_run_ng,fig=fig,epoch=900).show()
avg_run.weights_histogram_epochs2(avg_run_ng)
exit()


# grok_state_dic1=single_run.models[0]['model']
# weights_grok1=[grok_state_dic1[key] for key in grok_state_dic1.keys() if 'weight' in key]

# grok_state_dic2=single_run.models[100]['model']
# weights_grok2=[grok_state_dic2[key] for key in grok_state_dic2.keys() if 'weight' in key]


# print(torch.equal(weights_grok1[-1],weights_grok2[-1]))
single_run.weights_histogram_epochs2(single_run_ng)
# single_run.svd_one_epoch(single_run_ng,epoch=19000,fig=None).show()

single_run.svd_epochs(single_run_ng).show()
# single_run.pca_one_epoch(single_run_ng,epoch=19900,fig=None).show()
single_run.pca_epochs(single_run_ng).show()





# single_run.weights_histogram_epochs2(non_grokked_object=single_run_ng)

