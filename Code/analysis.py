from data_objects import *


data_object_file_name='../NewCluster/grok_True_time_1708677333/data_seed_0_time_1708678502'
non_grokked_file_name='../NewCluster/grok_False_time_1708679156/data_seed_0_time_1708680287'

data_object_file_name='../NewCluster/grok_True_time_1708703980/data_seed_0_time_1708704502'
non_grokked_file_name='../NewCluster/grok_False_time_1708704596/data_seed_0_time_1708705117'
with open(data_object_file_name, 'rb') as in_strm:
    single_run = dill.load(in_strm)
with open(non_grokked_file_name, 'rb') as in_strm:
    single_run_ng = dill.load(in_strm)

# single_run.losscurvesfig=single_run.make_loss_curves()
# single_run.losscurvesfig.show()
single_run.weightshistfig=single_run.make_weights_histogram(non_grokked_object=single_run_ng,epoch=900)
print(single_run.model_epochs())

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

