import os
from utils import load_model
import numpy as np
from ref_data.loadData import load_dataset
from pixel_operation import getPosition

def get_object_acc(labels,scores,name,dataset,resdir):

    gates=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]

    
    object_acc=compute_object_acc(labels,dataset)
           

    
    filename=os.path.join(resdir,'object_acc')
    if not os.path.exists(filename):
        os.makedirs(filename)
    filename=filename+'/'+name+'.txt'
    file = open(filename, 'w')
    file.close()
    
    np.savetxt(filename,object_acc,fmt="%.2f,%d,%d,%d,%d,%.5f,%d,%.5f")

def compute_object_acc(pred_labels,dataset):

    gates=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    results=np.zeros((len(gates),8))
    for n_gate in range(len(gates)):
        results[n_gate,0]=gates[n_gate]


    timestamps,ind=np.unique(dataset.test_data['timestamp'], return_index=True)
    timestamps=timestamps[np.argsort(ind)]
    i=0
    for timestamp in timestamps:

        idx=dataset.test_data['timestamp']==timestamp

        frame_idx=np.abs(dataset.test_data['pixel_labels_times']-timestamp)<1000
        pos,num = getPosition(dataset.test_data['pixel_labels_names'][frame_idx][0])
        result=0
        if (np.sum(pred_labels[idx])>0.0):
            patch_idx = pred_labels[idx]==1
            result=1
            pred_img=np.zeros((260,346))
            patch_pose = dataset.test_data['poses'][idx][patch_idx]
            for n in range(patch_pose.shape[0]):
                pred_img[int(259-patch_pose[n,1]*14-14):int(259-patch_pose[n,1]*14),int(patch_pose[n,0]*18):int(patch_pose[n,0]*18+18)]=1
                    
            gt_img=np.zeros((260,346))
            abnor_idx =pos!=0
            gt_img[abnor_idx]=1

            inter_img=np.sum(((pred_img+gt_img)==2)*1)
            union_img=np.sum(((pred_img+gt_img)!=0)*1)
            iou=inter_img/union_img
            for n_gate in range(len(gates)):
                if iou > gates[n_gate]:
                    results[n_gate,1] = results[n_gate,1]+1
                else:
                    results[n_gate,2] = results[n_gate,2]+1

                        
        if i == 0:
            pred_result=result
        else:
            pred_result=np.append(pred_result,result)
        i+=1
                
    results[:,3]=np.array([np.sum(((pred_result+dataset.test_data['labels'])==0)*1) for i in range(len(gates))])
    results[:,4]=np.array([np.sum(((pred_result-dataset.test_data['labels'])==-1)*1) for i in range(len(gates))])
    results[:,5]=np.divide(results[:,1]+results[:,3],results[:,1]+results[:,2]+results[:,3]+results[:,4])

    results[:,6]=np.array([np.sum(((pred_result+dataset.test_data['labels'])==2)*1) for i in range(len(gates))])
    results[:,7]=np.divide(results[:,3]+results[:,6],results[:,1]+results[:,2]+results[:,3]+results[:,4])

    return results








def main ():
    moddir='/xxxxx/features/norm_models'
    plotdir='/xxxxx/features/norm_results'
   
    names=[name for name in os.listdir(moddir)]
    names.sort()
    scores={}
    for name in names:
        feature=os.path.splitext(name.split('_')[2])[0]
        model=name.split('_')[0]
        which_set=name.split('_')[1]
        
        if feature not in scores.keys():
            scores[feature]={}

        if model not in scores[feature].keys():
            scores[feature][model]={}
            scores[feature][model]['err']={}

            scores[feature][model]['err']['test']=None
            scores[feature][model]['distance']={}

            scores[feature][model]['distance']['test']=None

        modname = os.path.join(moddir,name)
        model_data = load_model(modname)
        
        datasetdir=os.path.split(moddir)[0]+'/'+feature
        dataset = load_dataset('sigactcuboid',datasetdir)
        if which_set=='test':
            

            if feature!='SB':
                if model=='OneClassSVM':
                    labels=((model_data.diag['test']['scores']>0)*1).reshape(1,-1)
                    test_score=model_data.diag['test']['scores'].reshape(1,-1)
                    get_object_acc(labels,test_score,feature+'_'+model+'_'+'err',dataset,plotdir)
                    
                else:
                    labels=model_data.pred_labels
                    test_err=model_data.err['test']
                    get_object_acc(labels,test_err,feature+'_'+model+'_'+'err',dataset,plotdir)
                    
                    test_distance=model_data.distance['test']
                    # train_threshold=model_data.threshold
                    # train_mean=model_data.mean
                    # train_covMatInv=model_data.covMatInv

                    get_object_acc(labels,test_distance,feature+'_'+model+'_'+'distance',dataset,plotdir)

        






if __name__ == '__main__':
    main()