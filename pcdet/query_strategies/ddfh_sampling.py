
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import tqdm
from sklearn.cluster import kmeans_plusplus, KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from typing import Dict, List
import pickle, os
from collections import defaultdict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from sklearn.manifold import TSNE, Isomap
import time
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import warnings
from sklearn import mixture
from scipy.stats import entropy
warnings.filterwarnings('ignore')

class DDFHSmpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(DDFHSmpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

    def pairwise_squared_distances(self, x, y):
        # x: N * D * D * 1
        # y: M * D * D * 1
        # return: N * M    
        assert (len(x.shape)) > 1
        assert (len(y.shape)) > 1  
        n = x.shape[0]
        m = y.shape[0]
        x = x.view(n, -1)
        y = y.view(m, -1)
                                                              
        x_norm = (x**2).sum(1).view(n, 1)
        y_t = y.permute(1, 0).contiguous()
        y_norm = (y**2).sum(1).view(1, m)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist[dist != dist] = 0 # replace nan values with 0
        return torch.clamp(dist, 0.0, float('inf'))

    def select_func(self):
        pass
    
    def enable_dropout(self, model, enable=True):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))

    def query(self, leave_pbar=True, cur_epoch=None):
        def divNum(num, parts):
            p = int(num/parts)
            missing = int(num - p*parts)             
            return [p+1]*missing + [p]*(parts-missing)
        
        class_names = self.cfg.CLASS_NAMES
        total_select_nums = self.cfg.ACTIVE_TRAIN.SELECT_NUMS
        cls_select_nums = divNum(total_select_nums,len(class_names))
        
        self.model.eval()
        record_dict=defaultdict(list)
        

        # feed unlabeled data forward the model
        total_it_each_epoch = len(self.unlabelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        # self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SEL ECT_NUMS
        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)
                for batch_inx in range(len(pred_dicts)):
                    
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    
                    for instance_idx in range(len(pred_instance_labels)):
                        record_dict['unlabeled_frame_id'].append(unlabelled_batch['frame_id'][batch_inx])
                        record_dict['unlabeled_labels'].append(pred_instance_labels[instance_idx])
                        record_dict['unlabeled_rotation'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-1])
                        record_dict['unlabeled_height_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-4])
                        record_dict['unlabeled_width_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-3])
                        record_dict['unlabeled_length_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-2])
                        record_dict['unlabeled_pts'].append(pred_dicts[batch_inx]['pts_count'][instance_idx])
                        record_dict['unlabeled_box_volumes'].append(pred_dicts[batch_inx]['box_volumes'][instance_idx])
                        record_dict['unlabeled_embeddings'].append(pred_dicts[batch_inx]['pred_embeddings'][instance_idx].view(-1))
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                    
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
            
        ## Label
        # feed labeled data forward the model
        total_it_each_epoch = len(self.labelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_labelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
        val_dataloader_iter = iter(self.labelled_loader)
        val_loader = self.labelled_loader
        for cur_it in range(total_it_each_epoch):
            try:
                labelled_batch = next(val_dataloader_iter)
            except StopIteration:
                labelled_dataloader_iter = iter(val_loader)
                labelled_batch = next(labelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(labelled_batch)
                pred_dicts, _ = self.model(labelled_batch)
                
                for batch_inx in range(len(pred_dicts)):
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    
                    for instance_idx in range(len(pred_instance_labels)):
                        record_dict['labeled_frame_id'].append(labelled_batch['frame_id'][batch_inx])
                        record_dict['labeled_labels'].append(pred_instance_labels[instance_idx])
                        record_dict['labeled_rotation'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-1])
                        record_dict['labeled_height_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-4])
                        record_dict['labeled_width_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-3])
                        record_dict['labeled_length_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-2])
                        record_dict['labeled_pts'].append(pred_dicts[batch_inx]['pts_count'][instance_idx])
                        record_dict['labeled_box_volumes'].append(pred_dicts[batch_inx]['box_volumes'][instance_idx])
                        record_dict['labeled_embeddings'].append(pred_dicts[batch_inx]['pred_embeddings'][instance_idx].view(-1))
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= Process data, convert dictionary to dataframe ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in  str(k):
                    df[k] = [i.cpu().numpy() for i in v]

            return df
       
        df = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})

        ## ============= Combine unlabeled and labeled data ============= 
        combine_df = pd.concat([unlabeled_df,labeled_df])

        ## ============= Convert data type ============= 
        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## ============= Extract features ============= 
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']#remove ,'cls_entropy','confidence'
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)

        # Number of GMM components
        components = 10
        
        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        # Fit TSNE and GMM on unlabeled dataset
        for c_idx, select_num in enumerate(cls_select_nums): 
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne_running = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100,n_jobs=8)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)
            print(f"--- TSNE Unlabeled running time: %s seconds ---" % (time.time() - tsne_running))
            
            gmm_running = time.time()
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)
            print(f"--- GMM Unlabeled running time: %s seconds ---" % (time.time() - gmm_running))

        # Fit TSNE and GMM on labeled dataset
        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)
                feature_df.loc[(feature_df.labels == cidx),'combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values
                data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values
                
                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']
            redundant_frame_df = pred_df.groupby(['frame_id'])[['model_redundant']].sum()

            
            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
            

            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = pred_df['instance_distribution_gap'] 

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = 1+class_weigted_df/class_weigted_df.sum()
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            pred_df['dd_score']  = QuantileTransformer().fit_transform((pred_df['instance_model_score'].values)[:,None])[:,0] 
            saving_dict['pred_df'] = pred_df

            pred_df['nov_score'] = (1-QuantileTransformer().fit_transform(pred_df['model_redundant'].values[:,None])[:,0])

            feature_name = 'combine_features'
            labeled_df = feature_df[feature_df.labeled ==1]
            instance_embedding_std_df = pred_df.groupby(['frame_id','labels']).apply(lambda x: np.cov(np.concatenate([np.array(x[feature_name].tolist()), np.array(labeled_df.loc[labeled_df.labels == x.name[1], feature_name].tolist())]).T)).reset_index(name='feature_cov')
            feature_cov = np.array(instance_embedding_std_df.feature_cov.tolist())
            feature_cov=np.abs(feature_cov)
            a_var = np.diagonal(feature_cov,axis1=1, axis2=2)
            a_std_m = (a_var.reshape(len(a_var),-1,1)@np.moveaxis(a_var.reshape(len(a_var),-1,1), -1, -2))**0.5
            feature_cov =  feature_cov/a_std_m
            global_embedding_var = np.mean(a_var,axis=-1)
            feature_cov = np.array([i[np.triu_indices(len(i),1)].mean() for i in feature_cov])

            instance_embedding_std_df['global_embedding_var'] = global_embedding_var 
            instance_embedding_std_df['global_correlation'] = feature_cov 
            instance_embedding_std_df['global_embedding_var'] = QuantileTransformer().fit_transform(instance_embedding_std_df['global_embedding_var'].values[:,None])[:,0]
            instance_embedding_std_df['global_correlation'] = 1-QuantileTransformer().fit_transform(instance_embedding_std_df['global_correlation'].values[:,None])[:,0]
            instance_embedding_std_df['ff_score'] = QuantileTransformer().fit_transform((instance_embedding_std_df['global_embedding_var']*instance_embedding_std_df['global_correlation']).values[:,None])[:,0]

            instance_embedding_std_df = instance_embedding_std_df.groupby('frame_id').ff_score.mean()

            redundant_frame_df = pred_df.groupby(['frame_id'])[['nov_score']].mean()
            label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).dd_score.mean().loc[label_entropy_df.frame_id].reset_index()
            frame_df['ff_score'] = frame_df.frame_id.map(instance_embedding_std_df)
            frame_df['nov_score'] = frame_df.frame_id.map(redundant_frame_df.nov_score)            
            frame_df['dd_score'] = (frame_df['dd_score']+frame_df['ff_score'] + frame_df['nov_score'])*(label_entropy_df.frame_ent+1)
            _selected_frames = frame_df.sort_values(by='dd_score',ascending=False).iloc[:int(total_select_nums//10)].frame_id.tolist()

            selected_frames.extend(_selected_frames)
            feature_df.loc[feature_df.frame_id.isin(selected_frames), 'labeled'] = 1

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        saving_dict['feature_df'] = feature_df


        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
        print(len(set(selected_frames)))
        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(saving_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        return selected_frames