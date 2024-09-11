
import torch
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
import tqdm
from sklearn.cluster import kmeans_plusplus, KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from sklearn.neighbors import KernelDensity
from scipy.cluster.vq import vq
from typing import Dict, List
import pickle, os
from collections import defaultdict
from torch.distributions import Categorical
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA, KernelPCA
import numpy as np
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
import time
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import warnings
from sklearn import mixture
from scipy.stats import entropy
from torch import softmax, adaptive_avg_pool1d
from torch.nn.functional import adaptive_avg_pool2d
warnings.filterwarnings('ignore')

# confidence weight 50 -> 10 eps 1e-3
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedV2Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedV2Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        # record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        # record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        # record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        # record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(record_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        labels  =combine_df.labels
        combine_df = pd.get_dummies(combine_df,columns=['labels'])
        combine_df['labels'] = labels
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std','labels_1', 'labels_2', 'labels_3']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        # if self.rank == 0:
        #     pbar = tqdm.tqdm(total=total_select_nums, leave=leave_pbar,
                                # desc='greedy selected frames: ', dynamic_ncols=True)
        # for i in range(total_select_nums):
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            eps=1e-3
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)

            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            data_df['model_redundant'] = np.exp(pred)
            data_df['model_redundant_score'] = np.power(1.2,pred)
            data_df['model_sigmoid'] = 1/(1+(np.power(1.2,pred)))#1 - 2 / np.pi * np.arctan(np.pi / 2 * np.exp(pred)) #1-np.log1p(np.exp(pred))#

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)

            data_df['model_redundant_density'] = (data_df['model_redundant_score']+eps)/(np.power(1.2, pred)+eps)
            data_df['model_score'] = (np.power(1.2, pred)+eps)/(data_df['model_redundant_score']+eps)
            
            pred_list.append(data_df)
            
            if  i == 0 or i == total_select_nums-1:
                print(f"--- Model {cidx} running time: %s seconds ---" % (time.time() - model_start_time))
                print('Testing prediction mean: ',sum(data_df['model_pred'])/len(data_df['model_pred']))


        pred_df = pd.concat(pred_list)
        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))


        # filter 1
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        # # filter 2
        pred_df['model_score'] = pred_df['model_score']*pred_df['class_confidence_weighted']
        redundant_frame_id = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index(drop=False).groupby('frame_id').model_score.mean().reset_index(drop=False).sort_values(by='model_score')
        # redundant_frame_id = redundant_frame_id.frame_id[:200]
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_score>redundant_frame_id.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        #filter by label entropy
        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()
        

        

        if i == 0 or i == total_select_nums-1:
            print('==== select score ====')
            print(pred_df.groupby('labels').select_score.median())
            print('==== confidence ====')
            print(pred_df.groupby('labels').confidence.median())
            print('==== redundant ====')
            print(pred_df.groupby('labels').model_redundant.median())
        

        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
        return selected_frames

# confidence weight + relative confidence weighted 50 -> 10 eps 1e-3
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedV3Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedV3Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(record_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        labels  =combine_df.labels
        combine_df = pd.get_dummies(combine_df,columns=['labels'])
        combine_df['labels'] = labels
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std','labels_1', 'labels_2', 'labels_3']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        # if self.rank == 0:
        #     pbar = tqdm.tqdm(total=total_select_nums, leave=leave_pbar,
                                # desc='greedy selected frames: ', dynamic_ncols=True)
        # for i in range(total_select_nums):
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            eps=1e-3
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)

            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            data_df['model_redundant'] = np.exp(pred)
            data_df['model_redundant_score'] = np.power(1.2,pred)
            data_df['model_sigmoid'] = 1/(1+(np.power(1.2,pred)))#1 - 2 / np.pi * np.arctan(np.pi / 2 * np.exp(pred)) #1-np.log1p(np.exp(pred))#

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)

            data_df['model_redundant_density'] = (data_df['model_redundant_score']+eps)/(np.power(1.2, pred)+eps)
            data_df['model_score'] = (np.power(1.2, pred)+eps)/(data_df['model_redundant_score']+eps)
            
            pred_list.append(data_df)
            
            if  i == 0 or i == total_select_nums-1:
                print(f"--- Model {cidx} running time: %s seconds ---" % (time.time() - model_start_time))
                print('Testing prediction mean: ',sum(data_df['model_pred'])/len(data_df['model_pred']))


        pred_df = pd.concat(pred_list)
        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))


        # filter 1
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        # # filter 2
        pred_df['model_score'] = pred_df['model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        redundant_frame_id = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index(drop=False).groupby('frame_id').model_score.mean().reset_index(drop=False).sort_values(by='model_score')
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_score>redundant_frame_id.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        #filter by label entropy
        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()
        

        

        if i == 0 or i == total_select_nums-1:
            print('==== select score ====')
            print(pred_df.groupby('labels').select_score.median())
            print('==== confidence ====')
            print(pred_df.groupby('labels').confidence.median())
            print('==== redundant ====')
            print(pred_df.groupby('labels').model_redundant.median())
        

        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
        return selected_frames

# labeled/unlabeled Dynamic 50 -> 10 eps 1e-3
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV5Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV5Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(record_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        labels  =combine_df.labels
        combine_df = pd.get_dummies(combine_df,columns=['labels'])
        combine_df['labels'] = labels
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std','labels_1', 'labels_2', 'labels_3']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        # if self.rank == 0:
        #     pbar = tqdm.tqdm(total=total_select_nums, leave=leave_pbar,
                                # desc='greedy selected frames: ', dynamic_ncols=True)
        # for i in range(total_select_nums):
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred
            
            
            pred_list.append(data_df)
            
            


        pred_df = pd.concat(pred_list)

        eps=1e-3
        mean_target = 0.3
        labeled_mean_pred_score = min(np.mean(pred_df['model_pred']),-0.01)
        labeled_pow_factor = min(np.power(mean_target,1/labeled_mean_pred_score),np.exp(1))

        unlabeled_mean_pred_score = min(np.mean(pred_df['unlabeled_pred']),-0.01)
        unlabeled_pow_factor = min(np.power(mean_target,1/unlabeled_mean_pred_score),np.exp(1))
        
        pred_df['model_redundant'] = np.power(labeled_pow_factor ,pred_df['model_pred'])
        pred_df['model_redundant_score'] = np.power(labeled_pow_factor,pred_df['model_pred'])
        pred_df['model_redundant_score'] = pred_df['model_redundant_score']/pred_df['model_redundant_score'].mean()

        pred_df['unlabeled_redundant'] = np.power(unlabeled_pow_factor,pred_df['unlabeled_pred'])
        pred_df['unlabeled_redundant'] = pred_df['unlabeled_redundant']/pred_df['unlabeled_redundant'].mean()

        pred_df['model_redundant_density'] = (pred_df['model_redundant_score'])/(pred_df['unlabeled_redundant']+eps)
        pred_df['model_score'] = (pred_df['unlabeled_redundant'])/(pred_df['model_redundant_score']+eps)

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(),pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Redundant',pred_df.groupby('labels')['unlabeled_redundant'].min(),pred_df.groupby('labels')['unlabeled_redundant'].mean(), pred_df.groupby('labels')['unlabeled_redundant'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Current power factor: ',labeled_pow_factor, unlabeled_pow_factor)
        print('median power: ',labeled_mean_pred_score ,unlabeled_mean_pred_score)


        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))


        # filter 1
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.mean().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        # # filter 2
        pred_df['model_score'] = pred_df['model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        redundant_frame_id = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index(drop=False).groupby('frame_id').model_score.mean().reset_index(drop=False).sort_values(by='model_score')
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_score>redundant_frame_id.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        #filter by label entropy
        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()
        

        

        if i == 0 or i == total_select_nums-1:
            print('==== select score ====')
            print(pred_df.groupby('labels').select_score.median())
            print('==== confidence ====')
            print(pred_df.groupby('labels').confidence.median())
            print('==== redundant ====')
            print(pred_df.groupby('labels').model_redundant.median())
        

        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
        return selected_frames

# labeled/unlabeled Dynamic 30 -> 10 eps 1e-4
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV6Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV6Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(record_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        labels  =combine_df.labels
        combine_df = pd.get_dummies(combine_df,columns=['labels'])
        combine_df['labels'] = labels
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std','labels_1', 'labels_2', 'labels_3']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        # if self.rank == 0:
        #     pbar = tqdm.tqdm(total=total_select_nums, leave=leave_pbar,
                                # desc='greedy selected frames: ', dynamic_ncols=True)
        # for i in range(total_select_nums):
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=3000,random_state=3131,reg_covar=5e-3)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=3000,random_state=3131,reg_covar=5e-3)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred
            
            
            pred_list.append(data_df)
            
            


        pred_df = pd.concat(pred_list)

        eps=1e-4
        mean_target = 0.3
        labeled_mean_pred_score = min(np.mean(pred_df['model_pred']),-0.01)
        labeled_pow_factor = min(np.power(mean_target,1/labeled_mean_pred_score),np.exp(1))

        unlabeled_mean_pred_score = min(np.mean(pred_df['unlabeled_pred']),-0.01)
        unlabeled_pow_factor = min(np.power(mean_target,1/unlabeled_mean_pred_score),np.exp(1))
        
        pred_df['model_redundant'] = np.power(labeled_pow_factor ,pred_df['model_pred'])
        pred_df['model_redundant_score'] = np.power(labeled_pow_factor,pred_df['model_pred'])
        pred_df['model_redundant_score'] = pred_df['model_redundant_score']/pred_df['model_redundant_score'].mean()

        pred_df['unlabeled_redundant'] = np.power(unlabeled_pow_factor,pred_df['unlabeled_pred'])
        pred_df['unlabeled_redundant'] = pred_df['unlabeled_redundant']/pred_df['unlabeled_redundant'].mean()

        pred_df['model_redundant_density'] = (pred_df['model_redundant_score'])/(pred_df['unlabeled_redundant']+eps)
        pred_df['model_score'] = (pred_df['unlabeled_redundant'])/(pred_df['model_redundant_score']+eps)

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(),pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Redundant',pred_df.groupby('labels')['unlabeled_redundant'].min(),pred_df.groupby('labels')['unlabeled_redundant'].mean(), pred_df.groupby('labels')['unlabeled_redundant'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Current power factor: ',labeled_pow_factor, unlabeled_pow_factor)
        print('median power: ',labeled_mean_pred_score ,unlabeled_mean_pred_score)


        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))


        # filter 1
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.mean().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        # # filter 2
        pred_df['model_score'] = pred_df['model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        redundant_frame_id = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index(drop=False).groupby('frame_id').model_score.mean().reset_index(drop=False).sort_values(by='model_score')
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_score>redundant_frame_id.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        #filter by label entropy
        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()
        

        

        if i == 0 or i == total_select_nums-1:
            print('==== select score ====')
            print(pred_df.groupby('labels').select_score.median())
            print('==== confidence ====')
            print(pred_df.groupby('labels').confidence.median())
            print('==== redundant ====')
            print(pred_df.groupby('labels').model_redundant.median())
        

        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
        return selected_frames

# labeled/unlabeled Dynamic DR 30 -> 10 eps 1e-4
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV7Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV7Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(record_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        labels  =combine_df.labels
        combine_df = pd.get_dummies(combine_df,columns=['labels'])
        combine_df['labels'] = labels
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std','labels_1', 'labels_2', 'labels_3']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        # if self.rank == 0:
        #     pbar = tqdm.tqdm(total=total_select_nums, leave=leave_pbar,
                                # desc='greedy selected frames: ', dynamic_ncols=True)
        # for i in range(total_select_nums):
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=3000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=3000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred
            
            
            pred_list.append(data_df)
            
            


        pred_df = pd.concat(pred_list)

        eps=1e-4
        mean_target = 0.3
        labeled_mean_pred_score = min(np.mean(pred_df['model_pred']),-0.01)
        labeled_pow_factor = min(np.power(mean_target,1/labeled_mean_pred_score),np.exp(1))

        unlabeled_mean_pred_score = min(np.mean(pred_df['unlabeled_pred']),-0.01)
        unlabeled_pow_factor = min(np.power(mean_target,1/unlabeled_mean_pred_score),np.exp(1))
        
        pred_df['model_redundant'] = np.power(labeled_pow_factor ,pred_df['model_pred'])
        pred_df['model_redundant_score'] = np.power(labeled_pow_factor,pred_df['model_pred'])
        pred_df['model_redundant_score'] = pred_df['model_redundant_score']/pred_df['model_redundant_score'].mean()

        pred_df['unlabeled_redundant'] = np.power(unlabeled_pow_factor,pred_df['unlabeled_pred'])
        pred_df['unlabeled_redundant'] = pred_df['unlabeled_redundant']/pred_df['unlabeled_redundant'].mean()

        pred_df['model_redundant_density'] = (pred_df['model_redundant_score'])/(pred_df['unlabeled_redundant']+eps)
        pred_df['model_score'] = (pred_df['unlabeled_redundant'])/(pred_df['model_redundant_score']+eps)

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(),pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Redundant',pred_df.groupby('labels')['unlabeled_redundant'].min(),pred_df.groupby('labels')['unlabeled_redundant'].mean(), pred_df.groupby('labels')['unlabeled_redundant'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Current power factor: ',labeled_pow_factor, unlabeled_pow_factor)
        print('median power: ',labeled_mean_pred_score ,unlabeled_mean_pred_score)


        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))


        # # filter 2
        pred_df['model_score'] = pred_df['model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        redundant_frame_id = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index(drop=False).groupby('frame_id').model_score.mean().reset_index(drop=False).sort_values(by='model_score')
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_score>redundant_frame_id.model_score.quantile(7/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        
        # filter 1
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.mean().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(1/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        

        #filter by label entropy
        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()
        

        

        if i == 0 or i == total_select_nums-1:
            print('==== select score ====')
            print(pred_df.groupby('labels').select_score.median())
            print('==== confidence ====')
            print(pred_df.groupby('labels').confidence.median())
            print('==== redundant ====')
            print(pred_df.groupby('labels').model_redundant.median())
        

        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
        return selected_frames

# labeled/unlabeled Dynamic DR 30 -> 10 eps 1e-4 clip score 100
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV8Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV8Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred

            pred_list.append(data_df)
            
        
        pred_df = pd.concat(pred_list)

        eps=1e-5
        mean_target = 0.3
        labeled_mean_pred_score = min(np.mean(pred_df['model_pred']),-0.01)
        labeled_pow_factor = min(np.power(mean_target,1/labeled_mean_pred_score),np.exp(1))

        unlabeled_mean_pred_score = min(np.mean(pred_df['unlabeled_pred']),-0.01)
        unlabeled_pow_factor = min(np.power(mean_target,1/unlabeled_mean_pred_score),np.exp(1))
        
        pred_df['model_redundant'] = np.power(labeled_pow_factor ,pred_df['model_pred'])*pred_df['confidence']
        pred_df['model_redundant_score'] = np.power(labeled_pow_factor,pred_df['model_pred'])
        pred_df['model_redundant_score'] = pred_df['model_redundant_score']/pred_df['model_redundant_score'].mean()

        pred_df['unlabeled_redundant'] = np.power(unlabeled_pow_factor,pred_df['unlabeled_pred'])
        pred_df['unlabeled_redundant'] = pred_df['unlabeled_redundant']/pred_df['unlabeled_redundant'].mean()

        pred_df['model_score'] = (pred_df['unlabeled_redundant'])/(pred_df['model_redundant_score']+eps)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        pred_df['model_score'] = pred_df['model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        
        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(),pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Redundant',pred_df.groupby('labels')['unlabeled_redundant'].min(),pred_df.groupby('labels')['unlabeled_redundant'].mean(), pred_df.groupby('labels')['unlabeled_redundant'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Current power factor: ',labeled_pow_factor, unlabeled_pow_factor)
        print('median power: ',labeled_mean_pred_score ,unlabeled_mean_pred_score)

        saving_dict['feature_df'] = pred_df
        saving_dict['mean_target'] = mean_target
        saving_dict['eps']=eps
        
        
        # filter 1
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.mean().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(4/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        # filter 2
        redundant_frame_id = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index(drop=False).groupby('frame_id').model_score.mean().reset_index(drop=False).sort_values(by='model_score')
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_score>redundant_frame_id.model_score.quantile(3/4)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        #filter by label entropy
        selected_frames=[]
        _df = pred_df.groupby(['frame_id','labels']).confidence.sum()
        index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        select_confidence = np.array([0,0,0])

        for i in range(total_select_nums):
            entropy_df = pd.DataFrame()
            _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
            entropy_df['frame_id'] = _df.frame_id.unique()
            entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).confidence.apply(list).tolist()+select_confidence,base=3,axis=1)
            selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
            select_confidence = pred_df[pred_df.frame_id.isin(selected_frames)].groupby('labels').confidence.sum().values

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(),pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Redundant',pred_df.groupby('labels')['unlabeled_redundant'].min(),pred_df.groupby('labels')['unlabeled_redundant'].mean(), pred_df.groupby('labels')['unlabeled_redundant'].max())
        print('Selected Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Selected Confidence',pred_df.groupby('labels')['confidence'].min(), pred_df.groupby('labels')['confidence'].median(),pred_df.groupby('labels')['confidence'].max(),pred_df.groupby('labels')['confidence'].sum())


        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())

        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(saving_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        return selected_frames

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV9Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV9Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2)
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                data_df[f'tsne_{i}'] = tsne_feature[:,i]
            print(f"--- T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))

            ## ============= feature processing ============= 
            if 'model_pred' in data_df.columns:
                data_df = data_df.drop(columns=['model_pred'])
            feature_cols = list(data_df.columns.drop(['frame_id','embeddings','logits','labels','Set','labeled','cls_entropy','confidence','cls_std','reg_std']))
            id_col = 'frame_id'
            label_col = 'labeled'
            # qt = QuantileTransformer(n_quantiles=10, random_state=0)
            # data_df['cls_entropy'] = qt.fit_transform(data_df[['cls_entropy']].values)[:,0]
            scaler = StandardScaler()
            data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])
            feature_df_list.append(data_df)

        selected_frames=[]
        feature_df = pd.concat(feature_df_list)
        select_start_time = time.time()
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            train_data = training_data_df[feature_cols]
            train_label = training_data_df[label_col]
            validation_data = validation_data_df[feature_cols]
            validation_label = validation_data_df[label_col]
            
            model_start_time = time.time()
            ## ============= prepare training data =============
            
            components = len(feature_cols)*3
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred

            pred_list.append(data_df)
            
        
        pred_df = pd.concat(pred_list)
        eps=1e-3
        alpha=1.2

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        pred_df['model_redundant'] = np.exp(pred_df.model_pred)
        pred_df['model_score'] = (pred_df['unlabeled_pred']-pred_df['model_pred'])

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('Selected Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Selected Confidence',pred_df.groupby('labels')['confidence'].min(), pred_df.groupby('labels')['confidence'].median(),pred_df.groupby('labels')['confidence'].max(),pred_df.groupby('labels')['confidence'].sum())


        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())

        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(saving_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        return selected_frames


class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV10Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV10Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]

            train_data = np.array(training_data_df.embeddings.tolist())
            validation_data = np.array(validation_data_df.embeddings.tolist())

            
            
            ## ============= prepare training data =============
            
            components = 5
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)
        eps=1e-3
        alpha=1.2

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        pred_df['model_redundant'] = np.exp(pred_df.model_pred)
        pred_df['model_score'] = (pred_df['unlabeled_pred']-pred_df['model_pred'])

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('Selected Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Selected Confidence',pred_df.groupby('labels')['confidence'].min(), pred_df.groupby('labels')['confidence'].median(),pred_df.groupby('labels')['confidence'].max(),pred_df.groupby('labels')['confidence'].sum())


        if self.rank == 0:
            pbar.close()

        print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
        print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())

        # save Embedding    
        with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
            pickle.dump(saving_dict, f)
            print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
        return selected_frames


class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV11Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV11Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        embeddings = np.array(feature_df.embeddings.tolist())
        pca = PCA()
        pca_feature = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        num_dim = sum(np.cumsum(exp_var_pca)<0.9)
        pca_feature = pca_feature[:,:num_dim]
        print('num dum: {}'.format(num_dim))
                
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    
            instance_pca_feature = pca_feature[feature_df.labels == cidx,:]

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]

            train_data = instance_pca_feature[training_criteria,:]
            validation_data = instance_pca_feature[~training_criteria,:]

            
            
            ## ============= prepare training data =============
            
            components = 5
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df=data_df[~training_criteria]
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)
        eps=1e-3

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        pred_df['model_redundant'] = np.exp(pred_df.model_pred)
        pred_df['model_score'] = (pred_df['unlabeled_pred']-pred_df['model_pred'])

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.mean().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant<redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('Selected Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Selected Confidence',pred_df.groupby('labels')['confidence'].min(), pred_df.groupby('labels')['confidence'].median(),pred_df.groupby('labels')['confidence'].max(),pred_df.groupby('labels')['confidence'].sum())


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

# 30 -> 10 instance+box
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV12Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV12Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        embeddings = np.array(feature_df.embeddings.tolist())
        pca = PCA()
        pca_feature = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        num_dim = sum(np.cumsum(exp_var_pca)<0.9)
        pca_feature = pca_feature[:,:num_dim]
        print('num dum: {}'.format(num_dim))
                
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    
            instance_pca_feature = pca_feature[feature_df.labels == cidx,:]

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria]#.sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            data_df=data_df[~training_criteria]
            
            
            ## ============= instance feature training data =============
            train_data = instance_pca_feature[training_criteria,:]
            validation_data = instance_pca_feature[~training_criteria,:]
            
            components = 10
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred


            ## ============= bbox feature training data =============

            bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']

            train_data = training_data_df[bbox_feature_cols]
            validation_data = validation_data_df[bbox_feature_cols]
            
            components = 5
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['bbox_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['bbox_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)
        eps=1e-3

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        # pred_df['instance_feature_model_pred'] = pred_df['instance_feature_model_pred']/pred_df['instance_feature_model_pred'].std()
        # pred_df['instance_feature_unlabeled_pred'] = pred_df['instance_feature_unlabeled_pred']/pred_df['instance_feature_unlabeled_pred'].std()
        # pred_df['bbox_feature_model_pred'] = pred_df['bbox_feature_model_pred']/pred_df['bbox_feature_model_pred'].std()
        # pred_df['bbox_feature_unlabeled_pred'] = pred_df['bbox_feature_unlabeled_pred']/pred_df['bbox_feature_unlabeled_pred'].std()

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['bbox_model_redundant'] = np.exp(pred_df['bbox_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = pred_df['instance_model_score']/pred_df['instance_model_score'].std()
        pred_df['bbox_model_score'] = (pred_df['bbox_feature_unlabeled_pred']-pred_df['bbox_feature_model_pred'])
        pred_df['bbox_model_score'] = pred_df['bbox_model_score']/pred_df['bbox_model_score'].std()
        pred_df['model_score'] = (pred_df['instance_model_score']+pred_df['bbox_model_score'])*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nBBox Model Pred\n',pred_df.groupby('labels')['bbox_feature_model_pred'].min(), pred_df.groupby('labels')['bbox_feature_model_pred'].mean(), pred_df.groupby('labels')['bbox_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nBBox Unlabeled Pred\n',pred_df.groupby('labels')['bbox_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['bbox_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['bbox_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nBBox Score\n',pred_df.groupby('labels')['bbox_model_score'].min(), pred_df.groupby('labels')['bbox_model_score'].median(),pred_df.groupby('labels')['bbox_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nBBox Model Pred\n',pred_df.groupby('labels')['bbox_feature_model_pred'].min(), pred_df.groupby('labels')['bbox_feature_model_pred'].mean(), pred_df.groupby('labels')['bbox_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nBBox Unlabeled Pred\n',pred_df.groupby('labels')['bbox_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['bbox_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['bbox_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nBBox Score\n',pred_df.groupby('labels')['bbox_model_score'].min(), pred_df.groupby('labels')['bbox_model_score'].median(),pred_df.groupby('labels')['bbox_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())
        print('Selected Confidence',pred_df.groupby('labels')['confidence'].min(), pred_df.groupby('labels')['confidence'].median(),pred_df.groupby('labels')['confidence'].max(),pred_df.groupby('labels')['confidence'].sum())


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

# 30 -> 10 ul-l component 5 combine feature
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV13Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV13Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        embeddings = np.array(feature_df.embeddings.tolist())
        pca = PCA()
        pca_feature = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        num_dim = sum(np.cumsum(exp_var_pca)<0.9)
        pca_feature = pca_feature[:,:num_dim]
        print('num dum: {}'.format(num_dim))
                
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    
            instance_pca_feature = pca_feature[feature_df.labels == cidx,:]

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            data_df=data_df[~training_criteria]
            
            
            ## ============= instance feature training data =============
            
            bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
            instance_train_data = instance_pca_feature[training_criteria,:]
            instance_validation_data = instance_pca_feature[~training_criteria,:]

            bbox_train_data = training_data_df[bbox_feature_cols].values
            bbox_validation_data = validation_data_df[bbox_feature_cols].values

            train_data = np.concatenate([instance_train_data,bbox_train_data],axis=1)
            validation_data = np.concatenate([instance_validation_data,bbox_validation_data],axis=1)
            
            components = 5
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred


            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)
        eps=1e-3

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))

        pred_df['model_redundant'] = np.exp(pred_df['model_pred'])
        pred_df['model_score'] = (pred_df['unlabeled_pred']-pred_df['model_pred'])#*pred_df['related_confidence_weighted']

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.mean().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

# 50 -> 10 ul-l rw cw dynamic component var0.9 combine feature
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV14Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV14Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
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
                        record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
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
                        record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
                        record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
                        record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
                        record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            for k,v in df.items():
                if 'frame' not in str(k):
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
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
        pred_list=[]
        embeddings = np.array(feature_df.embeddings.tolist())
        pca = PCA()
        pca_feature = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        num_dim = sum(np.cumsum(exp_var_pca)<0.9)
        pca_feature = pca_feature[:,:num_dim]
        print('num dum: {}'.format(num_dim))
                
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    
            instance_pca_feature = pca_feature[feature_df.labels == cidx,:]

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria].sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            data_df=data_df[~training_criteria]
            
            
            ## ============= instance feature training data =============
            
            bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
            instance_train_data = instance_pca_feature[training_criteria,:]
            instance_validation_data = instance_pca_feature[~training_criteria,:]

            bbox_train_data = training_data_df[bbox_feature_cols].values
            bbox_validation_data = validation_data_df[bbox_feature_cols].values

            train_data = np.concatenate([instance_train_data,bbox_train_data],axis=1)
            validation_data = np.concatenate([instance_validation_data,bbox_validation_data],axis=1)

            # select number of component
            n_components_range = range(10,110,10)
            lowest_bic=1e9
            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, covariance_type='full',tol=1e-2)
                gmm.fit(validation_data)
                bic_score=gmm.bic(validation_data) 
                if bic_score < lowest_bic:
                    lowest_bic = bic_score
                else:
                    break
            print('number of component: {}'.format(n_components))
            model=mixture.GaussianMixture(n_components=n_components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['model_pred'] = pred
            

            model=mixture.GaussianMixture(n_components=n_components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred


            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)
        eps=1e-3

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))

        pred_df['model_redundant'] = np.exp(pred_df['model_pred'])
        pred_df['model_score'] = (pred_df['unlabeled_pred']-pred_df['model_pred'])

        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

# # 50 -> 10 ul-l rw cw dynamic component combine var0.8 feature
# class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV15Smpling(Strategy):
#     def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
#         super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV15Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

#     def pairwise_squared_distances(self, x, y):
#         # x: N * D * D * 1
#         # y: M * D * D * 1
#         # return: N * M    
#         assert (len(x.shape)) > 1
#         assert (len(y.shape)) > 1  
#         n = x.shape[0]
#         m = y.shape[0]
#         x = x.view(n, -1)
#         y = y.view(m, -1)
                                                              
#         x_norm = (x**2).sum(1).view(n, 1)
#         y_t = y.permute(1, 0).contiguous()
#         y_norm = (y**2).sum(1).view(1, m)
#         dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
#         dist[dist != dist] = 0 # replace nan values with 0
#         return torch.clamp(dist, 0.0, float('inf'))

#     def select_func(self):
#         pass
    
#     def enable_dropout(self, model, enable=True):
#         """ Function to enable the dropout layers during test-time """
#         i = 0
#         for m in model.modules():
#             if m.__class__.__name__.startswith('Dropout'):
#                 i += 1
#                 m.train()
#         print('**found and enabled {} Dropout layers for random sampling**'.format(i))

#     def query(self, leave_pbar=True, cur_epoch=None):
#         def divNum(num, parts):
#             p = int(num/parts)
#             missing = int(num - p*parts)             
#             return [p+1]*missing + [p]*(parts-missing)
        
#         class_names = self.cfg.CLASS_NAMES
#         total_select_nums = self.cfg.ACTIVE_TRAIN.SELECT_NUMS
#         cls_select_nums = divNum(total_select_nums,len(class_names))
        
#         self.model.eval()
#         record_dict=defaultdict(list)
        

#         # feed unlabeled data forward the model
#         total_it_each_epoch = len(self.unlabelled_loader)
#         if self.rank == 0:
#             pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
#                              desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
#         self.model.eval()
#         self.enable_dropout(self.model)
#         # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
#         val_dataloader_iter = iter(self.unlabelled_loader)
#         val_loader = self.unlabelled_loader
#         for cur_it in range(total_it_each_epoch):
#             try:
#                 unlabelled_batch = next(val_dataloader_iter)
#             except StopIteration:
#                 unlabelled_dataloader_iter = iter(val_loader)
#                 unlabelled_batch = next(unlabelled_dataloader_iter)
#             with torch.no_grad():
#                 load_data_to_gpu(unlabelled_batch)
#                 pred_dicts, _ = self.model(unlabelled_batch)
#                 for batch_inx in range(len(pred_dicts)):
                    
#                     selected = pred_dicts[batch_inx]['selected']
#                     pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
#                     self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
#                     # record_dict['unlabeled_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
#                     for instance_idx in range(len(pred_instance_labels)):
#                         record_dict['unlabeled_frame_id'].append(unlabelled_batch['frame_id'][batch_inx])
#                         record_dict['unlabeled_labels'].append(pred_instance_labels[instance_idx])
#                         record_dict['unlabeled_rotation'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-1])
#                         record_dict['unlabeled_height_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-4])
#                         record_dict['unlabeled_width_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-3])
#                         record_dict['unlabeled_length_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-2])
#                         record_dict['unlabeled_pts'].append(pred_dicts[batch_inx]['pts_count'][instance_idx])
#                         record_dict['unlabeled_box_volumes'].append(pred_dicts[batch_inx]['box_volumes'][instance_idx])
#                         record_dict['unlabeled_embeddings'].append(pred_dicts[batch_inx]['pred_embeddings'][instance_idx].view(-1))
#                         record_dict['unlabeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
#                         record_dict['unlabeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
#                         record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
#                         record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
#                         record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
#                         record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
#             if self.rank == 0:
#                 pbar.update()
#                 # pbar.set_postfix(disp_dict)
#                 pbar.refresh()
#         if self.rank == 0:
#             pbar.close()
            
            
#         ## Label
#         # feed labeled data forward the model
#         total_it_each_epoch = len(self.labelled_loader)
#         if self.rank == 0:
#             pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
#                              desc='evaluating_labelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
#         # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
#         val_dataloader_iter = iter(self.labelled_loader)
#         val_loader = self.labelled_loader
#         for cur_it in range(total_it_each_epoch):
#             try:
#                 labelled_batch = next(val_dataloader_iter)
#             except StopIteration:
#                 labelled_dataloader_iter = iter(val_loader)
#                 labelled_batch = next(labelled_dataloader_iter)
#             with torch.no_grad():
#                 load_data_to_gpu(labelled_batch)
#                 pred_dicts, _ = self.model(labelled_batch)
                
#                 for batch_inx in range(len(pred_dicts)):
#                     pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
#                     # record_dict['labeled_frame_ids'].append(labelled_batch['frame_id'][batch_inx])
#                     for instance_idx in range(len(pred_instance_labels)):
#                         record_dict['labeled_frame_id'].append(labelled_batch['frame_id'][batch_inx])
#                         record_dict['labeled_labels'].append(pred_instance_labels[instance_idx])
#                         record_dict['labeled_rotation'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-1])
#                         record_dict['labeled_height_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-4])
#                         record_dict['labeled_width_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-3])
#                         record_dict['labeled_length_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-2])
#                         record_dict['labeled_pts'].append(pred_dicts[batch_inx]['pts_count'][instance_idx])
#                         record_dict['labeled_box_volumes'].append(pred_dicts[batch_inx]['box_volumes'][instance_idx])
#                         record_dict['labeled_embeddings'].append(pred_dicts[batch_inx]['pred_embeddings'][instance_idx].view(-1))
#                         record_dict['labeled_reg_std'].append(pred_dicts[batch_inx]['pred_rcnn_reg_std'][instance_idx])
#                         record_dict['labeled_cls_std'].append(pred_dicts[batch_inx]['pred_rcnn_cls_std'][instance_idx])
#                         record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
#                         record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
#                         record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
#                         record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
#             if self.rank == 0:
#                 pbar.update()
#                 # pbar.set_postfix(disp_dict)
#                 pbar.refresh()
#         if self.rank == 0:
#             pbar.close()
            
#         record_dict = dict(record_dict)
        
#         print('** [Instance] start searching...**')
        
#         ## ============= process data ============= 
#         def process_df(df):
#             for k,v in df.items():
#                 if 'frame' not in str(k):
#                     df[k] = [i.cpu().numpy() for i in v]
#             return df
       
#         df = process_df(record_dict)
        
#         unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
#         labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
#         unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
#         labeled_df = pd.DataFrame.from_dict(labeled_dict)

#         unlabeled_df['Set'] = 'unlabeled'
#         labeled_df['Set'] = 'labeled'
#         unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
#         labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
#         combine_df = pd.concat([unlabeled_df,labeled_df])

#         for col in combine_df.columns:
#             if 'labels' in str(col) :
#                 combine_df[col] = combine_df[col].astype(int)
#             elif str(col) not in ['embeddings','logits','Set','frame_id']:
#                 combine_df[col] = combine_df[col].astype(float)
            
#         combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
#         ## Select each class

#         feature_df = combine_df.copy()
#         selected_frames=[]
#         saving_dict = dict()
#         saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
  
#         pred_list=[]
#         embeddings = np.array(feature_df.embeddings.tolist())
#         pca = PCA()
#         pca_feature = pca.fit_transform(embeddings)
#         exp_var_pca = pca.explained_variance_ratio_
#         num_dim = sum(np.cumsum(exp_var_pca)<0.8)
#         pca_feature = pca_feature[:,:num_dim]
#         print('num dum: {}'.format(num_dim))
                
#         for c_idx, select_num in enumerate(cls_select_nums):  
#             gmm_running = time.time()
#             cidx = c_idx+1
#             data_df = feature_df[feature_df.labels == cidx]    
#             instance_pca_feature = pca_feature[feature_df.labels == cidx,:]

#             ## ============= prepare training data ============= 
#             training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
#             training_data_df = data_df[training_criteria]
#             validation_data_df = data_df[~training_criteria]
#             data_df=data_df[~training_criteria]
            
            
#             ## ============= instance feature training data =============
            
#             bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
#             instance_train_data = instance_pca_feature[training_criteria,:]
#             instance_validation_data = instance_pca_feature[~training_criteria,:]

#             bbox_train_data = training_data_df[bbox_feature_cols].values
#             bbox_validation_data = validation_data_df[bbox_feature_cols].values

#             train_data = np.concatenate([instance_train_data,bbox_train_data],axis=1)
#             validation_data = np.concatenate([instance_validation_data,bbox_validation_data],axis=1)

#             # select number of component
#             n_components=10

#             print('number of component: {}'.format(n_components))
#             model=mixture.GaussianMixture(n_components=n_components, init_params='k-means++',random_state=3131)
#             model.fit(train_data)
#             pred = model.score_samples(validation_data)
#             data_df['model_pred'] = pred



#             model=mixture.GaussianMixture(n_components=n_components, init_params='k-means++',random_state=3131)
#             model.fit(validation_data)
#             pred = model.score_samples(validation_data)
#             data_df['unlabeled_pred'] = pred


#             pred_list.append(data_df)
#             print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
#         select_start_time = time.time()
#         pred_df = pd.concat(pred_list)

#         pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
#         pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))

#         pred_df['model_redundant'] = np.exp(pred_df['model_pred'])
#         pred_df['model_score'] = (pred_df['unlabeled_pred']-pred_df['model_pred'])*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']

#         print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
#         print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
#         print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
#         print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

#         saving_dict['feature_df'] = pred_df

#         # redundant filtering
#         redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
#         redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(5/10)].frame_id
#         pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


#         # filter 1
#         frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
#         redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(4/5)].frame_id
#         pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
#         frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

#         label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
#         selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


#         # #filter by label entropy
#         # selected_frames=[]
#         # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
#         # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
#         # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
#         # select_confidence = np.array([0,0,0])

#         # for i in range(total_select_nums):
#         #     if i % 2 == 1:
#         #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
#         #         entropy_df = pd.DataFrame()
#         #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
#         #         entropy_df['frame_id'] = _df.frame_id.unique()
#         #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
#         #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
#         #     else:
#         #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
#         pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
#         frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
#         saving_dict['selected_frames']=selected_frames
#         print('\n\n============== Selected =================\n')
#         print('Model Pred',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
#         print('Model Redundant',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
#         print('Unlabeled Pred',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
#         print('Model Score',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


#         if self.rank == 0:
#             pbar.close()

#         print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
#         print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
#         print(len(set(selected_frames)))
#         # save Embedding    
#         with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
#             pickle.dump(saving_dict, f)
#             print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
#         return selected_frames

# 50 -> 10 ul-l rw cw dynamic component combine var0.8 feature
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV15Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV15Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)

                    record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)

        # embedding 
        feature_df_list=[]
        tsne_feature_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):
            cidx = c_idx+1
            data_df = combine_df[combine_df.labels == cidx]
            ## ============= t-sne ============= 
            start_time = time.time()
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
            tsne_feature = tsne.fit_transform(np.array(data_df['embeddings'].tolist()))

            for i in range(tsne_feature.shape[1]):
                new_col=f'instance_tsne_{i}'
                data_df[new_col] = tsne_feature[:,i]
                feature_columns.append(new_col)

            feature_df_list.append(data_df)
            print(f"--- Embedding T-SNE {cidx} running time: %s seconds ---" % (time.time() - start_time))
        
        feature_df = pd.concat(feature_df_list)
        

        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_feature'],spatial_dict['unlabeled_spatial_feature']])
        tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
        combine_spatial_features = tsne.fit_transform(combine_spatial_features)
        
        
        #compute frame level score
        spatial_dict['labeled_spatial_feature'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_feature']),:]
        spatial_dict['unlabeled_spatial_feature'] = combine_spatial_features[len(spatial_dict['labeled_spatial_feature']):,:]
        
        for i in range(2):
            new_col=f'spatial_tsne_{i}'
            spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['unlabeled_spatial_feature'][:,i]))
            spatial_unlabeled_pred_mapping_dict.update(dict(zip(spatial_dict['labeled_spatial_frame_ids'], spatial_dict['labeled_spatial_feature'][:,i])))
            feature_df[new_col] = feature_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            feature_columns.append(new_col)

        feature_df[feature_columns] = StandardScaler().fit_transform(feature_df[feature_columns].values)

        components = 20
        pred_list=[]
        # compute instance level score
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]    

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            training_data_df = data_df[training_criteria]#.sample(frac=1)
            validation_data_df = data_df[~training_criteria]
            data_df=data_df[~training_criteria]
            

            ## ============= bbox feature training data =============

            train_data = training_data_df[feature_columns]
            validation_data = validation_data_df[feature_columns]
            
            model=mixture.GaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['model_pred'] = pred
            

            model=mixture.GaussianMixture(n_components=components, init_params='k-means++',random_state=3131)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['model_redundant'] =  np.exp(pred_df['model_pred'])
        pred_df['model_score']     = (pred_df['unlabeled_pred']-pred_df['model_pred'])*pred_df['class_confidence_weighted']

        print('\nModel Pred\n',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('\nUnlabeled Pred\n',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['pred_df'] = pred_df
        saving_dict['feature_df'] = feature_df
        saving_dict['spatial_dict'] = spatial_dict

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])
        # selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)
        # for i in range(total_select_nums):
        #     if i % 1 == 0:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nModel Pred\n',pred_df.groupby('labels')['model_pred'].min(), pred_df.groupby('labels')['model_pred'].mean(), pred_df.groupby('labels')['model_pred'].max())
        print('\nUnlabeled Pred\n',pred_df.groupby('labels')['unlabeled_pred'].min(), pred_df.groupby('labels')['unlabeled_pred'].mean(), pred_df.groupby('labels')['unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV16Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV16Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = torch.mean(pred_dicts[batch_inx]['spatial_features_2d'], dim=0)
                    record_dict['unlabeled_spatial_bev_features'].append(spatial_features_2d)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = torch.mean(pred_dicts[batch_inx]['spatial_features_2d'], dim=0)
                    record_dict['labeled_spatial_bev_features'].append(spatial_features_2d)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}

        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        saving_dict['spatial_dict'] = spatial_dict
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)

        # Spatial Feature Processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_bev_features'],spatial_dict['unlabeled_spatial_bev_features']],axis=0)
        for i in range(combine_spatial_features.shape[1]):
            scalers = StandardScaler()
            combine_spatial_features[:, i, :] = scalers.fit_transform(combine_spatial_features[:, i, :])
        combine_spatial_features = combine_spatial_features.reshape(combine_spatial_features.shape[0],-1).T
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features).T.reshape(-1,spatial_dict['labeled_spatial_bev_features'].shape[1],spatial_dict['labeled_spatial_bev_features'].shape[2])
        
        height_spatial_features = torch.tensor(np.mean(combine_spatial_features,axis=1))
        width_spatial_features = torch.tensor(np.mean(combine_spatial_features,axis=2))
       
        # height_spatial_features = adaptive_avg_pool1d(height_spatial_features,int(height_spatial_features.shape[-1]//10))
        # width_spatial_features = adaptive_avg_pool1d(width_spatial_features,int(width_spatial_features.shape[-1]//10))

        
        height_spatial_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(height_spatial_features)
        height_spatial_features = StandardScaler().fit_transform(height_spatial_features)
        width_spatial_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(width_spatial_features)
        width_spatial_features = StandardScaler().fit_transform(width_spatial_features)
        combine_spatial_features = np.concatenate([height_spatial_features,width_spatial_features],axis=1)

        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_bev_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_bev_features']):,:]
        
        
        ## spatial feature processing
        # combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_feature'],spatial_dict['unlabeled_spatial_feature']])
        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        
        components = 10 
        
        #compute frame level score
        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        
        # spatial_feaures_mapping_dict = dict(zip(spatial_dict['labeled_spatial_frame_ids'], spatial_dict['labeled_spatial_feature']))
        # spatial_feaures_mapping_dict.update(dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['unlabeled_spatial_feature'])))
        # feature_df['spatial_features'] = feature_df.frame_id.map(spatial_feaures_mapping_dict)


        # Instance Level 
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_feature = StandardScaler().fit_transform(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            train_data = features[training_criteria]#.sample(frac=1)
            validation_data = features[~training_criteria]
            data_df=data_df[~training_criteria]
            
            ## ============= instance feature training data =============
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
        pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        alpha = 1.2
        eps = 1e-3
        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = np.exp((pred_df['instance_model_score'])/(pred_df['instance_model_score'].quantile(.75)-pred_df['instance_model_score'].quantile(.25)))
        pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
        pred_df['spatial_model_score']  = np.exp((pred_df['spatial_model_score'])/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25)))
        pred_df['model_score']          = pred_df['spatial_model_score']*pred_df['instance_model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
        # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV17Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV17Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = torch.mean(pred_dicts[batch_inx]['spatial_features_2d'], dim=0)
                    record_dict['unlabeled_spatial_bev_features'].append(spatial_features_2d)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = torch.mean(pred_dicts[batch_inx]['spatial_features_2d'], dim=0)
                    record_dict['labeled_spatial_bev_features'].append(spatial_features_2d)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}

        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)

        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        saving_dict['spatial_dict'] = spatial_dict
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)

        # Spatial Feature Processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_bev_features'],spatial_dict['unlabeled_spatial_bev_features']],axis=0)
        for i in range(combine_spatial_features.shape[1]):
            scalers = StandardScaler()
            combine_spatial_features[:, i, :] = scalers.fit_transform(combine_spatial_features[:, i, :]) 
        height_spatial_features = np.mean(combine_spatial_features,axis=1)
        height_spatial_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(height_spatial_features)
        height_spatial_features = StandardScaler().fit_transform(height_spatial_features)
        width_spatial_features = np.mean(combine_spatial_features,axis=2)
        width_spatial_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(width_spatial_features)
        width_spatial_features = StandardScaler().fit_transform(width_spatial_features)
        combine_spatial_features = np.concatenate([height_spatial_features,width_spatial_features],axis=1)

        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_bev_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_bev_features']):,:]
        
        
        ## spatial feature processing
        # combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_feature'],spatial_dict['unlabeled_spatial_feature']])
        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        
        components = 10 
        
        #compute frame level score
        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        
        # spatial_feaures_mapping_dict = dict(zip(spatial_dict['labeled_spatial_frame_ids'], spatial_dict['labeled_spatial_feature']))
        # spatial_feaures_mapping_dict.update(dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['unlabeled_spatial_feature'])))
        # feature_df['spatial_features'] = feature_df.frame_id.map(spatial_feaures_mapping_dict)


        # Instance Level 
        pred_list=[]
        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca')
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_feature = StandardScaler().fit_transform(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            train_data = features[training_criteria]#.sample(frac=1)
            validation_data = features[~training_criteria]
            data_df=data_df[~training_criteria]
            
            ## ============= instance feature training data =============
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
        pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        alpha = 1.2
        eps = 1e-3
        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = np.exp((pred_df['instance_model_score'])/(pred_df['instance_model_score'].quantile(.75)-pred_df['instance_model_score'].quantile(.25)))
        pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
        pred_df['spatial_model_score']  = np.exp((pred_df['spatial_model_score'])/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25)))
        pred_df['model_score']          = pred_df['instance_model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']*pred_df['spatial_model_score']
        # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
        # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV18Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV18Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)

                    record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_feature'],spatial_dict['unlabeled_spatial_feature']])
        embeddings = np.array(combine_spatial_features)
        pca = PCA()
        pca_feature = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        print('num dum: {}'.format(num_dim))

        pred_list=[]
        components = 10 
        
        #compute frame level score
        spatial_dict['labeled_spatial_feature'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_feature']),:]
        spatial_dict['unlabeled_spatial_feature'] = combine_spatial_features[len(spatial_dict['labeled_spatial_feature']):,:]

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_feature'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_feature'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_feature'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_feature'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_feature = StandardScaler().fit_transform(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            train_data = features[training_criteria]#.sample(frac=1)
            validation_data = features[~training_criteria]
            data_df=data_df[~training_criteria]
            
            ## ============= instance feature training data =============
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
        pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        alpha = 1.2
        eps = 1e-3
        # pred_df['instance_model_score'] = (np.power(alpha, pred_df['instance_feature_unlabeled_pred'])+eps)/(np.power(alpha, pred_df['instance_feature_model_pred'])+eps)
        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score']/(pred_df['instance_model_score'].quantile(.75)-pred_df['instance_model_score'].quantile(.25)))
        pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
        pred_df['spatial_model_score']  = np.exp(pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25)))
        pred_df['model_score']          = pred_df['instance_model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
        # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

# 30 -> 10 instance+box
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV19Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV19Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)

                    record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_feature'],spatial_dict['unlabeled_spatial_feature']])
        embeddings = np.array(combine_spatial_features)
        pca = PCA()
        pca_feature = pca.fit_transform(embeddings)
        exp_var_pca = pca.explained_variance_ratio_
        num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        print('num dum: {}'.format(num_dim))

        pred_list=[]
        components = 10 
        
        #compute frame level score
        spatial_dict['labeled_spatial_feature'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_feature']),:]
        spatial_dict['unlabeled_spatial_feature'] = combine_spatial_features[len(spatial_dict['labeled_spatial_feature']):,:]

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_feature'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_feature'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_feature'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_feature'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_feature = StandardScaler().fit_transform(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            train_data = features[training_criteria]#.sample(frac=1)
            validation_data = features[~training_criteria]
            data_df=data_df[~training_criteria]
            
            ## ============= instance feature training data =============
            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
        pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        alpha = 1.2
        eps = 1e-3
        # pred_df['instance_model_score'] = (np.power(alpha, pred_df['instance_feature_unlabeled_pred'])+eps)/(np.power(alpha, pred_df['instance_feature_model_pred'])+eps)
        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score']/(pred_df['instance_model_score'].quantile(.75)-pred_df['instance_model_score'].quantile(.25)))
        pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
        # pred_df['spatial_model_score']  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
        pred_df['model_score']          = pred_df['instance_model_score']*pred_df['class_confidence_weighted']*pred_df['related_confidence_weighted']
        # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
        # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(5/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(4/5)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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


class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV20Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV20Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        # combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_feature'],spatial_dict['unlabeled_spatial_feature']])
        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        pred_list=[]
        components = 10
        
        # #compute frame level score
        # spatial_dict['labeled_spatial_feature'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_feature']),:]
        # spatial_dict['unlabeled_spatial_feature'] = combine_spatial_features[len(spatial_dict['labeled_spatial_feature']):,:]

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_feature = StandardScaler().fit_transform(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            train_data = features[training_criteria]#.sample(frac=1)
            validation_data = features[~training_criteria]
            data_df=data_df[~training_criteria]
            
            ## ============= instance feature training data =============
            model=mixture.BayesianGaussianMixture(n_components=features.shape[1], init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=features.shape[1], init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
        pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        alpha = 1.2
        eps = 1e-3
        # pred_df['instance_model_score'] = (np.power(alpha, pred_df['instance_feature_unlabeled_pred'])+eps)/(np.power(alpha, pred_df['instance_feature_model_pred'])+eps)
        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score']/(pred_df['instance_model_score'].quantile(.75)-pred_df['instance_model_score'].quantile(.25)))
        pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
        pred_df['spatial_model_score']  = np.exp(pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25)))
        pred_df['model_score']          = pred_df['spatial_model_score'] #pred_df['instance_model_score']*pred_df['related_confidence_weighted']*pred_df['class_confidence_weighted']
        # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
        # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV21Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV21Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        # combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        pred_list=[]
        components = 10
        
        # #compute frame level score
        # spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        # spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        for c_idx, select_num in enumerate(cls_select_nums):  
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_feature = StandardScaler().fit_transform(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            ## ============= prepare training data ============= 
            training_criteria = (data_df.labeled == 1)  #| (data_df.cluster_center == 1)
            train_data = features[training_criteria]#.sample(frac=1)
            validation_data = features[~training_criteria]
            data_df=data_df[~training_criteria]
            
            ## ============= instance feature training data =============
            model=mixture.BayesianGaussianMixture(n_components=features.shape[1], init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(train_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_model_pred'] = pred
            

            model=mixture.BayesianGaussianMixture(n_components=features.shape[1], init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(validation_data)
            pred = model.score_samples(validation_data)
            data_df['instance_feature_unlabeled_pred'] = pred

            pred_list.append(data_df)
            print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
            
        select_start_time = time.time()
        pred_df = pd.concat(pred_list)

        pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
        pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
        
        pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
        pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

        pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
        pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
        pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

        alpha = 1.2
        eps = 1e-3
        # pred_df['instance_model_score'] = (np.power(alpha, pred_df['instance_feature_unlabeled_pred'])+eps)/(np.power(alpha, pred_df['instance_feature_model_pred'])+eps)
        pred_df['instance_model_score'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
        pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score']/(pred_df['instance_model_score'].quantile(.75)-pred_df['instance_model_score'].quantile(.25)))
        pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
        pred_df['spatial_model_score']  = np.exp(pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25)))
        pred_df['model_score']          = pred_df['instance_model_score']*pred_df['related_confidence_weighted']*pred_df['class_confidence_weighted']
        pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].median()/pred_df['model_score'].median()) + pred_df['spatial_model_score']
        # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
        # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

        saving_dict['feature_df'] = pred_df
        saving_dict['eps']=eps

        # redundant filtering
        redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
        redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


        # filter 1
        frame_df = pred_df.groupby(['frame_id','labels']).model_score.mean().reset_index().groupby(['frame_id']).model_score.mean().reset_index()
        redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(2/3)].frame_id
        pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
        frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

        label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='counts').groupby('frame_id').counts.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
        selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:100].frame_id.tolist()


        # #filter by label entropy
        # selected_frames=[]
        # _df = pred_df.groupby(['frame_id','labels']).model_score.mean()
        # index = pd.MultiIndex.from_product([_df.index.levels[0].unique(),_df.index.levels[1].unique()])
        # confidence_process_df=_df.reindex(index, fill_value=0).reset_index()
        # select_confidence = np.array([0,0,0])

        # for i in range(total_select_nums):
        #     if i % 2 == 1:
        #         select_confidence = confidence_process_df[confidence_process_df.frame_id.isin(selected_frames)].groupby('labels').model_score.mean().values
        #         entropy_df = pd.DataFrame()
        #         _df = confidence_process_df[~confidence_process_df.frame_id.isin(selected_frames)]
        #         entropy_df['frame_id'] = _df.frame_id.unique()
        #         entropy_df['entropy'] = entropy(_df.groupby(['frame_id']).model_score.apply(list).tolist()+select_confidence,base=3,axis=1)
        #         selected_frames.append(entropy_df.sort_values('entropy',ascending=False).frame_id.iloc[0])
        #     else:
        #         selected_frames.append(frame_df[~frame_df.frame_id.isin(selected_frames)].sort_values('model_score',ascending=False).iloc[0].frame_id)

        
        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        print('\n\n============== Selected =================\n')
        print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV22Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV22Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        total_it_each_epoch =len(self.unlabelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        # self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        lda = LinearDiscriminantAnalysis(n_components = 2)
        instance_embeddings = np.array(feature_df.embeddings.tolist())
        lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
        # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
        # features = StandardScaler().fit_transform(features)

        feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred
                

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25))
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = class_weigted_df/class_weigted_df.sum()
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            # pred_df['instance_model_score'] = pred_df.groupby('labels')['instance_model_score'].transform(lambda x: np.exp(x.median()+x/(x.quantile(0.75)-x.quantile(0.25))))

            # pred_df['instance_model_score'] = np.exp(_df -_df.median()+pred_df['instance_model_score'].median())+eps
            
            pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
            _df  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
            pred_df['spatial_model_score'] = np.exp(np.clip(_df - _df.median()+pred_df['spatial_model_score'].median(),None,500))+eps
            pred_df['model_score']          = pred_df['instance_model_score']#*pred_df['global_instance_feature_ent']#* pred_df['class_weighted']# * pred_df['spatial_model_score']
            # pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].mean()/pred_df['model_score'].mean())*pred_df['spatial_model_score']
            # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
            # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            # redundant filtering
            redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


            # filter 1
            label_entropy_df = pred_df.groupby(['frame_id','labels']).size().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            frame_df['model_score'] = frame_df['model_score'] * label_entropy_df.frame_ent
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV23Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV23Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        total_it_each_epoch =len(self.unlabelled_loader)
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        # self.enable_dropout(self.model)
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        lda = LinearDiscriminantAnalysis(n_components = 2)
        instance_embeddings = np.array(feature_df.embeddings.tolist())
        lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
        # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
        # features = StandardScaler().fit_transform(features)

        feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='random',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred
                

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25))
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = class_weigted_df/class_weigted_df.sum()
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            # pred_df['instance_model_score'] = pred_df.groupby('labels')['instance_model_score'].transform(lambda x: np.exp(x.median()+x/(x.quantile(0.75)-x.quantile(0.25))))

            # pred_df['instance_model_score'] = np.exp(_df -_df.median()+pred_df['instance_model_score'].median())+eps
            
            pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
            _df  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
            pred_df['spatial_model_score'] = np.exp(np.clip(_df - _df.median()+pred_df['spatial_model_score'].median(),None,500))+eps
            pred_df['model_score']          = QuantileTransformer().fit_transform((pred_df['instance_model_score'] * pred_df['related_confidence_weighted'])[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#
            # pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].mean()/pred_df['model_score'].mean())*pred_df['spatial_model_score']
            # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
            # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            # redundant filtering
            redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]


            # filter 1
            label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            frame_df['model_score'] = frame_df['model_score'] * label_entropy_df.frame_ent
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

# redundant filtering + 1_ent 
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV24Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV24Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['labeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_model_pred'] = pred

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['unlabeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        # spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        # spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        lda = LinearDiscriminantAnalysis(n_components = 2)
        instance_embeddings = np.array(feature_df.embeddings.tolist())
        lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
        # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
        # features = StandardScaler().fit_transform(features)

        feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)
                data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            # pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            # pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            # pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = np.clip(pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25)),None,500)
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = 1+(class_weigted_df/class_weigted_df.sum())
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            # pred_df['instance_model_score'] = pred_df.groupby('labels')['instance_model_score'].transform(lambda x: np.exp(x.median()+x/(x.quantile(0.75)-x.quantile(0.25))))

            # pred_df['instance_model_score'] = np.exp(_df -_df.median()+pred_df['instance_model_score'].median())+eps
            
            # pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
            # _df  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
            # pred_df['spatial_model_score'] = np.exp(np.clip(_df - _df.median()+pred_df['spatial_model_score'].median(),None,500))+eps
            pred_df['model_score']          = QuantileTransformer().fit_transform((pred_df['instance_model_score'] * pred_df['related_confidence_weighted'] * pred_df['class_weighted'])[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#
            # pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].mean()/pred_df['model_score'].mean())*pred_df['spatial_model_score']
            # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
            # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            feature_name = 'combine_features'
            instance_embedding_std_df = pred_df.groupby(['frame_id','labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean()).reset_index(name='embedding_std')
            instance_embedding_std_df['count'] = pred_df.groupby(['frame_id','labels']).size().values
            mean_df = pred_df.groupby(['labels'])[feature_name].mean()
            mean_embedding_std_df = pred_df.groupby(['labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean())
            pred_df['mean_instance_std'] = pred_df.apply(lambda x:np.std([x[feature_name],mean_df[x.labels]]/mean_embedding_std_df[x.labels],axis=0).mean() ,axis=1)
            instance_embedding_std_df.loc[instance_embedding_std_df['count'] == 1,'embedding_std'] = pred_df.groupby(['frame_id','labels']).mean_instance_std.mean().reset_index()[instance_embedding_std_df['count'] == 1].mean_instance_std.values
            instance_embedding_std_df = instance_embedding_std_df.groupby('frame_id').embedding_std.mean()
            print(instance_embedding_std_df.describe())
            instance_embedding_std_df = 1+np.log1p(instance_embedding_std_df)
            print(instance_embedding_std_df.describe())

            # redundant filtering
            redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

            
            # filter 1
            label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            frame_df['embedding_std'] = frame_df.frame_id.map(instance_embedding_std_df)
            frame_df['model_score'] = (frame_df['model_score'] * frame_df['embedding_std']) * (1+label_entropy_df.frame_ent)
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

# 1_redundant score 1_ent no embedding std
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV25Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV25Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['labeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_model_pred'] = pred

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['unlabeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        # spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        # spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        lda = LinearDiscriminantAnalysis(n_components = 2)
        instance_embeddings = np.array(feature_df.embeddings.tolist())
        lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
        # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
        # features = StandardScaler().fit_transform(features)

        feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)
                data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            # pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            # pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            # pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = np.clip(pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25)),None,500)
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = 1+(class_weigted_df/class_weigted_df.sum())
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            # pred_df['instance_model_score'] = pred_df.groupby('labels')['instance_model_score'].transform(lambda x: np.exp(x.median()+x/(x.quantile(0.75)-x.quantile(0.25))))

            # pred_df['instance_model_score'] = np.exp(_df -_df.median()+pred_df['instance_model_score'].median())+eps
            
            # pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
            # _df  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
            # pred_df['spatial_model_score'] = np.exp(np.clip(_df - _df.median()+pred_df['spatial_model_score'].median(),None,500))+eps
            pred_df['model_score']          = QuantileTransformer().fit_transform((pred_df['instance_model_score'] * pred_df['related_confidence_weighted'] * pred_df['class_weighted']).values[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#
            # pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].mean()/pred_df['model_score'].mean())*pred_df['spatial_model_score']
            # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
            # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            feature_name = 'combine_features'
            instance_embedding_std_df = pred_df.groupby(['frame_id','labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean()).reset_index(name='embedding_std')
            instance_embedding_std_df['count'] = pred_df.groupby(['frame_id','labels']).size().values
            mean_df = pred_df.groupby(['labels'])[feature_name].mean()
            mean_embedding_std_df = pred_df.groupby(['labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean())
            pred_df['mean_instance_std'] = pred_df.apply(lambda x:np.std([x[feature_name],mean_df[x.labels]]/mean_embedding_std_df[x.labels],axis=0).mean() ,axis=1)
            instance_embedding_std_df.loc[instance_embedding_std_df['count'] == 1,'embedding_std'] = pred_df.groupby(['frame_id','labels']).mean_instance_std.mean().reset_index()[instance_embedding_std_df['count'] == 1].mean_instance_std.values
            instance_embedding_std_df = instance_embedding_std_df.groupby('frame_id').embedding_std.mean()
            print(instance_embedding_std_df.describe())
            instance_embedding_std_df = 1+np.log1p(instance_embedding_std_df)
            print(instance_embedding_std_df.describe())

            # redundant filtering
            # redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            # redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

            
            # filter 1
            label_confidence_frame_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            redundant_frame_df = pred_df.groupby(['frame_id']).model_redundant.sum()

            frame_df['embedding_std'] = frame_df.frame_id.map(instance_embedding_std_df)
            frame_df['redundant_score'] = frame_df.frame_id.map(redundant_frame_df)
            frame_df['redundant_score'] = (2-QuantileTransformer().fit_transform(frame_df['redundant_score'].values[:,None])[:,0])
            frame_df['model_score'] = frame_df['model_score'] * (1+label_confidence_frame_df.frame_ent) * frame_df['redundant_score'] #  * frame_df['embedding_std']
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

# 24 no embedding std : lower
class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV26Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV26Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['labeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_model_pred'] = pred

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['unlabeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        # spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        # spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        lda = LinearDiscriminantAnalysis(n_components = 2)
        instance_embeddings = np.array(feature_df.embeddings.tolist())
        lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
        # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
        # features = StandardScaler().fit_transform(features)

        feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)
                data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            # pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            # pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            # pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = np.clip(pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25)),None,500)
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = 1+(class_weigted_df/class_weigted_df.sum())
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            # pred_df['instance_model_score'] = pred_df.groupby('labels')['instance_model_score'].transform(lambda x: np.exp(x.median()+x/(x.quantile(0.75)-x.quantile(0.25))))

            # pred_df['instance_model_score'] = np.exp(_df -_df.median()+pred_df['instance_model_score'].median())+eps
            
            # pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
            # _df  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
            # pred_df['spatial_model_score'] = np.exp(np.clip(_df - _df.median()+pred_df['spatial_model_score'].median(),None,500))+eps
            pred_df['model_score']          = QuantileTransformer().fit_transform((pred_df['instance_model_score'] * pred_df['related_confidence_weighted'] * pred_df['class_weighted']).values[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#
            # pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].mean()/pred_df['model_score'].mean())*pred_df['spatial_model_score']
            # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
            # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            feature_name = 'combine_features'
            instance_embedding_std_df = pred_df.groupby(['frame_id','labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean()).reset_index(name='embedding_std')
            instance_embedding_std_df['count'] = pred_df.groupby(['frame_id','labels']).size().values
            mean_df = pred_df.groupby(['labels'])[feature_name].mean()
            mean_embedding_std_df = pred_df.groupby(['labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean())
            pred_df['mean_instance_std'] = pred_df.apply(lambda x:np.std([x[feature_name],mean_df[x.labels]]/mean_embedding_std_df[x.labels],axis=0).mean() ,axis=1)
            instance_embedding_std_df.loc[instance_embedding_std_df['count'] == 1,'embedding_std'] = pred_df.groupby(['frame_id','labels']).mean_instance_std.mean().reset_index()[instance_embedding_std_df['count'] == 1].mean_instance_std.values
            instance_embedding_std_df = instance_embedding_std_df.groupby('frame_id').embedding_std.mean()
            print(instance_embedding_std_df.describe())
            instance_embedding_std_df = 1+np.log1p(instance_embedding_std_df)
            print(instance_embedding_std_df.describe())

            # redundant filtering
            redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

            
            # filter 1
            label_confidence_frame_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            redundant_frame_df = pred_df.groupby(['frame_id']).model_redundant.sum()

            frame_df['embedding_std'] = frame_df.frame_id.map(instance_embedding_std_df)
            frame_df['redundant_score'] = frame_df.frame_id.map(redundant_frame_df)
            frame_df['redundant_score'] = (2-QuantileTransformer().fit_transform(frame_df['redundant_score'].values[:,None])[:,0])
            frame_df['model_score'] = frame_df['model_score'] * (1+label_confidence_frame_df.frame_ent) #* frame_df['redundant_score'] #  * frame_df['embedding_std']
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV27Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV27Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['labeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_model_pred'] = pred

        # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
        # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
        # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
        # model.fit(spatial_dict['unlabeled_spatial_features'])
        # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        # spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
        # spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        # spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        lda = LinearDiscriminantAnalysis(n_components = 2)
        instance_embeddings = np.array(feature_df.embeddings.tolist())
        lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
        # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
        # features = StandardScaler().fit_transform(features)

        feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)
                data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            # pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            # pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            # pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = np.clip(pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25)),None,500)
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = 1+(class_weigted_df/class_weigted_df.sum())
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

            feature_name = 'combine_features'
            instance_embedding_std_df = pred_df.groupby(['frame_id','labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean()).reset_index(name='embedding_std')
            instance_embedding_std_df['count'] = pred_df.groupby(['frame_id','labels']).size().values
            mean_df = pred_df.groupby(['labels'])[feature_name].mean()
            mean_embedding_std_df = pred_df.groupby(['labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean())
            pred_df['mean_instance_std'] = pred_df.apply(lambda x:np.std([x[feature_name],mean_df[x.labels]]/mean_embedding_std_df[x.labels],axis=0).mean() ,axis=1)
            instance_embedding_std_df.loc[instance_embedding_std_df['count'] == 1,'embedding_std'] = pred_df.groupby(['frame_id','labels']).mean_instance_std.mean().reset_index()[instance_embedding_std_df['count'] == 1].mean_instance_std.values
            instance_embedding_std_df = instance_embedding_std_df.groupby(['frame_id','labels']).embedding_std.mean()
            print(instance_embedding_std_df.describe())
            instance_embedding_std_df = 1+np.log1p(instance_embedding_std_df)
            print(instance_embedding_std_df.describe())

            pred_df['instance_embedding_std'] = pred_df.apply(lambda x:instance_embedding_std_df.loc[x.frame_id, x.labels] ,axis=1)
            pred_df['model_score']          = QuantileTransformer().fit_transform((pred_df['instance_model_score'] * pred_df['related_confidence_weighted'] * pred_df['class_weighted']*pred_df['instance_embedding_std'])[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            

            # redundant filtering
            redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

            
            # filter 1
            label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            # frame_df['embedding_std'] = frame_df.frame_id.map(instance_embedding_std_df)
            frame_df['model_score'] = (frame_df['model_score']* (1+label_entropy_df.frame_ent))
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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

class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV28Smpling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
        super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV28Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
        # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
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
                    
                    selected = pred_dicts[batch_inx]['selected']
                    pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
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
                        record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
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
                    
                    spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
                    spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
                    spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
                    # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
                    # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
                    # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
                    # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
                    record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
                    # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
                    # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
                    record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

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
                        record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
                        record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()
            
        record_dict = dict(record_dict)
        
        print('** [Instance] start searching...**')
        
        ## ============= process data ============= 
        def process_df(df):
            spatial_dict={}
            delete_cols=[]
            for k,v in df.items():
                
                if 'spatial' in str(k):
                    if 'frame' not in str(k):
                        spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
                    else:
                        spatial_dict[k]= v
                    delete_cols.append(k)
                elif 'frame' not in str(k):
                    df[k] = [i.cpu().numpy() for i in v]
            
            for col in delete_cols:
                del df[col]
            return df, spatial_dict
       
        df, spatial_dict = process_df(record_dict)
        
        unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
        labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
        # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
        # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
        unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
        labeled_df = pd.DataFrame.from_dict(labeled_dict)

        unlabeled_df['Set'] = 'unlabeled'
        labeled_df['Set'] = 'labeled'
        unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
        labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
        combine_df = pd.concat([unlabeled_df,labeled_df])

        for col in combine_df.columns:
            if 'labels' in str(col) :
                combine_df[col] = combine_df[col].astype(int)
            elif str(col) not in ['embeddings','logits','Set','frame_id']:
                combine_df[col] = combine_df[col].astype(float)
            
        combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
        ## Select each class
        feature_df = combine_df.copy()
        selected_frames=[]
        saving_dict = dict()
        saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
        bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
        feature_columns=[]
        feature_columns.extend(bbox_feature_cols)
        
        # ## spatial feature processing
        combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
        combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
        spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
        spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

        # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
        # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

        # embeddings = np.array(combine_spatial_features)
        # pca = PCA()
        # pca_feature = pca.fit_transform(embeddings)
        # exp_var_pca = pca.explained_variance_ratio_
        # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
        # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
        # print('num dum: {}'.format(num_dim))

        components = 10
        
        # #compute frame level score


        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['unlabeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_unlabeled_pred'] = pred

        model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
        model.fit(spatial_dict['labeled_spatial_features'])
        pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
        spatial_dict['spatial_feature_model_pred'] = pred
        spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
        
        spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
        
        ## Select each class

        unlabeled_models = []
        tsne_features = []
        selected_frames = []

        # lda = LinearDiscriminantAnalysis(n_components = 2)
        # instance_embeddings = np.array(feature_df.embeddings.tolist())
        # lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        # feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

        for c_idx, select_num in enumerate(cls_select_nums): 
            gmm_running = time.time()
            cidx = c_idx+1
            data_df = feature_df[feature_df.labels == cidx]

            ## ============= prepare instance data ============= 
            embeddings = np.array(data_df.embeddings.tolist())
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100)
            tsne_feature = tsne.fit_transform(embeddings)
            tsne_features.append(tsne_feature)
            features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
            features = StandardScaler().fit_transform(features)

            model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
            model.fit(features)
            unlabeled_models.append(model)

        for i in range(10):

            pred_list = []
            for c_idx, select_num in enumerate(cls_select_nums):  
                gmm_running = time.time()
                cidx = c_idx+1
                data_df = feature_df[(feature_df.labels == cidx)]

                ## ============= prepare instance data ============= 
                embeddings = np.array(data_df.embeddings.tolist())
                # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
                # tsne_feature = tsne.fit_transform(embeddings)
                tsne_feature = np.array(tsne_features[c_idx])
                features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
                features = StandardScaler().fit_transform(features)
                data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values

                ## ============= prepare training data ============= 
                training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
                print('number of labeled frames: ',training_criteria.sum())
                train_data = features[training_criteria]#.sample(frac=1)
                validation_data = features[~training_criteria]
                data_df=data_df[~training_criteria]
                
                ## ============= instance feature training data =============
                model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
                # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
                # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
                model.fit(train_data)
                pred = model.score_samples(validation_data)
                data_df['instance_feature_model_pred'] = pred

                pred = unlabeled_models[c_idx].score_samples(validation_data)
                data_df['instance_feature_unlabeled_pred'] = pred

                pred_list.append(data_df)
                print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
            select_start_time = time.time()
            pred_df = pd.concat(pred_list)

            pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
            pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

            pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
            pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)
            pred_df['spatial_distribution_gap'] = (pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred'])
            pred_df['spatial_model_score'] = np.clip(pred_df['spatial_distribution_gap']/(pred_df['spatial_distribution_gap'].quantile(.75)-pred_df['spatial_distribution_gap'].quantile(.25)),None,500)
            pred_df['spatial_model_score'] = np.exp(pred_df['spatial_model_score'])#+eps
            pred_df['spatial_model_score']          = QuantileTransformer().fit_transform(pred_df.spatial_model_score.values[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#


            pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
            # pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
            pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

            alpha = 1.2
            eps = 1e-20
            pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
            pred_df['instance_model_score'] = np.clip(pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25)),None,500)
            pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

            class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
            class_weigted_df = 1+(class_weigted_df/class_weigted_df.sum())
            pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)


            # pred_df['instance_embedding_std'] = pred_df.apply(lambda x:instance_embedding_std_df.loc[x.frame_id, x.labels] ,axis=1)
            pred_df['model_score']          = QuantileTransformer().fit_transform((pred_df['instance_model_score'] * pred_df['related_confidence_weighted'] * pred_df['class_weighted'])[:,None])[:,0] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#

            print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
            print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
            # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
            # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
            print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
            print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
            # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
            print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
            # print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
            print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

            saving_dict['feature_df'] = pred_df
            saving_dict['spatial_dict'] = spatial_dict
            saving_dict['eps']=eps

            feature_name = 'combine_features'
            instance_embedding_std_df = pred_df.groupby(['frame_id','labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean()).reset_index(name='embedding_std')
            instance_embedding_std_df['count'] = pred_df.groupby(['frame_id','labels']).size().values
            mean_df = pred_df.groupby(['labels'])[feature_name].mean()
            mean_embedding_std_df = pred_df.groupby(['labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean())
            pred_df['mean_instance_std'] = pred_df.apply(lambda x:np.std([x[feature_name],mean_df[x.labels]]/mean_embedding_std_df[x.labels],axis=0).mean() ,axis=1)
            instance_embedding_std_df.loc[instance_embedding_std_df['count'] == 1,'embedding_std'] = pred_df.groupby(['frame_id','labels']).mean_instance_std.mean().reset_index()[instance_embedding_std_df['count'] == 1].mean_instance_std.values
            instance_embedding_std_df = instance_embedding_std_df.groupby(['frame_id']).embedding_std.mean()
            print(instance_embedding_std_df.describe())
            instance_embedding_std_df = 1+np.log1p(instance_embedding_std_df)
            print(instance_embedding_std_df.describe())

            

            # redundant filtering
            redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
            redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
            pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

            
            # filter 1
            label_entropy_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
            frame_df = pred_df.groupby(['frame_id'])[['model_score','spatial_model_score']].mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
            frame_df['embedding_std'] = frame_df.frame_id.map(instance_embedding_std_df)
            frame_df['model_score'] = (frame_df['model_score']+frame_df['spatial_model_score'])* (1+label_entropy_df.frame_ent)
            _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

            # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
            # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
            # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
            # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
            selected_frames.extend(_selected_frames)

        pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
        frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
        saving_dict['selected_frames']=selected_frames
        # print('\n\n============== Selected =================\n')
        # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
        # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
        # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
        # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
        # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
        # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
        # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
        # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
        # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


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
 
# class InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV27Smpling(Strategy):
#     def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
#         super(InstanceClusterWithFrameaGMMConciseRedundantDensityWeightedDynamicpowGlobalV27Smpling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

#     def pairwise_squared_distances(self, x, y):
#         # x: N * D * D * 1
#         # y: M * D * D * 1
#         # return: N * M    
#         assert (len(x.shape)) > 1
#         assert (len(y.shape)) > 1  
#         n = x.shape[0]
#         m = y.shape[0]
#         x = x.view(n, -1)
#         y = y.view(m, -1)
                                                              
#         x_norm = (x**2).sum(1).view(n, 1)
#         y_t = y.permute(1, 0).contiguous()
#         y_norm = (y**2).sum(1).view(1, m)
#         dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
#         dist[dist != dist] = 0 # replace nan values with 0
#         return torch.clamp(dist, 0.0, float('inf'))

#     def select_func(self):
#         pass
    
#     def enable_dropout(self, model, enable=True):
#         """ Function to enable the dropout layers during test-time """
#         i = 0
#         for m in model.modules():
#             if m.__class__.__name__.startswith('Dropout'):
#                 i += 1
#                 m.train()
#         print('**found and enabled {} Dropout layers for random sampling**'.format(i))

#     def query(self, leave_pbar=True, cur_epoch=None):
#         def divNum(num, parts):
#             p = int(num/parts)
#             missing = int(num - p*parts)             
#             return [p+1]*missing + [p]*(parts-missing)
        
#         class_names = self.cfg.CLASS_NAMES
#         total_select_nums = self.cfg.ACTIVE_TRAIN.SELECT_NUMS
#         cls_select_nums = divNum(total_select_nums,len(class_names))
        
#         self.model.eval()
#         record_dict=defaultdict(list)
        

#         # feed unlabeled data forward the model
#         total_it_each_epoch = len(self.unlabelled_loader)
#         if self.rank == 0:
#             pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
#                              desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
#         self.model.eval()
#         # self.enable_dropout(self.model)
#         # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
#         val_dataloader_iter = iter(self.unlabelled_loader)
#         val_loader = self.unlabelled_loader
#         for cur_it in range(total_it_each_epoch):
#             try:
#                 unlabelled_batch = next(val_dataloader_iter)
#             except StopIteration:
#                 unlabelled_dataloader_iter = iter(val_loader)
#                 unlabelled_batch = next(unlabelled_dataloader_iter)
#             with torch.no_grad():
#                 load_data_to_gpu(unlabelled_batch)
#                 pred_dicts, _ = self.model(unlabelled_batch)
#                 for batch_inx in range(len(pred_dicts)):
                    
#                     selected = pred_dicts[batch_inx]['selected']
#                     pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
#                     self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
#                     spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
#                     spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
#                     spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
#                     # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
#                     # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
#                     # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
#                     # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
#                     record_dict['unlabeled_spatial_features'].append(spatial_features_2d.view(-1))
#                     # record_dict['unlabeled_height_spatial_features'].append(height_spatial_features)
#                     # record_dict['unlabeled_width_spatial_features'].append(width_spatial_features)
#                     record_dict['unlabeled_spatial_frame_ids'].append(unlabelled_batch['frame_id'][batch_inx])
                    
#                     for instance_idx in range(len(pred_instance_labels)):
#                         record_dict['unlabeled_frame_id'].append(unlabelled_batch['frame_id'][batch_inx])
#                         record_dict['unlabeled_labels'].append(pred_instance_labels[instance_idx])
#                         record_dict['unlabeled_rotation'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-1])
#                         record_dict['unlabeled_height_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-4])
#                         record_dict['unlabeled_width_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-3])
#                         record_dict['unlabeled_length_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-2])
#                         record_dict['unlabeled_pts'].append(pred_dicts[batch_inx]['pts_count'][instance_idx])
#                         record_dict['unlabeled_box_volumes'].append(pred_dicts[batch_inx]['box_volumes'][instance_idx])
#                         record_dict['unlabeled_embeddings'].append(pred_dicts[batch_inx]['pred_embeddings'][instance_idx].view(-1))
#                         record_dict['unlabeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
#                         record_dict['unlabeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
#                         record_dict['unlabeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
#                         record_dict['unlabeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
                    
#             if self.rank == 0:
#                 pbar.update()
#                 # pbar.set_postfix(disp_dict)
#                 pbar.refresh()
#         if self.rank == 0:
#             pbar.close()
            
            
#         ## Label
#         # feed labeled data forward the model
#         total_it_each_epoch = len(self.labelled_loader)
#         if self.rank == 0:
#             pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
#                              desc='evaluating_labelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
#         # select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS
#         val_dataloader_iter = iter(self.labelled_loader)
#         val_loader = self.labelled_loader
#         for cur_it in range(total_it_each_epoch):
#             try:
#                 labelled_batch = next(val_dataloader_iter)
#             except StopIteration:
#                 labelled_dataloader_iter = iter(val_loader)
#                 labelled_batch = next(labelled_dataloader_iter)
#             with torch.no_grad():
#                 load_data_to_gpu(labelled_batch)
#                 pred_dicts, _ = self.model(labelled_batch)
                
#                 for batch_inx in range(len(pred_dicts)):
#                     pred_instance_labels = pred_dicts[batch_inx]['pred_labels']
                    
#                     spatial_features_2d = pred_dicts[batch_inx]['spatial_features_2d']
#                     spatial_features_2d = torch.mean(spatial_features_2d, dim=0)[None,:]
#                     spatial_features_2d = adaptive_avg_pool2d(spatial_features_2d,10)
#                     # width_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=0)
#                     # height_spatial_features = torch.mean(torch.mean(spatial_features_2d, dim=0),dim=1)
#                     # height_spatial_features = adaptive_avg_pool1d(torch.tensor(height_spatial_features).view(1,-1),10).view(-1)
#                     # width_spatial_features = adaptive_avg_pool1d(torch.tensor(width_spatial_features).view(1,-1),10).view(-1)
#                     record_dict['labeled_spatial_features'].append(spatial_features_2d.view(-1))
#                     # record_dict['labeled_height_spatial_features'].append(height_spatial_features)
#                     # record_dict['labeled_width_spatial_features'].append(width_spatial_features)
#                     record_dict['labeled_spatial_frame_ids'].append(labelled_batch['frame_id'][batch_inx])

#                     for instance_idx in range(len(pred_instance_labels)):
#                         record_dict['labeled_frame_id'].append(labelled_batch['frame_id'][batch_inx])
#                         record_dict['labeled_labels'].append(pred_instance_labels[instance_idx])
#                         record_dict['labeled_rotation'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-1])
#                         record_dict['labeled_height_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-4])
#                         record_dict['labeled_width_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-3])
#                         record_dict['labeled_length_3d'].append(pred_dicts[batch_inx]['pred_boxes'][instance_idx][-2])
#                         record_dict['labeled_pts'].append(pred_dicts[batch_inx]['pts_count'][instance_idx])
#                         record_dict['labeled_box_volumes'].append(pred_dicts[batch_inx]['box_volumes'][instance_idx])
#                         record_dict['labeled_embeddings'].append(pred_dicts[batch_inx]['pred_embeddings'][instance_idx].view(-1))
#                         record_dict['labeled_pts_density'].append(pred_dicts[batch_inx]['pred_box_unique_density'][instance_idx])
#                         record_dict['labeled_confidence'].append(pred_dicts[batch_inx]['pred_scores'][instance_idx])
#                         record_dict['labeled_logits'].append(pred_dicts[batch_inx]['pred_logits'][instance_idx])
#                         record_dict['labeled_cls_entropy'].append(Categorical(probs = torch.sigmoid(pred_dicts[batch_inx]['pred_logits'][instance_idx])).entropy())
                    
#             if self.rank == 0:
#                 pbar.update()
#                 # pbar.set_postfix(disp_dict)
#                 pbar.refresh()
#         if self.rank == 0:
#             pbar.close()
            
#         record_dict = dict(record_dict)
        
#         print('** [Instance] start searching...**')
        
#         ## ============= process data ============= 
#         def process_df(df):
#             spatial_dict={}
#             delete_cols=[]
#             for k,v in df.items():
                
#                 if 'spatial' in str(k):
#                     if 'frame' not in str(k):
#                         spatial_dict[k]= np.array([ i.cpu().numpy() for i in v ])
#                     else:
#                         spatial_dict[k]= v
#                     delete_cols.append(k)
#                 elif 'frame' not in str(k):
#                     df[k] = [i.cpu().numpy() for i in v]
            
#             for col in delete_cols:
#                 del df[col]
#             return df, spatial_dict
       
#         df, spatial_dict = process_df(record_dict)
        
#         unlabeled_dict = {k:v for k,v in df.items() if 'unlabeled' in str(k)}
#         labeled_dict = {k:v for k,v in df.items() if 'unlabeled' not in str(k)}
        
#         # spatial_dict['labeled_spatial_feature'] = np.concatenate([spatial_dict['labeled_height_spatial_features'],spatial_dict['labeled_width_spatial_features']],axis=1)
#         # spatial_dict['unlabeled_spatial_feature'] = np.concatenate([spatial_dict['unlabeled_height_spatial_features'],spatial_dict['unlabeled_width_spatial_features']],axis=1)
        
#         unlabeled_df = pd.DataFrame.from_dict(unlabeled_dict)
#         labeled_df = pd.DataFrame.from_dict(labeled_dict)

#         unlabeled_df['Set'] = 'unlabeled'
#         labeled_df['Set'] = 'labeled'
#         unlabeled_df = unlabeled_df.rename(columns={ col:col.replace('unlabeled_','') for col in unlabeled_df.columns})
#         labeled_df = labeled_df.rename(columns={ col:col.replace('labeled_','') for col in labeled_df.columns})
#         combine_df = pd.concat([unlabeled_df,labeled_df])

#         for col in combine_df.columns:
#             if 'labels' in str(col) :
#                 combine_df[col] = combine_df[col].astype(int)
#             elif str(col) not in ['embeddings','logits','Set','frame_id']:
#                 combine_df[col] = combine_df[col].astype(float)
            
#         combine_df['labeled'] = combine_df.Set.apply(lambda x : 1 if x == 'labeled' else 0)
        
#         ## Select each class
#         feature_df = combine_df.copy()
#         selected_frames=[]
#         saving_dict = dict()
#         saving_dict['labeled_df'] = feature_df[feature_df.labeled == 1]
#         bbox_feature_cols=['rotation', 'height_3d', 'width_3d', 'length_3d','pts', 'box_volumes', 'pts_density','confidence','cls_entropy']
#         feature_columns=[]
#         feature_columns.extend(bbox_feature_cols)
        
#         # ## spatial feature processing
#         combine_spatial_features = np.concatenate([spatial_dict['labeled_spatial_features'],spatial_dict['unlabeled_spatial_features']])
#         combine_spatial_features = StandardScaler().fit_transform(combine_spatial_features)
#         spatial_dict['labeled_spatial_features'] = combine_spatial_features[:len(spatial_dict['labeled_spatial_features']),:]
#         spatial_dict['unlabeled_spatial_features'] = combine_spatial_features[len(spatial_dict['labeled_spatial_features']):,:]

#         # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
#         # combine_spatial_features = tsne.fit_transform(combine_spatial_features)

#         # embeddings = np.array(combine_spatial_features)
#         # pca = PCA()
#         # pca_feature = pca.fit_transform(embeddings)
#         # exp_var_pca = pca.explained_variance_ratio_
#         # num_dim = sum(np.cumsum(exp_var_pca)<0.8)
#         # combine_spatial_features = pca_feature[:,:len(bbox_feature_cols)]
#         # print('num dum: {}'.format(num_dim))

#         components = 10
        
#         # #compute frame level score

#         # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
#         # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
#         # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
#         # model.fit(spatial_dict['labeled_spatial_features'])
#         # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
#         # spatial_dict['spatial_feature_model_pred'] = pred

#         # model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-1)
#         # # model=mixture.GaussianMixture(n_components=spatial_dict['labeled_spatial_features'].shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-1)
#         # # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
#         # model.fit(spatial_dict['unlabeled_spatial_features'])
#         # pred = model.score_samples(spatial_dict['unlabeled_spatial_features'])
#         # spatial_dict['spatial_feature_unlabeled_pred'] = pred
   
#         # spatial_unlabeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_unlabeled_pred']))
#         # spatial_labeled_pred_mapping_dict = dict(zip(spatial_dict['unlabeled_spatial_frame_ids'], spatial_dict['spatial_feature_model_pred']))
#         ## Select each class

#         unlabeled_models = []
#         tsne_features = []
#         selected_frames = []

#         lda = LinearDiscriminantAnalysis(n_components = 2)
#         instance_embeddings = np.array(feature_df.embeddings.tolist())
#         lda_feature = lda.fit_transform(instance_embeddings, feature_df.labels-1)
        
#         # features = np.concatenate([feature_df[bbox_feature_cols],tsne_feature], axis=1)
#         # features = StandardScaler().fit_transform(features)

#         feature_df['global_instance_feature_ent'] =  1+entropy(lda.predict_proba(instance_embeddings),base=3,axis=1)

        

#         for c_idx, select_num in enumerate(cls_select_nums): 
#             gmm_running = time.time()
#             cidx = c_idx+1
#             data_df = feature_df[feature_df.labels == cidx]

#             ## ============= prepare instance data ============= 
#             embeddings = np.array(data_df.embeddings.tolist())
#             tsne = TSNE(n_components=2, learning_rate='auto', init='pca',perplexity=100)
#             tsne_feature = tsne.fit_transform(embeddings)
#             tsne_features.append(tsne_feature)
#             features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
#             features = StandardScaler().fit_transform(features)

#             model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
#             model.fit(features)
#             unlabeled_models.append(model)

#         for i in range(10):

#             pred_list = []
#             for c_idx, select_num in enumerate(cls_select_nums):  
#                 gmm_running = time.time()
#                 cidx = c_idx+1
#                 data_df = feature_df[(feature_df.labels == cidx)]

#                 ## ============= prepare instance data ============= 
#                 embeddings = np.array(data_df.embeddings.tolist())
#                 # tsne = TSNE(n_components=2, learning_rate='auto', init='random',)
#                 # tsne_feature = tsne.fit_transform(embeddings)
#                 tsne_feature = np.array(tsne_features[c_idx])
#                 features = np.concatenate([data_df[bbox_feature_cols],tsne_feature], axis=1)
#                 features = StandardScaler().fit_transform(features)
#                 data_df['combine_features'] = pd.DataFrame({'combine_features':list(features)}).combine_features.values

#                 ## ============= prepare training data ============= 
#                 training_criteria = ((data_df.labeled == 1)|(data_df.frame_id.isin(selected_frames)))  #| (data_df.cluster_center == 1)
#                 print('number of labeled frames: ',training_criteria.sum())
#                 train_data = features[training_criteria]#.sample(frac=1)
#                 validation_data = features[~training_criteria]
#                 data_df=data_df[~training_criteria]
                
#                 ## ============= instance feature training data =============
#                 model=mixture.BayesianGaussianMixture(n_components=components, init_params='k-means++',max_iter=1000,random_state=3131,reg_covar=1e-2)
#                 # model=mixture.GaussianMixture(n_components=features.shape[1], init_params='k-means++',random_state=3131,reg_covar=1e-3)
#                 # model=KernelDensity(bandwidth="silverman",kernel='exponential',breadth_first=True)
#                 model.fit(train_data)
#                 pred = model.score_samples(validation_data)
#                 data_df['instance_feature_model_pred'] = pred

#                 pred = unlabeled_models[c_idx].score_samples(validation_data)
#                 data_df['instance_feature_unlabeled_pred'] = pred

#                 pred_list.append(data_df)
#                 print(f"--- GMM running time: %s seconds ---" % (time.time() - gmm_running))
                
#             select_start_time = time.time()
#             pred_df = pd.concat(pred_list)

#             pred_df['related_confidence_weighted'] = np.clip(pred_df.confidence/pred_df.groupby('labels').confidence.transform(lambda x: x.mean()),0,1)
#             pred_df['class_confidence_weighted'] = pred_df['labels'].map(1-(pred_df.groupby('labels').confidence.mean()))
    

#             # pred_df['spatial_feature_unlabeled_pred'] = pred_df.frame_id.map(spatial_unlabeled_pred_mapping_dict)
#             # pred_df['spatial_feature_model_pred'] = pred_df.frame_id.map(spatial_labeled_pred_mapping_dict)

#             pred_df['instance_model_redundant'] = np.exp(pred_df['instance_feature_model_pred'])
#             # pred_df['spatial_model_redundant'] = np.exp(pred_df['spatial_feature_model_pred'])
#             pred_df['model_redundant'] = pred_df['instance_model_redundant']#+pred_df['bbox_model_redundant']

#             alpha = 1.2
#             eps = 1e-20
#             pred_df['instance_distribution_gap'] = (pred_df['instance_feature_unlabeled_pred']-pred_df['instance_feature_model_pred'])
#             pred_df['instance_model_score'] = np.clip(pred_df['instance_distribution_gap']/(pred_df['instance_distribution_gap'].quantile(.75)-pred_df['instance_distribution_gap'].quantile(.25)),None,500)
#             pred_df['instance_model_score'] = np.exp(pred_df['instance_model_score'])#+eps

#             class_weigted_df = np.exp(pred_df.groupby('labels').instance_distribution_gap.mean())
#             class_weigted_df = 1+(class_weigted_df/class_weigted_df.sum())
#             pred_df['class_weighted'] = pred_df.labels.map(class_weigted_df)

#             # pred_df['instance_model_score'] = pred_df.groupby('labels')['instance_model_score'].transform(lambda x: np.exp(x.median()+x/(x.quantile(0.75)-x.quantile(0.25))))

#             # pred_df['instance_model_score'] = np.exp(_df -_df.median()+pred_df['instance_model_score'].median())+eps
            
#             # pred_df['spatial_model_score']  = pred_df['spatial_feature_unlabeled_pred']-pred_df['spatial_feature_model_pred']
#             # _df  = pred_df['spatial_model_score']/(pred_df['spatial_model_score'].quantile(.75)-pred_df['spatial_model_score'].quantile(.25))
#             # pred_df['spatial_model_score'] = np.exp(np.clip(_df - _df.median()+pred_df['spatial_model_score'].median(),None,500))+eps
#             pred_df['model_score']          = QuantileTransformer().fit_transform(pred_df['instance_model_score'].values[:,None])[:,0] * pred_df['related_confidence_weighted'] * pred_df['class_weighted'] #pred_df['class_weighted']# * pred_df['spatial_model_score']#*pred_df['global_instance_feature_ent']#
#             # pred_df['model_score']          = pred_df['model_score']*(pred_df['instance_model_score'].mean()/pred_df['model_score'].mean())*pred_df['spatial_model_score']
#             # pred_df['model_score']          = pred_df['model_score']/(pred_df['model_score'].quantile(.75)-pred_df['model_score'].quantile(.25))
#             # pred_df['model_score']          = pred_df['model_score'] + alpha*pred_df['spatial_model_score']

#             print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
#             print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
#             # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
#             # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
#             print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
#             print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
#             # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
#             print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
#             print('\nGlobal Instance Ent\n',pred_df.groupby('labels')['global_instance_feature_ent'].mean())
#             print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())

#             saving_dict['feature_df'] = pred_df
#             saving_dict['spatial_dict'] = spatial_dict
#             saving_dict['eps']=eps

#             feature_name = 'combine_features'
#             instance_embedding_std_df = pred_df.groupby(['frame_id','labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean()).reset_index(name='embedding_std')
#             instance_embedding_std_df['count'] = pred_df.groupby(['frame_id','labels']).size().values
#             mean_df = pred_df.groupby(['labels'])[feature_name].mean()
#             mean_embedding_std_df = pred_df.groupby(['labels'])[feature_name].apply(lambda x: np.std(x.tolist(),axis=0).mean())
#             pred_df['mean_instance_std'] = pred_df.apply(lambda x:np.std([x[feature_name],mean_df[x.labels]]/mean_embedding_std_df[x.labels],axis=0).mean() ,axis=1)
#             instance_embedding_std_df.loc[instance_embedding_std_df['count'] == 1,'embedding_std'] = pred_df.groupby(['frame_id','labels']).mean_instance_std.mean().reset_index()[instance_embedding_std_df['count'] == 1].mean_instance_std.values
#             instance_embedding_std_df = instance_embedding_std_df.groupby('frame_id').embedding_std.mean()
#             print(instance_embedding_std_df.describe())
#             instance_embedding_std_df = 1+np.log1p(instance_embedding_std_df)
#             print(instance_embedding_std_df.describe())

#             # redundant filtering
#             # redundant_frame_id = pred_df.groupby(['frame_id']).model_redundant.sum().reset_index(drop=False)
#             # redundant_frame_id = redundant_frame_id[redundant_frame_id.model_redundant < redundant_frame_id.model_redundant.quantile(3/10)].frame_id
#             # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]

            
#             # filter 1
#             label_confidence_frame_df = pred_df.groupby(['frame_id','labels']).confidence.sum().reset_index(name='frame_ent').groupby('frame_id').frame_ent.apply(lambda x: entropy(x,base=3)).reset_index(drop=False)
#             frame_df = pred_df.groupby(['frame_id']).model_score.mean().reset_index()#.groupby(['frame_id']).model_score.mean().loc[label_entropy_df.frame_id].reset_index()
#             redundant_frame_df = pred_df.groupby(['frame_id']).model_redundant.sum()

#             frame_df['embedding_std'] = frame_df.frame_id.map(instance_embedding_std_df)
#             frame_df['redundant_score'] = frame_df.frame_id.map(redundant_frame_df)
#             frame_df['redundant_score'] = (2-QuantileTransformer().fit_transform(frame_df['redundant_score'].values[:,None])[:,0])
#             frame_df['model_score'] = frame_df['model_score'] * (1+label_confidence_frame_df.frame_ent) * frame_df['redundant_score'] * frame_df['embedding_std']
#             _selected_frames = frame_df.sort_values(by='model_score',ascending=False).iloc[:10].frame_id.tolist()

#             # redundant_frame_id = frame_df[frame_df.model_score > frame_df.model_score.quantile(5/6)].frame_id
#             # pred_df = pred_df[pred_df.frame_id.isin(redundant_frame_id)]
#             # frame_df = frame_df[frame_df.frame_id.isin(redundant_frame_id)]

            
#             # _selected_frames = label_entropy_df.sort_values(by='counts',ascending=False).iloc[:10].frame_id.tolist()
#             selected_frames.extend(_selected_frames)

#         pred_df = pred_df[pred_df.frame_id.isin(selected_frames)]
#         frame_df = frame_df[frame_df.frame_id.isin(selected_frames)]
#         saving_dict['selected_frames']=selected_frames
#         # print('\n\n============== Selected =================\n')
#         # print('\nInstance Model Pred\n',pred_df.groupby('labels')['instance_feature_model_pred'].min(), pred_df.groupby('labels')['instance_feature_model_pred'].mean(), pred_df.groupby('labels')['instance_feature_model_pred'].max())
#         # print('\nInstance Unlabeled Pred\n',pred_df.groupby('labels')['instance_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['instance_feature_unlabeled_pred'].max())
#         # print('\nSpatial Model Pred\n',pred_df.groupby('labels')['spatial_feature_model_pred'].min(), pred_df.groupby('labels')['spatial_feature_model_pred'].mean(), pred_df.groupby('labels')['spatial_feature_model_pred'].max())
#         # print('\nSpatial Unlabeled Pred\n',pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].min(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].mean(), pred_df.groupby('labels')['spatial_feature_unlabeled_pred'].max())
#         # print('\nModel Redundant\n',pred_df.groupby('labels')['model_redundant'].min(), pred_df.groupby('labels')['model_redundant'].mean(), pred_df.groupby('labels')['model_redundant'].max())
#         # print('\nInstance Score\n',pred_df.groupby('labels')['instance_model_score'].min(), pred_df.groupby('labels')['instance_model_score'].median(),pred_df.groupby('labels')['instance_model_score'].max())
#         # print('\nSpatial Score\n',pred_df.groupby('labels')['spatial_model_score'].min(), pred_df.groupby('labels')['spatial_model_score'].median(),pred_df.groupby('labels')['spatial_model_score'].max())
#         # print('\nClass Weighted\n',pred_df.groupby('labels')['class_weighted'].mean())
#         # print('\nModel Score\n',pred_df.groupby('labels')['model_score'].min(), pred_df.groupby('labels')['model_score'].median(),pred_df.groupby('labels')['model_score'].max())


#         if self.rank == 0:
#             pbar.close()

#         print(f"--- Searching running time: %s seconds ---" % (time.time() - select_start_time))
#         print(feature_df[feature_df.frame_id.isin(selected_frames)].groupby('labels').frame_id.count())
#         print(len(set(selected_frames)))
#         # save Embedding    
#         with open(os.path.join(self.active_label_dir, 'feature_record_epoch_{}_rank_{}.pkl'.format(cur_epoch, self.rank)), 'wb') as f:
#             pickle.dump(saving_dict, f)
#             print('successfully saved selected frames for epoch {} for rank {}'.format(cur_epoch, self.rank))
        
#         return selected_frames
