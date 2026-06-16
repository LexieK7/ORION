import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=5,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/cc_trainset.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            #label_dict =  {'Not identified':0, 'Present':1},
                            #label_dict = {'Moderately differentiated':0, 'Poorly differentiated':1 },
                            label_dict = {'pMMR':0, 'dMMR':1 },
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=20
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/cc_trainset.csv', 
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Anaplastic large cell lymphoma,ALK negative':0, 'Anaplastic large cell lymphoma,ALK positive':1, 'Angioimmunoblastic T-cell lymphoma':2, 
                            'B-lymphoblastic lymphoma':3, 'Burkitt lymphoma':4, 'Chronic lymphocytic leukemia, small lymphocytic lymphoma':5, 
                            'Classical Hodgkins lymphoma, lymphocyte-rich type':6, 'Classical Hodgkins lymphoma, mixed cell type':7, 'Classical Hodgkins lymphoma, nodular sclerosis type':8, 
                            'Diffuse large B-cell lymphoma, non-specific type':9, 'Extranodal NK-T-cell lymphoma':10, 'Follicular lymphoma':11, 
                            'High-grade B-cell lymphoma with MYC and BCL2 rearrangement':12, 'Lymphoma in the intraductal marginal zone':13, 'Lymphoma of the extranodal margin area of mucosa-associated tissue':14, 
                            'Mantle cell lymphoma':15, 'Nodular lymphocyte-predominant Hodgkins lymphoma':16, 'Peripheral T-cell lymphoma, NOS':17, 
                             'Plasma cell tumor':18, 'T-lymphoblastic lymphoma':19, 
                              },
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



