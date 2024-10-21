import numpy as np
import torch
import networkx as nx
from networkx.algorithms.community import louvain_communities
import os
import scipy.sparse as sp
import pandas as pd
import nibabel as nib

flag = 'train' # "train" or "test" or "val"
def get_graph(raw, k = 0.1):
    x_input = ((raw - torch.min(raw, dim=0)[0]) /
               torch.unsqueeze(torch.max(raw, dim=0)[0] - torch.min(raw, dim=0)[0], dim=0))
    A = np.corrcoef(np.asarray(x_input.T))
    a, _ = torch.topk(torch.tensor(A), k = int(node_number * k), dim=1)
    a_min = torch.min(a, dim=-1).values
    a_min = a_min.unsqueeze(-1).repeat(1, node_number)
    ge = torch.ge(torch.tensor(A), a_min)
    zero = torch.zeros_like(torch.tensor(A))
    graph = torch.where(ge, torch.tensor(A), zero)
    G = nx.from_numpy_array(np.array(graph))

    return G, x_input, raw, np.array(graph)

# first step -- AAL brain region & community feature
filepath = 'dataset/'+ flag + '/'
filenames = os.listdir(filepath)
label = np.loadtxt('dataset/aal_atlas.csv', delimiter=',')
cg_score = pd.read_csv('dataset/S900_Release_Subjects.csv')
cg_score = pd.DataFrame(cg_score)
r_max = int(label.max())
i = 0
region_fmri = np.zeros((len(filenames), r_max, 1200))
region_A = np.zeros((len(filenames), r_max, 246))
community_fmri = np.zeros((len(filenames), r_max * 3, 1200))
community_A = np.zeros((len(filenames), r_max * 3, 246))
gt = np.zeros((len(filenames), 1))
for f in filenames:
    s = f[-4:]
    if s != '.nii':
        continue
    s1 = f
    sample_index = cg_score[cg_score.Subject.values == int(100408)].index.to_list()
    # sample_index = cg_score[cg_score.Subject.values == int(f[:6])].index.to_list()
    cg_score_sample = cg_score.loc[sample_index]
    gt[i, :] = cg_score_sample.SCPT_SEN.values  # score_name in {'SCPT_SEN', "ProcSpeed_Unadj", ...}
    s1 = f[:6]
    dirs = 'dataset/'+ flag + '/' + 'input/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    nii_path = os.path.join(filepath, f)
    data = np.asarray(nib.load(nii_path).get_fdata())
    cortex = data[:, :59412]
    cortex = cortex[:, :59412]
    for r in range(int(r_max)):
        region_index = np.where(label == r + 1)
        region = cortex[:, region_index]
        region = np.squeeze(region, axis = 1)
        # community
        region = torch.FloatTensor(region)
        _, node_number = region.size()
        G, region_norm, _, _ = get_graph(region, k=0.3)
        community = list(louvain_communities(G, 1.2))
        if len(community) < 3:
            num = len(community)
        else:
            num = 3
        for c in range(num):
            subgraph = torch.tensor(nx.to_numpy_array(G, community[i]))
            _, com_number = subgraph.size()
            A = torch.ones((com_number, com_number))
            subgraph = subgraph.to(torch.float32)
            A = torch.where(subgraph > 0.0, A, subgraph)
            ed = sp.coo_matrix(A)
            indices = np.vstack((ed.row, ed.col))
            index = torch.LongTensor(indices)
            value = torch.FloatTensor(ed.data)
            edge_index = torch.sparse_coo_tensor(index, value, ed.shape)
            x_sub = region_norm[:, list(community[c])].T
            output = torch.sparse.mm(edge_index, x_sub)
            index_max = torch.argmax(torch.sum(A, axis=0), axis=0)
            com_feature = output[index_max, :]
            if c == 0:
                com_feature_all = torch.unsqueeze(com_feature, dim=1)
            else:
                com_feature_all = torch.cat((com_feature_all, torch.unsqueeze(com_feature, dim=1)), dim=1)
        if len(community) == 1:
            com_feature = torch.zeros(cortex.size(0), 2)
            com_feature_all = torch.cat((com_feature_all, com_feature), dim=1)
        if len(community) == 2:
            com_feature = torch.zeros(cortex.size(0), 1)
            com_feature_all = torch.cat((com_feature_all, com_feature), dim=1)

        # region
        A = torch.ones((community, community))
        ed = sp.coo_matrix(A)
        indices = np.vstack((ed.row, ed.col))
        index = torch.LongTensor(indices)
        value = torch.FloatTensor(ed.data)
        edge_index = torch.sparse_coo_tensor(index, value, ed.shape)
        output = torch.sparse.mm(edge_index, com_feature_all[:,:community].T)
        index_max = torch.argmax(torch.sum(A, axis=0), axis=0)
        region_feature = output[index_max, :]
        if r == 0:
            region_feature_all = torch.unsqueeze(region_feature, dim=1)
        else:
            region_feature_all = torch.cat((region_feature_all, torch.unsqueeze(region_feature, dim=1)), dim=1)
    print('done sub' + s1)

    region_fmri[i,:,:] = region_feature_all.T
    region_A[i,:,:] = torch.corrcoef(region_fmri)
    community_fmri[i,:,:] = com_feature_all.T
    community_A[i,:,:] = torch.corrcoef(community_fmri)
    i = i + 1
    np.savez(dirs + flag +'.npz', region_fmri = region_feature_all, region_A = region_A,
             community_fmri = community_fmri, community_A = community_A, gt = gt)
