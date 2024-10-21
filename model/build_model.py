import torch
from layer.gcn import GCN
from hgl import HGL, HGL_f
from layer.mlp import MLP

def build_model(args, device, fine_grained):
    if fine_grained:
        model = HGL_f(args, GCN(args.node_number_coarse, args, args.node_number_coarse, pre_len=args.pre_len),
                        GCN(args.node_number_fine, args, args.node_number_fine, pre_len=args.pre_len),
                        MLP(2 * args.hidden_dim, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU,
                            pre_len=args.pre_len)).to(device)

    else:
        model = HGL(args, GCN(args.node_number_coarse, args, args.node_number_coarse, pre_len=args.pre_len),
                        MLP(2 * args.node_number_coarse, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU,
                            pre_len=args.pre_len)).to(device)
    return model

