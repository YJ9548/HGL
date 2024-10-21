import torch
import torch.nn as nn


class HGL_f(torch.nn.Module):
    def __init__(self, args, gnn1, gnn2, discriminator=lambda x, y: x @ y.t()):
        super(HGL_f, self).__init__()
        self.gnn_c = gnn1
        self.gnn_f = gnn2
        self.pooling = args.pooling
        self.output_linear = nn.Linear(in_features = 2 * args.pre_len, out_features=args.pre_len)

    def forward(self, data, data_f):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x_f, edge_index_f, edge_attr_f, batch_f = data_f.x, data_f.edge_index, data_f.edge_attr, data_f.batch
        g_c = self.gnn_c(x, edge_index, edge_attr, batch)
        g_f = self.gnn_f(x_f, edge_index_f, edge_attr_f, batch_f)
        g = torch.cat((0.8 * g_c, 0.2 * g_f), dim=1)
        output = self.output_linear(g)
        
        return output


class HGL(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(HGL, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.output_linear = nn.Linear(in_features=args.pre_len, out_features=args.pre_len)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        g = self.gnn(x, edge_index, edge_attr, batch)
        output = self.output_linear(g)

        return output
