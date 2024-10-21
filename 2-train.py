import time
import argparse
import random

import numpy as np
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.util import *
from model.build_model import build_model
import scipy.stats as stats
import os

from collections import OrderedDict

cudnn.benchmark = True
cudnn.fastest = True

import warnings
warnings.filterwarnings("ignore")  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--lr',                type=float, default=1e-2            )
parser.add_argument('--epochs',            type=int,   default=500            )
parser.add_argument('--workers',           type=int,   default=0               )
parser.add_argument('--BN',                type=int,   default=64             )
parser.add_argument('--test_BN',           type=int,   default=64              )
parser.add_argument('--display',           type=int,   default=10              )
parser.add_argument('--pre_len',           type=int,   default=1               )
parser.add_argument('--early_stop_maxtry', type=int,   default=200             )
# parser.add_argument('--model_name',        type=str,   default='gcn'           )
parser.add_argument('--dropout',           type=float, default=0.5             )
parser.add_argument('--fine_grained',      action='store_true', default = False )
parser.add_argument('--node_number_coarse',type=int,   default=82              )
parser.add_argument('--node_number_fine',  type=int,   default=246             )
parser.add_argument('--node_features', type=str, default='adj',
                    choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix', 'eigenvector', 'eigen_norm'])
parser.add_argument('--pooling', type=str, default='concat',
                    choices=['sum', 'concat', 'mean'])
parser.add_argument('--gcn_mp_type', type=str, default='edge_node_concate',
                    choices=['weighted_sum', 'bin_concate', 'edge_weight_concate', 'edge_node_concate', 'node_concate'])
parser.add_argument('--gat_mp_type', type=str, default='attention_weighted',
                    choices=['attention_weighted', 'attention_edge_weighted', 'sum_attention_edge', 'edge_node_concate', 'node_concate'])
parser.add_argument('--enable_nni', action='store_true')
parser.add_argument('--n_GNN_layers', type=int, default=2)
parser.add_argument('--n_MLP_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=6)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gat_hidden_dim', type=int, default=8)
parser.add_argument('--edge_emb_dim', type=int, default=4)
parser.add_argument('--bucket_sz', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=112078)
parser.add_argument('--diff', type=float, default=0.2)
parser.add_argument('--mixup', type=int, default=1)
parser.add_argument('--gamma', type=float, default=1e-8)
args = parser.parse_args()
args.manualSeed = 101
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
print("Random Seed: ", args.manualSeed)


def main():
    matrics_val = np.zeros((6,1))
    matrics_test = np.zeros((6,1))
    device = torch.device('cuda')

    # read dataset
    cat_data = np.load('dataset/train/input/train.npz', allow_pickle=True)
    train_region_fmri = cat_data['region_fmri']
    train_community_fmri = cat_data['community_fmri']
    train_region_A = cat_data['region_A']
    train_community_A = cat_data['community_A']
    train_label = cat_data['gt']
    scaler = StandardScaler(min = train_label.min(0)[0], max = train_label.max(0)[0])
    train_label = scaler.transform(train_label)

    cat_data = np.load('dataset/val/input/val.npz', allow_pickle=True)
    val_region_fmri = cat_data['region_fmri']
    val_community_fmri = cat_data['community_fmri']
    val_region_A = cat_data['region_A']
    val_community_A = cat_data['community_A']
    val_label = cat_data['gt']
    val_label = scaler.transform(val_label)

    cat_data = np.load('dataset/test/input/test.npz', allow_pickle=True)
    test_region_fmri = cat_data['region_fmri']
    test_community_fmri = cat_data['community_fmri']
    test_region_A = cat_data['region_A']
    test_community_A = cat_data['community_A']
    test_label = cat_data['gt']
    test_label = scaler.transform(test_label)

    scaler.max = torch.tensor(scaler.max).to(device)
    scaler.min = torch.tensor(scaler.min).to(device)

    if args.fine_grained:
        train_loader = DataLoader_f(train_region_fmri, train_community_fmri, train_label, train_region_A, train_community_A, batch_size=args.BN, shuffle=False)
        val_loader = DataLoader_f(val_region_fmri, val_community_fmri, val_label, val_region_A, val_community_A,
                                  batch_size=args.BN, shuffle=False)
        test_loader= DataLoader_f(test_region_fmri, test_community_fmri, test_label, test_region_A, test_community_A, batch_size=args.test_BN, shuffle=False)
        GCN_f = build_model(args, device, fine_grained=args.fine_grained)
        GCN_f.train()
        optimizer = torch.optim.SGD(GCN_f.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    else:
        train_loader = DataLoader(train_region_fmri, train_label, train_region_A, batch_size=args.BN, shuffle=False)
        val_loader = DataLoader(val_region_fmri, val_label, val_region_A, batch_size=args.BN, shuffle=False)
        test_loader = DataLoader(test_region_fmri, test_label, test_region_A, batch_size=args.test_BN, shuffle=False)
        GCN = build_model(args, device, fine_grained=args.fine_grained)
        GCN.train()
        optimizer = torch.optim.SGD(GCN.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # 优化器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 600], gamma=0.1)
    mmin_val_loss = -10
    trycnt = 0

    for epoch in range(args.epochs):
        print('Model Train')
        total_loss = 0
        if args.fine_grained:
            for iter, (fmri, fmri_f, gt, corr, corr_f) in enumerate(train_loader):
                fmri, fmri_f, gt, corr, corr_f = fmri.cuda(), fmri_f.cuda(), gt.cuda(), corr.cuda(), corr_f.cuda()
                graph_batch = graph_data(fmri.float(), corr.float())
                graph_batch_f = graph_data(fmri_f.float(), corr_f.float())
                predict = GCN_f(graph_batch, graph_batch_f)
                optimizer.zero_grad()
                loss = torch.sqrt(F.mse_loss(predict, gt))
                loss.backward()
                total_loss = total_loss + loss.item()
                optimizer.step()

        else:
            for iter, (fmri, gt, corr) in enumerate(train_loader):
                fmri, gt, corr = fmri.cuda(), gt.cuda(), corr.cuda()
                graph_batch = graph_data(fmri.float(), corr.float())
                out = GCN(graph_batch)
                optimizer.zero_grad()
                gt = torch.squeeze(gt[:, :1], 1)
                out = torch.squeeze(out, 1)
                loss = torch.sqrt(F.mse_loss(out, gt))
                loss.backward()
                print("loss:%5.5f" % (loss))
                total_loss = total_loss + loss.item()
                optimizer.step()

        lr_scheduler.step()
        print("total_loss:%5.5f" % (total_loss / ((iter + 1))))
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        total_loss = total_loss / (iter + 1)
        mae = torch.mean(torch.abs(gt - predict))
        rmse = torch.sqrt(torch.mean((gt - predict) ** 2))
        mape = torch.mean(torch.abs((gt - predict) / gt)) * 100
        cc = np.corrcoef(torch.cat((gt, predict), dim=1).T.cpu().detach().numpy())
        print("total_loss:%5.5f, MAE: %5.5f, RMSE:%5.5f, MAPE:%5.5f, CC:%5.5f" % (
            total_loss, mae, rmse, mape, cc[0, 1]))

        print('Model Val')
        if args.fine_grained:
            GCN_f.eval()
            with torch.no_grad():
                for iter, (fmri, fmri_f, gt, corr, corr_f) in enumerate(val_loader):
                    fmri, fmri_f, gt, corr, corr_f = fmri.cuda(), fmri_f.cuda(), gt.cuda(), corr.cuda(), corr_f.cuda()
                    graph_batch = graph_data(fmri.float(), corr.float())
                    graph_batch_f = graph_data(fmri_f.float(), corr_f.float())
                    predict = GCN_f(graph_batch, graph_batch_f)
                    predict_val = inverse_transform(predict, scaler)
                    real_val = inverse_transform(gt, scaler)

                if iter == 0:
                    prediction_val = predict_val
                    ground_truth_val = real_val
                else:
                    prediction_val = torch.cat((prediction_val, predict_val), dim=0)
                    ground_truth_val = torch.cat((ground_truth_val, real_val), dim=0)

        else:
            GCN.eval()
            with torch.no_grad():
                for iter, (fmriv, gtv, corrv) in enumerate(val_loader):
                    fmriv, gtv, corrv = fmriv.cuda(), gtv.cuda(), corrv.cuda()
                    graph_batch = graph_data(fmriv.float(), corrv.float())
                    predict = GCN(graph_batch)
                    predict_val = inverse_transform(predict, scaler)
                    real_val = inverse_transform(gt, scaler)
                    optimizer.zero_grad()

                    if iter == 0:
                        prediction_val = predict_val
                        ground_truth_val = real_val
                    else:
                        prediction_val = torch.cat((prediction_val, predict_val), dim=0)
                        ground_truth_val = torch.cat((ground_truth_val, real_val), dim=0)

        mae = torch.mean(torch.abs(ground_truth_val - prediction_val))
        rmse = torch.sqrt(torch.mean((ground_truth_val - prediction_val) ** 2))
        mape = torch.mean(torch.abs((ground_truth_val - prediction_val) / ground_truth_val)) * 100
        prediction_val = np.asarray(prediction_val.cpu())
        ground_truth_val = np.asarray(ground_truth_val.cpu())
        cc = np.corrcoef(np.concatenate((ground_truth_val, prediction_val), axis=1).T)
        print("MAE: %5.5f, RMSE:%5.5f, MAPE:%5.5f, CC:%5.5f" % (mae, rmse, mape, cc[0, 1]))
        correlation, pvalue = stats.spearmanr(ground_truth_val, prediction_val)
        print("Pearson Correlation Coefficient", correlation, pvalue)

        if mmin_val_loss < cc[0, 1]:
            if epoch == 0:
                mmin_val_loss = mmin_val_loss
            else:
                mmin_val_loss = cc[0, 1]
            mmin_epoch = epoch
            if args.fine_grained:
                best_model = GCN_f
            else:
                best_model = GCN
            torch.save(best_model, 'model_net_best.pkl')
            trycnt = 0
            matrics_val[0, 0] = np.asarray(mae.cpu())
            matrics_val[1, 0] = np.asarray(rmse.cpu())
            matrics_val[2, 0] = np.asarray(mape.cpu())
            matrics_val[3, 0] = np.asarray(abs(cc[0, 1]))
            matrics_val[4, 0] = correlation
            matrics_val[5, 0] = pvalue
            np.savetxt('pre_val.csv', prediction_val, delimiter=',')
            np.savetxt('real_val.csv', ground_truth_val, delimiter=',')
            np.savetxt('matrics_val.csv', matrics_val, delimiter=',')
        print('early stop trycnt:', trycnt, mmin_epoch)

        # for early stop
        trycnt += 1
        if args.early_stop_maxtry < trycnt:
            print('early stop!')
            print('Model Test')
            GCN = torch.load('model_net_best.pkl').eval()
            with torch.no_grad():
                if args.fine_grained:
                    for iter, (fmri, fmri_f, gt, corr, corr_f) in enumerate(test_loader):
                        fmri, fmri_f, gt, corr, corr_f = fmri.cuda(), fmri_f.cuda(), gt.cuda(), corr.cuda(), corr_f.cuda()
                        graph_batch = graph_data(fmri.float(), corr.float())
                        graph_batch_f = graph_data(fmri_f.float(), corr_f.float())
                        predict = GCN_f(graph_batch, graph_batch_f)
                        predict_test = inverse_transform(predict, scaler)
                        real_test = inverse_transform(gt, scaler)

                        if iter == 0:
                            prediction_test = predict_test
                            ground_truth_test = real_test
                        else:
                            prediction_test = torch.cat((prediction_test, predict_test), dim=0)
                            ground_truth_test = torch.cat((ground_truth_test, real_test), dim=0)

                else:
                    for iter, (fmri, corr, gt) in enumerate(test_loader):
                        fmri, gt, corr = fmri.cuda(), corr.cuda(), gt.cuda()
                        graph_batch = graph_data(fmri.float(), corr.float())
                        predict = GCN(graph_batch)
                        predict_test = inverse_transform(predict, scaler)
                        real_test = inverse_transform(gt, scaler)

                        if iter == 0:
                            prediction_test = predict_test
                            ground_truth_test = real_test
                        else:
                            prediction_test = torch.cat((prediction_test, predict_test), dim=0)
                            ground_truth_test = torch.cat((ground_truth_test, real_test), dim=0)

                mae = torch.mean(torch.abs(ground_truth_test - prediction_test))
                rmse = torch.sqrt(torch.mean((ground_truth_test - prediction_test) ** 2))
                mape = torch.mean(torch.abs((ground_truth_test - prediction_test) / ground_truth_test)) * 100
                cc = np.corrcoef(np.concatenate(
                    (ground_truth_test.cpu().detach().numpy(), prediction_test.cpu().detach().numpy()), axis=1).T)
                print("MAE: %5.5f, RMSE:%5.5f, MAPE:%5.5f, CC:%5.5f" % (mae, rmse, mape, cc[0, 1]))
                prediction_test = np.asarray(prediction_test.cpu().detach().numpy())
                ground_truth_test = np.asarray(ground_truth_test.cpu().detach().numpy())
                correlation, pvalue = stats.spearmanr(ground_truth_test, prediction_test)
                print("Pearson Correlation Coefficient", correlation, pvalue)
                matrics_test[0, 0] = np.asarray(mae.cpu())
                matrics_test[1, 0] = np.asarray(rmse.cpu())
                matrics_test[2, 0] = np.asarray(mape.cpu())
                matrics_test[3, 0] = np.asarray(abs(cc[0, 1]))
                matrics_test[4, 0] = correlation
                matrics_test[5, 0] = pvalue
                np.savetxt('matrics_test.csv', matrics_test, delimiter=',')
                np.savetxt('pre_test.csv', prediction_test, delimiter=',')
                np.savetxt('real_test.csv', ground_truth_test, delimiter=',')
            return

    print('Model Test')
    GCN = torch.load('model_net_best.pkl').eval()
    with torch.no_grad():
        if args.fine_grained:
            for iter, (fmri, fmri_f, gt, corr, corr_f) in enumerate(test_loader):
                fmri, fmri_f, gt, corr, corr_f = fmri.cuda(), fmri_f.cuda(), gt.cuda(), corr.cuda(), corr_f.cuda()
                graph_batch = graph_data(fmri.float(), corr.float())
                graph_batch_f = graph_data(fmri_f.float(), corr_f.float())
                predict = GCN_f(graph_batch, graph_batch_f)
                predict_test = inverse_transform(predict, scaler)
                real_test = inverse_transform(gt, scaler)

                if iter == 0:
                    prediction_test = predict_test
                    ground_truth_test = real_test
                else:
                    prediction_test = torch.cat((prediction_test, predict_test), dim=0)
                    ground_truth_test = torch.cat((ground_truth_test, real_test), dim=0)

        else:
            for iter, (fmri, corr, gt) in enumerate(test_loader):
                fmri, gt, corr = fmri.cuda(), corr.cuda(), gt.cuda()
                graph_batch = graph_data(fmri.float(), corr.float())
                predict = GCN(graph_batch)
                predict_test = inverse_transform(predict, scaler)
                real_test = inverse_transform(gt, scaler)

                if iter == 0:
                    prediction_test = predict_test
                    ground_truth_test = real_test
                else:
                    prediction_test = torch.cat((prediction_test, predict_test), dim=0)
                    ground_truth_test = torch.cat((ground_truth_test, real_test), dim=0)

        mae = torch.mean(torch.abs(ground_truth_test - prediction_test))
        rmse = torch.sqrt(torch.mean((ground_truth_test - prediction_test) ** 2))
        mape = torch.mean(torch.abs((ground_truth_test - prediction_test) / ground_truth_test)) * 100
        cc = np.corrcoef(np.concatenate(
            (ground_truth_test.cpu().detach().numpy(), prediction_test.cpu().detach().numpy()), axis=1).T)
        print("MAE: %5.5f, RMSE:%5.5f, MAPE:%5.5f, CC:%5.5f" % (mae, rmse, mape, cc[0, 1]))
        prediction_test = np.asarray(prediction_test.cpu().detach().numpy())
        ground_truth_test = np.asarray(ground_truth_test.cpu().detach().numpy())
        correlation, pvalue = stats.spearmanr(ground_truth_test, prediction_test)
        print("Pearson Correlation Coefficient", correlation, pvalue)
        matrics_test[0, 0] = np.asarray(mae.cpu())
        matrics_test[1, 0] = np.asarray(rmse.cpu())
        matrics_test[2, 0] = np.asarray(mape.cpu())
        matrics_test[3, 0] = np.asarray(abs(cc[0, 1]))
        matrics_test[4, 0] = correlation
        matrics_test[5, 0] = pvalue
        np.savetxt('matrics_test.csv', matrics_test, delimiter=',')
        np.savetxt('pre_test.csv', prediction_test, delimiter=',')
        np.savetxt('real_test.csv', ground_truth_test, delimiter=',')
        return


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))


