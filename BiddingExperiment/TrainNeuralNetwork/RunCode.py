import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import time
import torch.nn.functional as F
import pickle
from sklearn.metrics  import roc_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import plot_roc_curve


def train(net, train_loader, train_loader_oversampled, val_loader, \
    val_loader_oversampled, Y_tr_np, Y_val_np, optimizer, lossFnTr, \
    lossFnVal, init_epoch, final_epoch, name, device, lossSigm, \
    checkForDiv = False, tol = 1000000, saveNetDict = False, \
    pathToSave = None, scheduler =  None):
    net.train()
    epoch = init_epoch
    performance_tr_val = []
    while epoch <final_epoch:
        t_start = time.time()
        for _, (data, target) in enumerate(train_loader_oversampled):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = torch.squeeze(net(data)).to(device)
            loss = lossFnTr(output, target)
            loss.backward()
            optimizer.step()
        t_end = time.time()
        print('Finished Epoch '+ str(epoch)+', It took '+str(t_end-t_start)+' Seconds')
        if scheduler is not None:
            scheduler.step()
        epoch += 1
        epochBeforeChecking = epoch
        if checkForDiv:
            with torch.no_grad():
                for param in net.parameters():
                    if torch.max(torch.abs(param.data)) > tol:
                        print(name+ ' diverged at epoch '+str(epochBeforeChecking))
                        epoch = final_epoch + 1
        with torch.no_grad():
            if saveNetDict:
                torch.save(net.state_dict(), pathToSave + name +'_Epoch_'+str(epochBeforeChecking)+'.pt')
            # tr_over_loss = val_only_loss(net, lossFnVal, train_loader_oversampled, device)
            val_over_loss = val_only_loss(net, lossFnVal, val_loader_oversampled, device)
            # print('Running train auc')
            # tr_auc, tr_prec, tr_rec, tr_f1 = \
            #     val_for_auc_prec_rec(net, train_loader, device, lossSigm, Y_tr_np)
            print('Running val auc')
            val_auc, val_prec, val_rec, val_f1 = \
                val_for_auc_prec_rec(net, val_loader, device, lossSigm, Y_val_np)
            # all_res_this_epoch = [epoch, tr_over_loss, tr_auc, tr_prec, tr_rec, tr_f1, \
            #     val_over_loss, val_auc, val_prec, val_rec, val_f1]
            all_res_this_epoch = [epoch, val_over_loss, val_auc, val_prec, val_rec, val_f1]
            performance_tr_val.append(all_res_this_epoch)
            print('all_res_this_epoch: ' + str(all_res_this_epoch))
    return performance_tr_val
    

def val_only_loss(net, lossFn, data_loader_over, device):
    net.eval()
    val_loss = 0 
    with torch.no_grad():
        for data, target in data_loader_over:
            data, target = data.to(device), target.to(device)
            output = torch.squeeze(net(data)).to(device)
            val_loss += lossFn(output, target).item()
    val_loss /= len(data_loader_over.dataset)
    return val_loss


def val_for_auc_prec_rec(net, data_loader, device, lossSigm, labels):
    net.eval()
    correct = 0
    true_pos, true_neg, all_pos, all_neg = 0, 0, 0, 0
    sel_pos, sel_neg = 0.0, 0.0
    sigmFn = torch.nn.Sigmoid()
    softmaxFn = torch.nn.Softmax(dim =1)
    all_pred_prob = torch.empty(0).to(device)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = net(data).to(device)
            all_pos += torch.sum(target).item()
            all_neg += target.numel() - torch.sum(target).item()
            curr_pred_prob = 0
            curr_pred_thres = 0
            delta_corr_pos = 0
            delta_corr = 0
            if lossSigm: 
                curr_pred_prob = sigmFn(output)
                curr_pred_thres = (torch.squeeze(sigmFn(output)) >= 0.5).type(torch.float).to(device)
                delta_corr_pos = torch.sum(target * curr_pred_thres).item()
                delta_corr = torch.sum(target == curr_pred_thres).item()
                all_pred_prob = \
                    torch.cat((all_pred_prob, curr_pred_prob), dim =0)
            else:
                curr_pred_prob = softmaxFn(output)[:,1]
                curr_pred_thres = \
                    torch.squeeze(softmaxFn(output).argmax(dim = 1, keepdim = True)).type(torch.long).to(device)
                delta_corr_pos = torch.sum(target * curr_pred_thres).item()
                delta_corr = torch.sum(target == curr_pred_thres).item()
                all_pred_prob = \
                    torch.cat((all_pred_prob, curr_pred_prob), dim =0)
            # print('lossSigm: ' + str(lossSigm))
            # print('curr_pred_prob.size(): ' + str(curr_pred_prob.size()))
            sel_pos += torch.sum(curr_pred_thres).item()
            sel_neg += target.numel() - torch.sum(curr_pred_thres).item()
            true_pos += delta_corr_pos
            correct += delta_corr
            true_neg += delta_corr - delta_corr_pos
    ## Get precision, recall and f1 (f1 just bc is free)
    precision = (true_pos + 0.0)/(sel_pos + 0.00001)
    recall = (true_pos + 0.0)/(all_pos + 0.00001)
    f1_score = 2 * (precision * recall)/ (precision + recall + 0.00001) 

    ## calculate auc
    all_pred_prob_np = all_pred_prob.cpu().numpy()
    # print('np.shape(all_pred_prob_np): ' + str(np.shape(all_pred_prob_np)))
    # print('np.shape(labels): ' + str(np.shape(labels)))
    tpr, fpr, _ = roc_curve(labels[:len(all_pred_prob_np)], all_pred_prob_np, pos_label = 1) 
    auc = auc_score(tpr, fpr)
    return auc, precision, recall, f1_score



# def train(net, train_loader, val_loader, optimizer, lossFnTr, lossFnVal, init_epoch, final_epoch, \
#     name, device, lossSigm, checkForDiv = False, tol = 1000000, saveNetDict = False, \
#     pathToSave = None,  saveAccToo = False, scheduler =  None):
#     net.train()
#     epoch = init_epoch
#     performance_tr_val = []
#     while epoch <final_epoch:
#         t_start = time.time()
#         for _, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = torch.squeeze(net(data)).to(device)
#             loss = lossFnTr(output, target)
#             loss.backward()
#             optimizer.step()
#         t_end = time.time()
#         print('Finished Epoch '+ str(epoch)+', It took '+str(t_end-t_start)+' Seconds')
#         if scheduler is not None:
#             scheduler.step()
#         epoch += 1
#         epochBeforeChecking = epoch
#         if checkForDiv:
#             with torch.no_grad():
#                 for param in net.parameters():
#                     if torch.max(torch.abs(param.data)) > tol:
#                         print(name+ ' diverged at epoch '+str(epochBeforeChecking))
#                         epoch = final_epoch + 1
#         with torch.no_grad():
#             if saveNetDict:
#                 torch.save(net.state_dict(), pathToSave + name +'_Epoch_'+str(epochBeforeChecking)+'.pt')
#             if epoch!= (final_epoch + 1):
#                 if saveAccToo:
#                     # lossTr, accTr, recallTr, recallNegTr = \
#                     #     val(net, lossFnVal, train_loader, device, lossSigm, withAcc = saveAccToo)
#                     val_loss, all_pos, all_neg, correct, true_pos, sel_pos, true_neg, sel_neg = \
#                         val(net, lossFnVal, val_loader, device, lossSigm, withAcc = saveAccToo)
#                     performance_tr_val.append([epochBeforeChecking, str(val_loss), str(all_pos), str(all_neg),\
#                         str(correct), str(true_pos), str(sel_pos), str(true_neg), str(sel_neg)])
#                 else:
#                     #lossTr = val(net, lossFnVal, train_loader, device = device, withAcc = saveAccToo)
#                     lossVal = val(net, lossFnVal, val_loader, device, lossSigm, withAcc = saveAccToo)
#                     # print('Train Loss: '+str(lossTr)+', Validation Loss: '+str(lossVal))
#                     # performance_tr_val.append([str(epochBeforeChecking), str(lossTr), str(lossVal)])
#                     print('Train Loss: '+'skipped'+', Validation Loss: '+str(lossVal))
#                     performance_tr_val.append([str(epochBeforeChecking), 'skipped', str(lossVal)])
#             else:
#                 if saveAccToo:
#                     performance_tr_val.append([str(epochBeforeChecking), str(np.nan), str(np.nan), str(np.nan), \
#                         str(np.nan), str(np.nan), str(np.nan), str(np.nan)])
#                 else:
#                     performance_tr_val.append([str(epochBeforeChecking), str(np.nan), str(np.nan)])
#     return performance_tr_val
    

# def val(net, lossFn, val_loader, device, lossSigm, withAcc = True):
#     net.eval()
#     val_loss = 0
#     correct = 0
#     true_pos, true_neg, all_pos, all_neg = 0, 0, 0, 0
#     sel_pos, sel_neg = 0, 0
#     sigmFn = torch.nn.Sigmoid()
#     softmaxFn = torch.nn.Softmax(dim =1)
#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
#             output = torch.squeeze(net(data)).to(device)
#             val_loss += lossFn(output, target).item() # sum up batch loss
#             all_pos += torch.sum(target).item()
#             all_neg += target.numel() - torch.sum(target).item()
#             if withAcc:
#                 if not lossSigm:
#                     output = softmaxFn(output)
#                     # print('after applying softmax, output.size(): ' + str(output.size()))
#                     pred = torch.squeeze(output.argmax(dim = 1, keepdim = True)) # get the index of the max probability
#                     sel_pos += torch.sum(pred).item()
#                     sel_neg += target.numel() - torch.sum(pred).item()
#                     # print('pred, pred.size(): ' + str(pred.size()))
#                     delta_corr_pos = torch.sum(target * pred).item()
#                     delta_corr = torch.sum(target == pred).item()
#                     true_pos += delta_corr_pos
#                     correct += delta_corr
#                     true_neg += delta_corr - delta_corr_pos
#                 else:
#                     pred = (sigmFn(output) >= 0.5).type(torch.float) # get the index of the max log-probability
#                     sel_pos += torch.sum(pred).item()
#                     sel_neg += target.numel() - torch.sum(pred).item()
#                     delta_corr_pos = torch.sum(target * pred).item()
#                     delta_corr = torch.sum(target == pred).item()
#                     true_pos += delta_corr_pos
#                     correct += delta_corr
#                     true_neg += delta_corr - delta_corr_pos
#     val_loss /= len(val_loader.dataset)
#     if withAcc:
#         return [val_loss, all_pos, all_neg, correct, true_pos, sel_pos, true_neg, sel_neg]
#     else:
#         return val_loss