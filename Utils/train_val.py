import torch
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score,
                             roc_curve, matthews_corrcoef, cohen_kappa_score, confusion_matrix,auc)
import numpy as np
import os
from torch import nn
from torch.cuda.amp import autocast as autocast, GradScaler

def vit_lr_schedule(epoch):
    if epoch < 50:
        lr = 1e-5
    elif epoch < 75:
        lr = 5e-6
    else:
        lr = 1e-6
    return lr


def view_results(mil_feature=None, mil_head=None, train_loader=None,
                          loss_fn=None, proba_mode=False, gpu_device=None, proba_value=0.85,
                          batch_size=4, bags_len=100, abla_type='tic'):
    mil_feature.eval()
    mil_head.eval()
    train_acc = []
    train_loss = []
    for train_img_list, train_label in train_loader:
        train_label = train_label.cuda()
        with torch.no_grad():
            train_pre_y = torch.zeros((1, 768)).cuda()
            for train_img in train_img_list:
                train_pre_y = torch.cat((train_pre_y, mil_feature(train_img.cuda())))
            train_pre_y = train_pre_y[1:]
            train_pre_y, _, _ = mil_head(train_pre_y)
            train_loss.append(loss_fn(train_pre_y, train_label).detach().cpu().numpy())
            train_pre_label = torch.argmax(train_pre_y, dim=1)
        train_acc.append(accuracy_score(train_label.detach().cpu().numpy(),
                                        train_pre_label.detach().cpu().numpy()))
    return train_acc, train_loss, train_label, train_pre_label


def testing_for_parallel(mil_feature=None, mil_head=None,class_num=None,
                                gpu_device=None, test_loader=None):
    mil_feature.eval()
    mil_head.eval()
    test_acc = []
    sum_label = torch.zeros(2).cuda(gpu_device)
    pre_y_sum = torch.zeros(2,class_num).cuda(gpu_device)
    pre_label = torch.zeros(2).cuda(gpu_device)
    for img_list, label in test_loader:
        label = label.cuda()
        with torch.no_grad():
            pre_y = torch.zeros((1, 768)).cuda()
            for img in img_list:
                pre_y = torch.cat((pre_y, mil_feature(img.cuda())))
            pre_y = pre_y[1:]
            pre_y, _, _ = mil_head(pre_y)
            test_pre_label = torch.argmax(pre_y, dim=1)
            test_acc.append(accuracy_score(label.detach().cpu().numpy(),
                                        test_pre_label.detach().cpu().numpy()))
            pre_label = torch.cat((pre_label, test_pre_label))
            sum_label = torch.cat((sum_label, label))
            pre_y_sum = torch.cat((pre_y_sum, pre_y))
    pre_label = pre_label[2:]
    pre_y_sum = torch.softmax(pre_y_sum[2:],dim=1)
    sum_label = sum_label[2:]
    print('-----------------------------------------------------------------------')
    print(' test_acc:{:.4}'.format(np.mean(test_acc)))
    print('-----------------------------------------------------------------------')
    print('classification_report:', '\n',
          classification_report(sum_label.cpu().numpy(), pre_label.cpu().numpy(), digits=4))
    print('-----------------------------------------------------------------------')
    print('AUC:',
          roc_auc_score(to_category(sum_label, class_num=class_num).ravel(),
                        pre_y_sum.cpu().numpy().ravel()))
    print('-----------------------------------------------------------------------')
    print('MCC:', matthews_corrcoef(sum_label.cpu().numpy(), pre_label.cpu().numpy()))
    print('kappa:', cohen_kappa_score(sum_label.cpu().numpy(), pre_label.cpu().numpy()))
    print('confusion matrix:',confusion_matrix(sum_label.cpu().numpy(), pre_label.cpu().numpy()))
    fpr, tpr, _ = roc_curve(to_category(sum_label, class_num=class_num).ravel(),
                        pre_y_sum.cpu().numpy().ravel())
    print(auc(fpr, tpr))

def to_category(label_tensor=None, class_num=3):
    label_tensor = label_tensor.cpu().numpy()
    label_inter = np.zeros((label_tensor.size, class_num))
    for i in range(label_tensor.size):
        label_inter[i, int(label_tensor[i])] = 1
    return label_inter

def training_for_parallel(mil_feature=None, mil_head=None, train_loader=None, val_loader=None, test_loader=None,
                            proba_mode=False, lr_fn=None, epoch=100, gpu_device=0, onecycle_mr=1e-2, proba_value=0.85,
                            weight_path=r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx\SwinT_1.pth',
                            batch_size=4, bags_len=100, max_input_len=None, current_lr=None,abla_type='tic', use_amp=True):
    loss_fn = nn.CrossEntropyLoss()
    mil_paras = [{'params': mil_feature.parameters()},
                 {'params': mil_head.parameters()}]
    print('########################## training results #########################')

    print('')
    print(f'###################### training in {abla_type}##########################')

    scaler = GradScaler()
    best_val_acc = 0
    for i in range(epoch):
        rmp_optim = torch.optim.RMSprop(mil_paras, lr=vit_lr_schedule(i))
        mil_feature.train()
        mil_head.train()
        for img_data_list, img_label in train_loader:
            # torch.autograd.set_detect_anomaly(True)
            img_label = img_label.cuda()
            pre_y = torch.zeros((1, 768)).cuda()
            if use_amp:
                with autocast():
                    for img_data in img_data_list:

                        if img_data.shape[0] <= max_input_len:
                            pre_y = torch.cat((pre_y, mil_feature(img_data.cuda())))
                        else:
                            # max_input_len in case OOM
                            group_count = int(img_data.shape[0] / max_input_len)
                            for img_data_i in range(group_count):
                                groupIn_y = mil_feature(
                                    img_data[max_input_len * img_data_i:(max_input_len * (img_data_i + 1)), :, :,
                                    :].cuda())


                                pre_y = torch.cat((pre_y, groupIn_y))
                                torch.cuda.empty_cache()
                            if group_count * max_input_len < img_data.shape[0]:
                                remain_y = mil_feature(img_data[group_count * max_input_len:, :, :, :].cuda())
                                pre_y = torch.cat((pre_y, remain_y))
                            else:
                                pass

                    pre_y = pre_y[1:]
                    pre_y, min_dis, max_dis = mil_head(pre_y)
                    loss_value = loss_fn(pre_y, img_label) + min_dis - max_dis
                scaler.scale(loss_value).backward()
                scaler.step(rmp_optim)
                scaler.update()
                rmp_optim.zero_grad()
            else:
                for img_data in img_data_list:
                    if img_data.shape[0] <= max_input_len:
                        pre_y = torch.cat((pre_y, mil_feature(img_data.cuda())))
                    else:
                        group_count = int(img_data.shape[0] / max_input_len)
                        for img_data_i in range(group_count):
                            groupIn_y = mil_feature(
                                img_data[max_input_len * img_data_i:(max_input_len * (img_data_i + 1)), :, :, :].cuda())

                            pre_y = torch.cat((pre_y, groupIn_y))
                            torch.cuda.empty_cache()
                        if group_count * max_input_len < img_data.shape[0]:
                            remain_y = mil_feature(img_data[group_count * max_input_len:, :, :, :].cuda())
                            pre_y = torch.cat((pre_y, remain_y))
                        else:
                            pass
                pre_y = pre_y[1:]
                pre_y, min_dis, max_dis = mil_head(pre_y)

                loss_value = loss_fn(pre_y, img_label) + min_dis - max_dis
                loss_value.backward()
                rmp_optim.step()
                rmp_optim.zero_grad()

        val_acc, val_loss, _, _ = view_results(mil_feature=mil_feature, mil_head=mil_head,
                                                        train_loader=val_loader, loss_fn=loss_fn,
                                                        proba_mode=proba_mode, gpu_device=gpu_device,
                                                        proba_value=proba_value, batch_size=batch_size,
                                                        bags_len=bags_len, abla_type=abla_type)

        print('epoch ' + str(i + 1),
              ' val_loss:{:.4}'.format(np.mean(val_loss)),
              ' val_acc:{:.4}'.format(np.mean(val_acc)))
        print('')
        cur_val_acc = np.mean(val_acc)
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_epoch = i + 1
            g = mil_feature.state_dict()
            torch.save(g, os.path.join(weight_path,
                                       f'SwinT_Feature_ValAcc_{best_val_acc}_Epoch{best_epoch}.pth'))
            g_1 = mil_head.state_dict()
            torch.save(g_1, os.path.join(weight_path,
                                         f'SwinT_Head_ValAcc_{best_val_acc}_Epoch{best_epoch}.pth'))
        print(f'Best model is saved at Epoch:{best_epoch} with Val_Acc:{best_val_acc}')
        print('')


    test_acc, test_loss, _, _ = view_results(mil_feature=mil_feature, mil_head=mil_head,
                                                      train_loader=test_loader, loss_fn=loss_fn,
                                                      proba_mode=proba_mode, gpu_device=gpu_device,
                                                      proba_value=proba_value, batch_size=batch_size,
                                                      bags_len=bags_len, abla_type=abla_type)

    print(' test_acc:{:.4}'.format(np.mean(test_acc)))
    g = mil_feature.state_dict()
    torch.save(g, os.path.join(weight_path,
                                f'SwinT_Feature_Final.pth'))
    g_1 = mil_head.state_dict()
    torch.save(g_1, os.path.join(weight_path,
                                    f'SwinT_Head_Final.pth'))

    return test_acc