import torch
from sklearn.metrics import roc_auc_score #, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
# import matplotlib.pyplot as plt

# def plot_roc(y_true, y_pred_prob):
#     fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
#     roc_auc = auc(fpr, tpr)
#
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()
#
#
# def get_f1_score(y_true, y_pred):
#     """
#     Attention!
#     tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     cf_m = confusion_matrix(y_true, y_pred)
#     print("tn:", cf_m[0, 0])
#     print("fp:", cf_m[0, 1])
#     print("fn:", cf_m[1, 0])
#     print("tp:", cf_m[1, 1])
#     precision = cf_m[1, 1] / (cf_m[1, 1] + cf_m[0, 1])
#     recall = cf_m[1, 1] / (cf_m[1, 1] + cf_m[1, 0])
#     f1 = 2 * (precision * recall) / (precision + recall)
#
#     return precision, recall, f1
#
#
# def get_recall(y_true, y_pred):
#     """
#     Attention!
#     tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]
#     :param y_true:
#     :param y_pred:
#     :return:
#     """
#     cf_m = confusion_matrix(y_true, y_pred)
#
#     # print(cf_m)
#     return cf_m[1, 1] / (cf_m[1, 1] + cf_m[1, 0])
#
# def plot_p_r_curve(y_true, y_pred_prob, best_logits, train_idx, val_idx):
#     thresholds = [0]
#     precision, recall, thresholds_2 = precision_recall_curve(y_true, y_pred_prob)
#
#     # print('p',precision[0])
#     # print('r',recall[0])
#     # print('t',thresholds[0])
#
#     # avp = average_precision_score(y_true, y_pred_prob)
#     thresholds.extend(thresholds_2)
#
#     gain_list = []
#
#     thresholds_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
#                        0.95]
#     for each in thresholds_list:
#         gain_list.append(torch.sum(torch.gt(best_logits, each)) - torch.sum(torch.gt(best_logits[train_idx], each))
#                          - torch.sum(torch.gt(best_logits[val_idx], each)))
#
#     plt.plot(recall, precision, color='blue', lw=2, label='P-R Curve')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve for Binary Classification')
#     plt.legend(loc="top right")
#     plt.savefig("gcn_pr.png")
#     plt.close()
#     plt.plot(thresholds, precision, color='blue', lw=2, label='Threshold-Precision Curve')
#     plt.xlabel('Threshold')
#     plt.ylabel('Precision')
#     plt.title('Threshold-Precision Curve for Binary Classification')
#     plt.legend(loc="top right")
#     plt.savefig("gcn_tp.png")
#     plt.close()
#     plt.plot(thresholds, recall, color='blue', lw=2, label='Threshold-Recall Curve')
#     plt.xlabel('Threshold')
#     plt.ylabel('Recall')
#     plt.title('Threshold-Recall Curve for Binary Classification')
#     plt.legend(loc="top right")
#     plt.savefig("gcn_tr.png")
#     plt.close()
#     plt.plot(thresholds_list, gain_list, color='blue', lw=1, label='Threshold-Gain Curve')
#     plt.xlabel('Threshold')
#     plt.ylabel('Gain')
#     plt.title('Threshold-Gain Curve for Binary Classification')
#     plt.legend(loc="top right")
#     plt.savefig("gcn_tg.png")
#
#
# device = "cpu"
# def assign_a_gpu(gpu_no):
#     device = torch.device("cuda:%s"%(str(gpu_no)) if torch.cuda.is_available() else "cpu")
#     return device