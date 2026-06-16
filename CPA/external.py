import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
import re
from torch.utils.data import DataLoader, sampler
from sklearn.metrics import confusion_matrix, roc_auc_score
from models.model_clam import CLAM_MB, CLAM_SB, CLAM_transformer
from utils.utils import print_network
from utils.core_utils import summary
from datasets.dataset_generic import Generic_MIL_Dataset
from topk.svm import SmoothTop1SVM
from sklearn.metrics import precision_recall_fscore_support



def collate_MIL(batch):
    slide_ids = [item[0] for item in batch]
    imgs = torch.cat([item[1] for item in batch], dim=0)
    labels = torch.LongTensor([item[2] for item in batch])
    return [imgs, labels, slide_ids]


def convert_attention_branch_keys(state_dict, from_branch=2, to_branch=3):
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'instance_loss_fn' in k:
            continue
        k = k.replace('.module', '')
        k = re.sub(r'(attention_net)\.{}(\.)'.format(from_branch),
                   r'\1.{}\2'.format(to_branch), k)
        new_state_dict[k] = v
    return new_state_dict


# ===============================
# Config
# ===============================
class_num = 20

data_path = "DATA-PATH"
csv_path = "./cc_testset.csv"
weight_dir = "MODEL-WEIGHT-DIR"
output_path = "[external_HP]ours.xlsx"


# ===============================
# Evaluation
# ===============================
fold_results = []
all_fold_preds = {}
all_fold_votes = {}
all_fold_labels = {}

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

    for j in range(5):

        print(f"\n===== Fold {j} =====")

        instance_loss_fn = SmoothTop1SVM(n_classes=class_num).cuda()

        model_dict = {
            "dropout": True,
            'n_classes': class_num,
            'k_sample': 8
        }

        model = CLAM_transformer(**model_dict)

        #model = CLAM_MB(**model_dict)

        print_network(model)

        model_weight = f"{weight_dir}/s_{j}_checkpoint.pt"

        print("Loading model from:", model_weight)

        ckpt = torch.load(model_weight)

        ckpt_clean = convert_attention_branch_keys(
            ckpt,
            from_branch=2,
            to_branch=3
        )

        model.load_state_dict(ckpt_clean, strict=True)

        model.relocate()

        model.eval()

        print("Preparing dataset...")

        dataset = Generic_MIL_Dataset(
            csv_path=csv_path,
            data_dir=data_path,
            shuffle=False,
            print_info=True,
            label_dict={
                'Anaplastic large cell lymphoma,ALK negative': 0,
                'Anaplastic large cell lymphoma,ALK positive': 1,
                'Angioimmunoblastic T-cell lymphoma': 2,
                'B-lymphoblastic lymphoma': 3,
                'Burkitt lymphoma': 4,
                'Chronic lymphocytic leukemia, small lymphocytic lymphoma': 5,
                'Classical Hodgkins lymphoma, lymphocyte-rich type': 6,
                'Classical Hodgkins lymphoma, mixed cell type': 7,
                'Classical Hodgkins lymphoma, nodular sclerosis type': 8,
                'Diffuse large B-cell lymphoma, non-specific type': 9,
                'Extranodal NK-T-cell lymphoma': 10,
                'Follicular lymphoma': 11,
                'High-grade B-cell lymphoma with MYC and BCL2 rearrangement': 12,
                'Lymphoma in the intraductal marginal zone': 13,
                'Lymphoma of the extranodal margin area of mucosa-associated tissue': 14,
                'Mantle cell lymphoma': 15,
                'Nodular lymphocyte-predominant Hodgkins lymphoma': 16,
                'Peripheral T-cell lymphoma, NOS': 17,
                'Plasma cell tumor': 18,
                'T-lymphoblastic lymphoma': 19,
            },
            patient_strat=False,
            ignore=[]
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler.SequentialSampler(dataset),
            collate_fn=collate_MIL
        )

        print("Evaluating model...")

        patient_results, test_error, auc, acc_logger, prec, recall, f1 = summary(
            model,
            loader,
            class_num
        )

        fold_records = []

        y_true = []
        y_pred = []
        y_prob = []

        for patient_id, patient in patient_results.items():

            label = patient['label']

            prob = np.array(
                patient['prob'][0],
                dtype=np.float32
            )

            prob = prob / (prob.sum() + 1e-12)

            pred = np.argmax(prob)

            y_true.append(label)
            y_pred.append(pred)
            y_prob.append(prob)

            fold_records.append({
                "Patient_ID": patient_id,
                "True_Label": label,
                "Pred_Label": int(pred),
                "Probabilities": prob.tolist()
            })

            if patient_id not in all_fold_preds:
                all_fold_preds[patient_id] = []
                all_fold_votes[patient_id] = []
                all_fold_labels[patient_id] = label

            all_fold_preds[patient_id].append(prob)
            all_fold_votes[patient_id].append(pred)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # ===============================
        # Robust Multi-class AUROC
        # ===============================
        auc_score = None

        try:

            present_classes = np.unique(y_true)

            y_prob_filtered = y_prob[:, present_classes]

            y_prob_filtered = y_prob_filtered / (
                y_prob_filtered.sum(axis=1, keepdims=True) + 1e-12
            )

            auc_score = roc_auc_score(
                y_true,
                y_prob_filtered,
                multi_class='ovr',
                labels=present_classes
            )

        except Exception as e:
            print("AUC skipped due to:", e)

        accuracy = np.mean(y_true == y_pred)

        fold_results.append({
            "Fold": j,
            "AUC": auc_score,
            "Accuracy": accuracy,
            "Precision": prec,
            "Recall": recall,
            "F1": f1
        })

        df_fold = pd.DataFrame(fold_records)

        df_fold.to_excel(
            writer,
            sheet_name=f"Fold_{j}",
            index=False
        )


        torch.cuda.empty_cache()

    # ===============================
    # Ensemble 1: Mean probability
    # ===============================
    print("\n===== Ensemble (Mean Probability) =====")

    ensemble_mean_records = []

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for pid, prob_list in all_fold_preds.items():

        probs = np.stack(prob_list)

        mean_prob = probs.mean(axis=0)

        mean_prob = mean_prob / (mean_prob.sum() + 1e-12)

        pred_label = int(np.argmax(mean_prob))

        true_label = all_fold_labels[pid]

        y_true_all.append(true_label)
        y_pred_all.append(pred_label)
        y_prob_all.append(mean_prob)

        ensemble_mean_records.append({
            "Patient_ID": pid,
            "True_Label": true_label,
            "Pred_Label": pred_label,
            "Mean_Probabilities": mean_prob.tolist()
        })

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    # ===============================
    # Ensemble AUROC
    # ===============================
    auc_mean = None

    try:

        present_classes = np.unique(y_true_all)

        y_prob_filtered = y_prob_all[:, present_classes]

        y_prob_filtered = y_prob_filtered / (
            y_prob_filtered.sum(axis=1, keepdims=True) + 1e-12
        )

        auc_mean = roc_auc_score(
            y_true_all,
            y_prob_filtered,
            multi_class='ovr',
            labels=present_classes
        )

    except Exception as e:
        print("Ensemble AUC skipped:", e)

    # Accuracy
    acc_mean = np.mean(y_true_all == y_pred_all)

    # Precision / Recall / F1
    prec_mean, recall_mean, f1_mean, _ = precision_recall_fscore_support(
        y_true_all,
        y_pred_all,
        average="macro",
        zero_division=0
    )

    df_ensemble_mean = pd.DataFrame(ensemble_mean_records)

    df_ensemble_mean.to_excel(
        writer,
        sheet_name="Ensemble_mean",
        index=False
    )

    # ===============================
    # Ensemble 2: Majority Voting
    # ===============================
    print("\n===== Ensemble (Majority Voting) =====")

    ensemble_vote_records = []

    y_true_vote = []
    y_pred_vote = []

    for pid, preds in all_fold_votes.items():

        true_label = all_fold_labels[pid]

        vote_pred = int(np.bincount(preds).argmax())

        y_true_vote.append(true_label)
        y_pred_vote.append(vote_pred)

        ensemble_vote_records.append({
            "Patient_ID": pid,
            "True_Label": true_label,
            "Vote_Pred": vote_pred,
            "Votes": preds
        })

    acc_vote = np.mean(
        np.array(y_true_vote) == np.array(y_pred_vote)
    )

    df_ensemble_vote = pd.DataFrame(ensemble_vote_records)

    df_ensemble_vote.to_excel(
        writer,
        sheet_name="Ensemble_vote",
        index=False
    )

    # ===============================
    # Summary Sheet
    # ===============================
    df_summary = pd.DataFrame(fold_results)

    df_summary.loc[len(df_summary)] = {
        "Fold": "Ensemble_mean",
        "AUC": auc_mean,
        "Accuracy": acc_mean,
        "Precision": prec_mean,
        "Recall": recall_mean,
        "F1": f1_mean
    }

    df_summary.loc[len(df_summary)] = {
        "Fold": "Ensemble_vote",
        "AUC": "-",
        "Accuracy": acc_vote,
        "Precision": "-",
        "Recall": "-",
        "F1": "-"
    }

    df_summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

print("\nSaved all fold and ensemble results to:", output_path)
