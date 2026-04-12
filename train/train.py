from tasks.diagnosis_prediction import sequential_diagnosis_prediction_mimic3
from model.LINKO import Mega
from pyhealth.trainer import Trainer
import torch
import numpy as np
from pyhealth.datasets import MIMIC3Dataset
from utils.eval_test import evaluate, get_group_labels1, calculate_confidence_interval
import os
from pathlib import Path
from utils.data import customized_set_task_mimic3
import random
from pyhealth.datasets import split_by_patient, get_dataloader
from multiprocessing import freeze_support
from itertools import chain


def nfold_experiment(mimic3sample, epochs , ds_size_ratio, print_results=True, record_results=True):

    os.makedirs("output", exist_ok=True)
    os.makedirs("results_prompting", exist_ok=True)

    data = mimic3sample.samples
    co_occurrence_counts, groups1 = get_group_labels1(data)

    folds = int(os.getenv("FOLDS", "5"))
    smoke_mode = os.getenv("SMOKE_FOLDS", "0") == "1"
    if smoke_mode:
        folds = 1


    list_top_k = [3,5,7,10, 15, 20, 30]
    metrics_dict = {'roc_auc_samples': [], 'pr_auc_samples': [], 'f1_samples': []}

    for group_name in groups1.keys():
        metrics_dict[f'roc_auc_samples_{group_name}'] = []
        metrics_dict[f'pr_auc_samples_{group_name}'] = []


    for k in list_top_k:
        metrics_dict[f'acc_at_k={k}'] = []
        metrics_dict[f'hit_at_k={k}'] = []
        for group_name in groups1.keys():
            metrics_dict[f'Group_acc_at_k={k}@' + group_name] = []
            metrics_dict[f'Group_hit_at_k={k}@' + group_name] = []


    # For smoke mode (single fold), use holdout split so train set is never empty.
    if folds <= 1:
        train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1], seed=45)
        fold_splits = [(train_ds, val_ds, test_ds)]
    else:
        patient_ids = list(mimic3sample.patient_to_index.keys())
        rng = np.random.default_rng(45)
        rng.shuffle(patient_ids)
        fold_patient_splits = np.array_split(patient_ids, folds)

        fold_splits = []
        for fold_idx in range(folds):
            test_patients = set(fold_patient_splits[fold_idx].tolist())
            train_pool_patients = [
                p
                for i, split in enumerate(fold_patient_splits)
                if i != fold_idx
                for p in split.tolist()
            ]

            # Keep at least one patient in the training split.
            val_size = int(0.1 * len(train_pool_patients))
            val_size = max(1, val_size)
            val_size = min(val_size, max(0, len(train_pool_patients) - 1))

            val_patients = set(train_pool_patients[:val_size])
            train_patients = set(train_pool_patients[val_size:])

            if len(train_patients) == 0 and len(val_patients) > 0:
                moved = next(iter(val_patients))
                val_patients.remove(moved)
                train_patients.add(moved)

            train_idx = list(chain.from_iterable(mimic3sample.patient_to_index[p] for p in train_patients))
            val_idx = list(chain.from_iterable(mimic3sample.patient_to_index[p] for p in val_patients))
            test_idx = list(chain.from_iterable(mimic3sample.patient_to_index[p] for p in test_patients))

            fold_splits.append(
                (
                    torch.utils.data.Subset(mimic3sample, train_idx),
                    torch.utils.data.Subset(mimic3sample, val_idx),
                    torch.utils.data.Subset(mimic3sample, test_idx),
                )
            )

    for fold_idx, (train_ds, val_ds, test_ds) in enumerate(fold_splits):
        print(f'----------------------fold:{fold_idx + 1}/{len(fold_splits)}-----------------------')

        torch.manual_seed(45 + fold_idx)
        np.random.seed(45 + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(45 + fold_idx)

        train_loader = get_dataloader(train_ds, batch_size=252, shuffle=True)
        val_loader = get_dataloader(val_ds, batch_size=252, shuffle=False)
        test_loader = get_dataloader(test_ds, batch_size=252, shuffle=False)

        print('preprocessing done!')

        # Stage 3: define model
        use_gpu = os.getenv("USE_GPU", "1") == "1"
        if use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "USE_GPU=1 but CUDA is not available. "
                    "Install a CUDA-enabled PyTorch build or set USE_GPU=0."
                )
            device = "cuda:0"
        else:
            device = "cpu"

        if ds_size_ratio==1.0:
            ds_size_ratio_model = ''
        else:
            ds_size_ratio_model = '_' + str(ds_size_ratio)

        model = Mega(
            dataset=mimic3sample,
            train_dataset = train_ds,
            feature_keys=["conditions", "drugs", "procedures"],
            label_key="label",
            mode="multilabel",
            embedding_dim=256,dropout=0.5,nheads=1,nlayers=1,
            G_dropout=0.1,n_G_heads=4,n_G_layers=1,
            threshold3=0.00, threshold2=0.02, threshold1=0.00,
            n_hap_layers=1, n_hap_heads=2, hap_dropout=0.2,
            llm_model='llama3.1:latest', gpt_embd_path='saved_files/gpt_code_emb/tx-emb-3-small/include_all_parents2/', #gpt_embd_path='saved_files/gpt_code_emb/tx-emb-3-small/' => so far best results
            ds_size_ratio=ds_size_ratio_model,device=device, seed=45 + fold_idx,
        )
        model.to(device)


        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = torch.nn.DataParallel(model)

        model.to(device)

        # Stage 4: model training

        exp_path = f"./output/OntoFAR_{ds_size_ratio}/EXP_fold_{fold_idx + 1}"
        resume_training = os.getenv("RESUME_TRAINING", "0") == "1"
        resume_ckpt = os.getenv("RESUME_CKPT", "").strip()
        checkpoint_path = None
        if resume_training:
            if resume_ckpt:
                checkpoint_path = resume_ckpt
            else:
                candidate_last = os.path.join(exp_path, "last.ckpt")
                if os.path.isfile(candidate_last):
                    checkpoint_path = candidate_last
            if checkpoint_path is None:
                print(f"Resume requested for fold {fold_idx + 1}, but no checkpoint was found. Starting fresh.")
            else:
                print(f"Resuming fold {fold_idx + 1} from checkpoint: {checkpoint_path}")

        trainer = Trainer(model=model,
                          checkpoint_path=checkpoint_path,
                          metrics = ['roc_auc_samples', 'pr_auc_samples', 'f1_samples'],
                          enable_logging=True,
                          output_path=f"./output/OntoFAR_{ds_size_ratio}",
                          exp_name=f'EXP_fold_{fold_idx + 1}',
                          device=device)

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            optimizer_class =  torch.optim.Adam,
            optimizer_params = {"lr": 1e-3},
            weight_decay=0.0,
            monitor="pr_auc_samples",
            monitor_criterion='max',
            load_best_model_at_last=True
        )


        all_metrics = [

            "pr_auc_samples",
            "roc_auc_samples",
            "f1_samples",
        ]

        y_true, y_prob, loss = trainer.inference(val_loader)

        result = evaluate(y_true, y_prob, co_occurrence_counts, groups1, list_top_k=list_top_k, all_metrics=all_metrics)

        print ('\n', result)

        metrics_dict['pr_auc_samples'].append(result['pr_auc_samples'])
        metrics_dict['roc_auc_samples'].append(result['roc_auc_samples'])
        metrics_dict['f1_samples'].append(result['f1_samples'])

        for group_name in groups1.keys():
            metrics_dict[f'roc_auc_samples_{group_name}'].append(result[f'roc_auc_samples_{group_name}'])
            metrics_dict[f'pr_auc_samples_{group_name}'].append(result[f'pr_auc_samples_{group_name}'])


        for k in list_top_k:
            metrics_dict[f'acc_at_k={k}'].append(result[f'acc_at_k={k}'])
            metrics_dict[f'hit_at_k={k}'].append(result[f'hit_at_k={k}'])
            for group_name in groups1.keys():
                metrics_dict[f'Group_acc_at_k={k}@' + group_name].append(result[f'Group_acc_at_k={k}@' + group_name])
                metrics_dict[f'Group_hit_at_k={k}@' + group_name].append(result[f'Group_hit_at_k={k}@' + group_name])

    if print_results:
        print()
        print('mean pr_auc_samples:', np.mean(metrics_dict['pr_auc_samples']))
        print('max pr_auc_samples:', np.max(metrics_dict['pr_auc_samples']))
        print('min pr_auc_samples:', np.min(metrics_dict['pr_auc_samples']))
        print('CI pr_auc_samples:', calculate_confidence_interval(metrics_dict['pr_auc_samples']))

        print()

        print('mean roc_auc_samples:', np.mean(metrics_dict['roc_auc_samples']))
        print('max roc_auc_samples:', np.max(metrics_dict['roc_auc_samples']))
        print('min roc_auc_samples:', np.min(metrics_dict['roc_auc_samples']))
        print('CI roc_auc_samples:', calculate_confidence_interval(metrics_dict['roc_auc_samples']))
        print()

        print('mean f1_samples:', np.mean(metrics_dict['f1_samples']))
        print('max f1_samples:', np.max(metrics_dict['f1_samples']))
        print('min f1_samples:', np.min(metrics_dict['f1_samples']))
        print('CI f1_samples:', calculate_confidence_interval(metrics_dict['f1_samples']))
        print()

        for group_name in groups1:
            print()
            print(f'mean pr_auc_samples_{group_name}:', np.mean(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'max pr_auc_samples_{group_name}:', np.max(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'min pr_auc_samples_{group_name}:', np.min(metrics_dict[f'pr_auc_samples_{group_name}']))
            print(f'CI pr_auc_samples_{group_name}:',
                  calculate_confidence_interval(metrics_dict[f'pr_auc_samples_{group_name}']))
            print()

            print(f'mean roc_auc_samples_{group_name}:', np.mean(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'max roc_auc_samples_{group_name}:', np.max(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'min roc_auc_samples_{group_name}:', np.min(metrics_dict[f'roc_auc_samples_{group_name}']))
            print(f'CI roc_auc_samples_{group_name}:',
                  calculate_confidence_interval(metrics_dict[f'roc_auc_samples_{group_name}']))
            print()

        for k in list_top_k:
            print('------------------------------------------')

            print(f'mean acc_at_k={k}:', np.mean(metrics_dict[f'acc_at_k={k}']))
            print(f'max acc_at_k={k}:', np.max(metrics_dict[f'acc_at_k={k}']))
            print(f'min acc_at_k={k}:', np.min(metrics_dict[f'acc_at_k={k}']))
            print(f'CI acc_at_k={k}:', calculate_confidence_interval(metrics_dict[f'acc_at_k={k}']))
            print()

            print(f'mean hit_at_k={k}:', np.mean(metrics_dict[f'hit_at_k={k}']))
            print(f'max hit_at_k={k}:', np.max(metrics_dict[f'hit_at_k={k}']))
            print(f'min hit_at_k={k}:', np.min(metrics_dict[f'hit_at_k={k}']))
            print(f'CI hit_at_k={k}:', calculate_confidence_interval(metrics_dict[f'hit_at_k={k}']))
            print()

            for group_name in groups1:
                print(f'mean Group_acc_at_k={k}@{group_name}:',
                      np.mean(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'max Group_acc_at_k={k}@{group_name}:',
                      np.max(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'min Group_acc_at_k={k}@{group_name}:',
                      np.min(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print(f'CI Group_acc_at_k={k}@{group_name}:',
                      calculate_confidence_interval(metrics_dict[f'Group_acc_at_k={k}@' + group_name]))
                print()

                print(f'mean Group_hit_at_k={k}@{group_name}:',
                      np.mean(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'max Group_hit_at_k={k}@{group_name}:',
                      np.max(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'min Group_hit_at_k={k}@{group_name}:',
                      np.min(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print(f'CI Group_hit_at_k={k}@{group_name}:',
                      calculate_confidence_interval(metrics_dict[f'Group_hit_at_k={k}@' + group_name]))
                print()

    if record_results:
        with open(f'results_prompting/metrics_results_BestModel_OntoFAR_{ds_size_ratio}.txt', 'w') as file:
            file.write('\n')
            file.write(f'mean pr_auc_samples: {np.mean(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'max pr_auc_samples: {np.max(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'min pr_auc_samples: {np.min(metrics_dict["pr_auc_samples"])}\n')
            file.write(f'CI pr_auc_samples: {calculate_confidence_interval(metrics_dict["pr_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean roc_auc_samples: {np.mean(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'max roc_auc_samples: {np.max(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'min roc_auc_samples: {np.min(metrics_dict["roc_auc_samples"])}\n')
            file.write(f'CI roc_auc_samples: {calculate_confidence_interval(metrics_dict["roc_auc_samples"])}\n')
            file.write('\n')

            file.write(f'mean f1_samples: {np.mean(metrics_dict["f1_samples"])}\n')
            file.write(f'max f1_samples: {np.max(metrics_dict["f1_samples"])}\n')
            file.write(f'min f1_samples: {np.min(metrics_dict["f1_samples"])}\n')
            file.write(f'CI f1_samples: {calculate_confidence_interval(metrics_dict["f1_samples"])}\n')
            file.write('\n')

            for group_name in groups1:
                file.write('\n')
                file.write(
                    f'mean pr_auc_samples_{group_name}: {np.mean(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'max pr_auc_samples_{group_name}: {np.max(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'min pr_auc_samples_{group_name}: {np.min(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write(
                    f'CI pr_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"pr_auc_samples_{group_name}"])}\n')
                file.write('\n')

                file.write(
                    f'mean roc_auc_samples_{group_name}: {np.mean(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'max roc_auc_samples_{group_name}: {np.max(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'min roc_auc_samples_{group_name}: {np.min(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write(
                    f'CI roc_auc_samples_{group_name}: {calculate_confidence_interval(metrics_dict[f"roc_auc_samples_{group_name}"])}\n')
                file.write('\n')

            for k in list_top_k:
                file.write('------------------------------------------\n')

                file.write(f'mean acc_at_k={k}: {np.mean(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'max acc_at_k={k}: {np.max(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'min acc_at_k={k}: {np.min(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write(f'CI acc_at_k={k}: {calculate_confidence_interval(metrics_dict[f"acc_at_k={k}"])}\n')
                file.write('\n')

                file.write(f'mean hit_at_k={k}: {np.mean(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'max hit_at_k={k}: {np.max(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'min hit_at_k={k}: {np.min(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write(f'CI hit_at_k={k}: {calculate_confidence_interval(metrics_dict[f"hit_at_k={k}"])}\n')
                file.write('\n')

                for group_name in groups1:
                    file.write(
                        f'mean Group_acc_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_acc_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_acc_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_acc_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_acc_at_k={k}@" + group_name])}\n')
                    file.write('\n')

                    file.write('------------------------------------------\n')

                    file.write(
                        f'mean Group_hit_at_k={k}@{group_name}: {np.mean(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'max Group_hit_at_k={k}@{group_name}: {np.max(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'min Group_hit_at_k={k}@{group_name}: {np.min(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write(
                        f'CI Group_hit_at_k={k}@{group_name}: {calculate_confidence_interval(metrics_dict[f"Group_hit_at_k={k}@" + group_name])}\n')
                    file.write('\n')

    return

def main():
    freeze_support()

    project_root = Path(__file__).resolve().parents[1]
    mimic3_root = os.getenv("MIMIC3_ROOT", str(project_root / "datasets" / "MIMIC_III"))
    print(f"[DATA] Using MIMIC-III root: {mimic3_root}")

    mimic3_ds = MIMIC3Dataset(
        root=mimic3_root,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 4}})},
        dev=os.getenv("MIMIC_DEV", "0") == "1",
    )
    print('--mimic-III loaded.')

    mimic3sample = customized_set_task_mimic3(
        dataset=mimic3_ds,
        task_fn=sequential_diagnosis_prediction_mimic3,
        ccs_label=False,
        ds_size_ratio=1.0,
        seed=45,
    )
    print('--datasets created.')
    print(mimic3sample.stat())

    epochs = int(os.getenv("EPOCHS", "230"))
    nfold_experiment(mimic3sample, epochs=epochs, ds_size_ratio=1.0)


if __name__ == "__main__":
    main()