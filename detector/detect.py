import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from transformers import AdamW
from openprompt.plms import load_plm
from openprompt.prompts import SoftVerbalizer, MixedTemplate
from openprompt.pipeline_base import PromptDataLoader, PromptForClassification
from detector.utils import OpenPromptProcessor


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate(classifier, dataloader, use_cuda):
    all_preds, all_labels = [], []
    
    for inputs in dataloader:
        if use_cuda:
            inputs = inputs.cuda()
        
        logits = classifier(inputs)
        labels = inputs['label']
        
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    
    return all_labels, all_preds


def compute_metrics(labels, preds):
    matrix = confusion_matrix(labels, preds)
    tn, fp, fn, tp = matrix.ravel()
    macro_f1 = f1_score(labels, preds, average='macro') 
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn)
    auc = roc_auc_score(labels, preds)
    
    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
            'macro_f1': macro_f1, 'accuracy': acc, 
            'fpr': fpr, 'auc': auc, 'conf_matrix': matrix}


def log_best_results(all_cf, all_acc, all_mf1, all_fpr, all_auc):
    '''
    print(f'Best test acc: {max(all_acc)} / epoch: {all_acc.index(max(all_acc))}')
    print(f'Best test macro f1: {max(all_mf1)} / epoch: {all_mf1.index(max(all_mf1))}')
    print(f'Best test fpr: {min(all_fpr)} / epoch: {all_fpr.index(min(all_fpr))}')
    print(f'Best test auc: {max(all_auc)} / epoch: {all_auc.index(max(all_auc))}')
    '''

    best_epochs = {
        "accuracy": all_acc.index(max(all_acc)),
        "macro_f1": all_mf1.index(max(all_mf1)),
        "fpr": all_fpr.index(min(all_fpr)),
        "auc": all_auc.index(max(all_auc))
    }

    for metric, epoch in best_epochs.items():
        print(f'Based on the best {metric} ... ')
        print(f'Epoch: {epoch}')
        print(f'Accuracy: {all_acc[epoch]}')
        print(f'Macro F1: {all_mf1[epoch]}')
        print(f'FPR: {all_fpr[epoch]}')
        print(f'AUC: {all_auc[epoch]}')
        best_cf_dict = all_cf[epoch]
        tn, fp, fn, tp = best_cf_dict['tn'], best_cf_dict['fp'], best_cf_dict['fn'], best_cf_dict['tp']
        print(f'[Confusion Matrix] TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')


def main(args):
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    with open(args.template_path, 'r') as f:
        template_text = json.load(f)['template']

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(args.save_dir, f"{timestamp}")
    os.makedirs(result_dir)

    with open(os.path.join(result_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    processor = OpenPromptProcessor()
    dataset = {
        'train': processor.get_examples(args.data_dir, "train"),
        'test': processor.get_examples(args.data_dir, "test")
    }

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name, model_path=args.model_path)
    template = MixedTemplate(model=plm, tokenizer=tokenizer, text=template_text)

    train_dataloader = PromptDataLoader(dataset=dataset["train"], 
                                        template=template, 
                                        tokenizer=tokenizer, 
                                        tokenizer_wrapper_class=WrapperClass, 
                                        max_seq_length=args.max_seq_length, 
                                        decoder_max_length=args.decoder_max_length, 
                                        batch_size=args.train_batch_size, 
                                        shuffle=True, 
                                        teacher_forcing=False, 
                                        predict_eos_token=False, 
                                        truncate_method="head")

    test_dataloader = PromptDataLoader(dataset=dataset['test'], 
                                       template=template, 
                                       tokenizer=tokenizer, 
                                       tokenizer_wrapper_class=WrapperClass, 
                                       max_seq_length=args.max_seq_length, 
                                       decoder_max_length=args.decoder_max_length, 
                                       batch_size=args.test_batch_size, 
                                       shuffle=False, 
                                       teacher_forcing=False, 
                                       predict_eos_token=False, 
                                       truncate_method='head')

    verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=2)
    classifier = PromptForClassification(plm=plm, template=template, verbalizer=verbalizer, freeze_plm=False)
    classifier = classifier.cuda() if args.use_cuda else classifier

    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in classifier.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in classifier.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
        
    optimizer_grouped_parameters2 = [
        {'params': classifier.verbalizer.group_parameters_1, "lr": args.lr_group_1},
        {'params': classifier.verbalizer.group_parameters_2, "lr": args.lr_group_2},
    ]
    
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.lr_group_1, no_deprecation_warning=True)
    optimizer2 = AdamW(optimizer_grouped_parameters2, no_deprecation_warning=True)

    all_cf, all_acc, all_mf1, all_fpr, all_auc = [], [], [], [], []
    all_results = []
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        classifier.train()
        total_loss = 0

        for inputs in train_dataloader:
            if args.use_cuda:
                inputs = inputs.cuda()
            logits = classifier(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            total_loss += loss.item()
            
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()

        labels, preds = evaluate(classifier, test_dataloader, args.use_cuda)
        metrics = compute_metrics(labels, preds)

        all_results.append([epoch, labels, preds])
        
        all_cf.append(metrics)
        all_acc.append(metrics['accuracy'])
        all_mf1.append(metrics['macro_f1'])
        all_fpr.append(metrics['fpr'])
        all_auc.append(metrics['auc'])

    df_results = pd.DataFrame(all_results, columns=['epoch', 'labels', 'preds'])
    df_results.to_json(os.path.join(result_dir, 'eval_results.jsonl'), orient='records', lines=True)
    log_best_results(all_cf, all_acc, all_mf1, all_fpr, all_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help='Directory to load the data'
                        )
    parser.add_argument('--save_dir', 
                        type=str, 
                        required=True, 
                        help='Directory to save the results'
                        )
    parser.add_argument('--use_cuda', 
                        type=bool, 
                        required=True, 
                        help='Whether to use cuda'
                        )
    parser.add_argument('--train_batch_size', 
                        type=int, 
                        required=True,
                        help='Batch size for training'
                        )
    parser.add_argument('--test_batch_size', 
                        type=int, 
                        required=True, 
                        help='Batch size for testing'
                        )
    parser.add_argument('--max_seq_length', 
                        type=int, 
                        required=True,
                        help='Max sequence length'
                        )
    parser.add_argument('--decoder_max_length', 
                        type=int, 
                        required=True,
                        help='Max length for decoder'
                        )
    parser.add_argument('--num_epochs', 
                        type=int, 
                        required=True,
                        help='Number of epochs'
                        )
    parser.add_argument('--seed', 
                        type=int, 
                        required=True,
                        help='Random seed'
                        )
    parser.add_argument('--cuda_visible_devices', 
                        type=str, 
                        required=True,
                        help='CUDA visible devices'
                        )
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True,
                        help='Name of the model'
                        )
    parser.add_argument('--model_path', 
                        type=str, 
                        required=True,
                        help='Path to the model'
                        )
    parser.add_argument('--template_path', 
                        type=str, 
                        required=True,
                        help='Path to the enhanced continuous template file'
                        )
    parser.add_argument('--lr_group_1', 
                        type=float, 
                        default=3e-5,
                        help='Learning rate for the first group of parameters'
                        )
    parser.add_argument('--lr_group_2', 
                        type=float, 
                        default=3e-4,
                        help='Learning rate for the second group of parameters'
                        )

    args = parser.parse_args()
    main(args)
