import os, pickle, random
import numpy as np
import torch
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassifierDataset(torch.utils.data.Dataset):

    def __init__(self, tokenized_sentences, labels):
        self.tokenized_sentences = tokenized_sentences
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_sentences.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def read_data(path, split, max_samples=None, lowercase=False):
    data, labels = [], []
    for style in ['0', '1']:
        full_path = path + split + f'.{style}'
        with open(full_path) as input_file:
            if max_samples is not None:
                data_tmp = input_file.read().split('\n')[:max_samples//2]
                labels_tmp = np.full(len(data_tmp), int(style))[:max_samples//2]
            else:
                data_tmp = input_file.read().split('\n')
                labels_tmp = np.full(len(data_tmp), int(style))
            if lowercase:
                data_tmp = [d.lower() for d in data_tmp]
            data.extend(data_tmp)
            labels.extend(labels_tmp)
    return data, labels


def classifier_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    metrics = {'acc': accuracy_score(labels, predictions),
               'prec': precision_score(labels, predictions, average='macro', zero_division=0),
               'rec': recall_score(labels, predictions, average='macro', zero_division=0),
               'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0)}
    return metrics


def compute_metric(predictions, references, metric_name, lang='en'):
    bleu = evaluate.load('sacrebleu')
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # predictions = lists | references = list of lists
    scores = []
    if metric_name in ['bleu', 'rouge', 'bertscore']:
        for pred, ref in zip(predictions, references):
            if metric_name == 'bleu':
                res = bleu.compute(predictions=[pred], references=[ref])
                scores.append(res['score'])
            elif metric_name == 'rouge':
                tmp_rouge1, tmp_rouge2, tmp_rougeL = [], [], []
                for r in ref:
                    res = rouge.compute(predictions=[pred], references=[r], use_aggregator=False)
                    tmp_rouge1.append(res['rouge1'][0].fmeasure)
                    tmp_rouge2.append(res['rouge2'][0].fmeasure)
                    tmp_rougeL.append(res['rougeL'][0].fmeasure)
                scores.append([max(tmp_rouge1), max(tmp_rouge2), max(tmp_rougeL)])
            elif metric_name == 'bertscore':
                res = bertscore.compute(predictions=[pred], references=[ref], lang=lang, device=device)
                scores.extend(res['f1'])
    else:
        raise Exception(f"Metric {metric_name} is not supported.")
    return scores

def compute_classif_metrics(pred_A, pred_B, pretrained_classifier_eval, batch_size=64, max_sequence_length=64, only_ab=False):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    truncation, padding = 'longest_first', 'max_length'
    classifier = AutoModelForSequenceClassification.from_pretrained(pretrained_classifier_eval)
    classifier_tokenizer = AutoTokenizer.from_pretrained(f'{pretrained_classifier_eval}tokenizer/')
    classifier.to(device)
    classifier.eval()

    if not only_ab:
        y_pred, y_true = [], np.concatenate((np.full(len(pred_A), 0), np.full(len(pred_B), 1)))
    else:
        y_pred, y_true = [], np.full(len(pred_B), 1)

    if not only_ab:
        for i in range(0, len(pred_A), batch_size):
            batch_a = pred_A[i:i+batch_size]
            inputs = classifier_tokenizer(batch_a, truncation=truncation, padding=padding, max_length=max_sequence_length, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                output = classifier(**inputs)
            y_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
    for i in range(0, len(pred_B), batch_size):
        batch_b = pred_B[i:i+batch_size]
        inputs = classifier_tokenizer(batch_b, truncation=truncation, padding=padding, max_length=max_sequence_length, return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():
            output = classifier(**inputs)
        y_pred.extend(np.argmax(output.logits.cpu().numpy(), axis=1))
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1

def eval_ext(metric_names, pred_A, pred_B, parallel_dl_evalAB, parallel_dl_evalBA, args=None, return_metrics=False):
    # add non-existing fields to args for compatibility
    if 'batch_size' not in args: args.batch_size = 64
    if 'max_sequence_length' not in args: args.max_sequence_length = 64
    if 'lowercase_out' not in args: args.lowercase_out = False
    if 'lowercase_ref' not in args: args.lowercase_ref = False
    if 'only_ab' not in args: args.only_ab = False
    if 'method' not in args: args.method = args.model_tag.replace('/', '_')
    if 'pred_base_path' not in args: args.pred_base_path = args.output_folder
    #####################

    scores_AB_bleu_ref, scores_BA_bleu_ref = [], []
    scores_AB_r1_ref, scores_BA_r1_ref, scores_AB_r2_ref, scores_BA_r2_ref, scores_AB_rL_ref, scores_BA_rL_ref = [], [], [], [], [], []
    scores_AB_bscore, scores_BA_bscore = [], []

    for i, batch in enumerate(parallel_dl_evalAB) if type(parallel_dl_evalAB) != list else zip(*[range(0, len(parallel_dl_evalAB), args.batch_size)]*2):
        pred_B_batch = pred_B[i*args.batch_size:i*args.batch_size+args.batch_size] if type(parallel_dl_evalAB) != list else pred_B[i:i+args.batch_size]
        references_b = list(batch[1]) if type(parallel_dl_evalAB) != list else parallel_dl_evalAB[i:i+args.batch_size]
        if args.lowercase_out:
            pred_B_batch = [out.lower() for out in pred_B_batch]
        if args.lowercase_ref:
            references_b = [[ref.lower() for ref in refs] for refs in references_b]
        if 'bleu' in metric_names:
            scores_AB_bleu_ref.extend(compute_metric(pred_B_batch, references_b, 'bleu'))
        if 'rouge' in metric_names:
            scores_rouge_ref = np.array(compute_metric(pred_B_batch, references_b, 'rouge'))
            scores_AB_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            scores_AB_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            scores_AB_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
        if 'bertscore' in metric_names:
            scores_AB_bscore.extend(compute_metric(pred_B_batch, references_b, 'bertscore'))
    if 'bleu' in metric_names: avg_AB_bleu_ref = np.mean(scores_AB_bleu_ref)
    else: avg_AB_bleu_ref = -100
    if 'rouge' in metric_names: avg_AB_r1_ref, avg_AB_r2_ref, avg_AB_rL_ref = np.mean(scores_AB_r1_ref), np.mean(scores_AB_r2_ref), np.mean(scores_AB_rL_ref)
    else: avg_AB_r1_ref, avg_AB_r2_ref, avg_AB_rL_ref = -100, -100, -100
    if 'bertscore' in metric_names: avg_AB_bscore = np.mean(scores_AB_bscore)
    else: avg_AB_bscore = -100

    for i, batch in enumerate(parallel_dl_evalBA) if type(parallel_dl_evalBA) != list else zip(*[range(0, len(parallel_dl_evalBA), args.batch_size)]*2):
        pred_A_batch = pred_A[i*args.batch_size:i*args.batch_size+args.batch_size] if type(parallel_dl_evalBA) != list else pred_A[i:i+args.batch_size]
        references_a = list(batch[1]) if type(parallel_dl_evalBA) != list else parallel_dl_evalBA[i:i+args.batch_size]
        if args.lowercase_out:
            pred_A_batch = [out.lower() for out in pred_A_batch]
        if args.lowercase_ref:
            references_a = [[ref.lower() for ref in refs] for refs in references_a]
        if 'bleu' in metric_names:
            scores_BA_bleu_ref.extend(compute_metric(pred_A_batch, references_a, 'bleu'))
        if 'rouge' in metric_names:
            scores_rouge_ref = np.array(compute_metric(pred_A_batch, references_a, 'rouge'))
            scores_BA_r1_ref.extend(scores_rouge_ref[:, 0].tolist())
            scores_BA_r2_ref.extend(scores_rouge_ref[:, 1].tolist())
            scores_BA_rL_ref.extend(scores_rouge_ref[:, 2].tolist())
        if 'bertscore' in metric_names:
            scores_BA_bscore.extend(compute_metric(pred_A_batch, references_a, 'bertscore'))
    if 'bleu' in metric_names: avg_BA_bleu_ref = np.mean(scores_BA_bleu_ref)
    else: avg_BA_bleu_ref = -100
    if 'rouge' in metric_names: avg_BA_r1_ref, avg_BA_r2_ref, avg_BA_rL_ref = np.mean(scores_BA_r1_ref), np.mean(scores_BA_r2_ref), np.mean(scores_BA_rL_ref)
    else: avg_BA_r1_ref, avg_BA_r2_ref, avg_BA_rL_ref = -100, -100, -100
    if 'bertscore' in metric_names: avg_BA_bscore = np.mean(scores_BA_bscore)
    else: avg_BA_bscore = -100
    if not args.only_ab:
        avg_2dir_bleu_ref = (avg_AB_bleu_ref + avg_BA_bleu_ref) / 2
    else:
        avg_2dir_bleu_ref = -100

    metrics = {'method':args.method,
               'ref-BLEU A->B':avg_AB_bleu_ref, 'ref-BLEU B->A':avg_BA_bleu_ref,
               'ref-BLEU avg':avg_2dir_bleu_ref,
               'ref-ROUGE-1 A->B':avg_AB_r1_ref, 'ref-ROUGE-1 B->A':avg_BA_r1_ref,
               'ref-ROUGE-2 A->B':avg_AB_r2_ref, 'ref-ROUGE-2 B->A':avg_BA_r2_ref,
               'ref-ROUGE-L A->B':avg_AB_rL_ref, 'ref-ROUGE-L B->A':avg_BA_rL_ref,
               'BERTScore A->B':avg_AB_bscore, 'BERTScore B->A':avg_BA_bscore,
               'only A->B':args.only_ab}

    if 'acc' in metric_names:
        if args.lowercase_out:
            pred_A = [out.lower() for out in pred_A]
            pred_B = [out.lower() for out in pred_B]
        acc, prec, rec, f1 = compute_classif_metrics(pred_A, pred_B, args.pretrained_classifier_eval, args.batch_size, args.max_sequence_length, args.only_ab)
        metrics['style accuracy'] = acc
        metrics['style precision'] = prec
        metrics['style recall'] = rec
        metrics['style F1 score'] = f1
    else:
        acc = 0
    if not args.only_ab:
        g_bleu_acc = (avg_2dir_bleu_ref*acc*100)**0.5
        h_bleu_acc = 2*avg_2dir_bleu_ref*acc*100/(avg_2dir_bleu_ref + acc*100)
        metrics['g-BLEU-acc'] = g_bleu_acc
        metrics['h-BLEU-acc'] = h_bleu_acc
    else:
        g_bleu_acc = (avg_AB_bleu_ref*acc*100)**0.5
        h_bleu_acc = 2*avg_AB_bleu_ref*acc*100/(avg_AB_bleu_ref + acc*100)
        metrics['g-BLEU-acc'] = g_bleu_acc
        metrics['h-BLEU-acc'] = h_bleu_acc

    base_path = args.pred_base_path
    suffix = f'{args.method}_test'
    pickle.dump(metrics, open(f"{base_path}metrics_{suffix}.pickle", 'wb'))

    print()
    for m, v in metrics.items():
        if v != -100:
            print(f'{m}: {v}')
    if return_metrics: return metrics


def create_mixed_style_data(src_A, src_B, ref_path, pattern, direction, ratioAB):
    # input = <A, B>, ref1 = <r1A, r1B>, ref2 = <r2A, r2B>, ref3 = <r3A, r3B>, ref4 = <r4A, r4B>
    # 0-100 ~ AB: <B> - BA: it is traditional TST | 100-0 ~ AB: it is traditional TST - BA: <A>
    # 50-50 ~ AB: <A,B> - BA: <B,A>
    # 66-33 ~ AB: <A,B,A> - BA: <A,B,A> | 33-66 ~ AB: <B,A,B> - BA: <B,A,B>
    # 75-25 ~ AB: <A,B,A,A> - BA: <A,B,A,A> | 25-75 ~ AB: <B,A,B,B> - BA: <B,A,B,B>
    src_A_texts, src_B_texts = [line.strip() for line in open(src_A, 'r').readlines()], [line.strip() for line in open(src_B, 'r').readlines()]
    r1AB, r2AB, r3AB, r4AB = [line.strip() for line in open(ref_path+f'reference0.0.txt', 'r').readlines()], [line.strip() for line in open(ref_path+f'reference1.0.txt', 'r').readlines()], [line.strip() for line in open(ref_path+f'reference2.0.txt', 'r').readlines()], [line.strip() for line in open(ref_path+f'reference3.0.txt', 'r').readlines()]
    r1BA, r2BA, r3BA, r4BA = [line.strip() for line in open(ref_path+f'reference0.1.txt', 'r').readlines()], [line.strip() for line in open(ref_path+f'reference1.1.txt', 'r').readlines()], [line.strip() for line in open(ref_path+f'reference2.1.txt', 'r').readlines()], [line.strip() for line in open(ref_path+f'reference3.1.txt', 'r').readlines()]
    assert len(src_A_texts) == len(r1AB) == len(r2AB) == len(r3AB) == len(r4AB) and len(src_B_texts) == len(r1BA) == len(r2BA) == len(r3BA) == len(r4BA)
    refs_AB, refs_BA = [[r1, r2, r3, r4] for r1, r2, r3, r4 in zip(r1AB, r2AB, r3AB, r4AB)], [[r1, r2, r3, r4] for r1, r2, r3, r4 in zip(r1BA, r2BA, r3BA, r4BA)]
    tot_A, tot_B = len(src_A_texts), len(src_B_texts)
    mixed_src, mixed_refs = [], []
    to_extract_A, to_extract_B = pattern.count('A')-int(direction=='AB'), pattern.count('B')-int(direction=='BA')
    for i, (src, tgt) in enumerate(zip(src_A_texts, refs_AB)) if direction=='AB' else enumerate(zip(src_B_texts, refs_BA)):
        mixed_sentence = []
        r1, r2, r3, r4 = [], [], [], []
        candidates_a, candidates_b = list(range(tot_A)), list(range(tot_B))
        if i in candidates_a: candidates_a.remove(i)
        if i in candidates_b: candidates_b.remove(i)
        ix_a = random.sample(candidates_a, to_extract_A) if to_extract_A > 0 else -1
        ix_b = random.sample(candidates_b, to_extract_B) if to_extract_B > 0 else -1
        for el in pattern.split(','):
            if el == 'A':
                if direction == 'AB' and src not in mixed_sentence:
                    mixed_sentence.append(src)
                    r1.append(tgt[0]); r2.append(tgt[1]); r3.append(tgt[2]); r4.append(tgt[3])
                else:
                    mixed_sentence.append(src_A_texts[ix_a[-1]])
                    if direction == 'AB':
                        r1.append(refs_AB[ix_a[-1]][0]); r2.append(refs_AB[ix_a[-1]][1]); r3.append(refs_AB[ix_a[-1]][2]); r4.append(refs_AB[ix_a[-1]][3])
                    else:
                        r1.append(src_A_texts[ix_a[-1]]); r2.append(src_A_texts[ix_a[-1]]); r3.append(src_A_texts[ix_a[-1]]); r4.append(src_A_texts[ix_a[-1]])
                    ix_a.pop()
            else:
                if direction == 'BA' and src not in mixed_sentence:
                    mixed_sentence.append(src)
                    r1.append(tgt[0]); r2.append(tgt[1]); r3.append(tgt[2]); r4.append(tgt[3])
                else:
                    mixed_sentence.append(src_B_texts[ix_b[-1]])
                    if direction == 'BA':
                        r1.append(refs_BA[ix_b[-1]][0]); r2.append(refs_BA[ix_b[-1]][1]); r3.append(refs_BA[ix_b[-1]][2]); r4.append(refs_BA[ix_b[-1]][3])
                    else:
                        r1.append(src_B_texts[ix_b[-1]]); r2.append(src_B_texts[ix_b[-1]]); r3.append(src_B_texts[ix_b[-1]]); r4.append(src_B_texts[ix_b[-1]])
                    ix_b.pop()
        mixed_src.append(' '.join(mixed_sentence))
        mixed_refs.append([' '.join(r1), ' '.join(r2), ' '.join(r3), ' '.join(r4)])
    all_r1, all_r2, all_r3, all_r4 = list(np.array(mixed_refs)[:, 0]), list(np.array(mixed_refs)[:, 1]), list(np.array(mixed_refs)[:, 2]), list(np.array(mixed_refs)[:, 3])
    assert len(mixed_src) == len(mixed_refs) and len(all_r1) == len(all_r2) == len(all_r3) == len(all_r4)
    out_path = f'/content/{ratioAB}/'
    if not os.path.exists(out_path): os.makedirs(out_path)
    with open(f"{out_path}/test.{0 if direction == 'AB' else 1}", 'w') as f:
        f.writelines('\n'.join(mixed_src))
    for i, refs in enumerate([all_r1, all_r2, all_r3, all_r4]):
        with open(f"{out_path}/reference{i}.{0 if direction == 'AB' else 1}.txt", 'w') as f:
            f.writelines('\n'.join(refs))
