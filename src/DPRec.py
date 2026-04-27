import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from model import DPRec
import logging
import copy
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import ttest_1samp


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def train(args, data_info):
    logging.info("================== training====================")
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    user_triple_set = data_info[5]
    item_triple_set = data_info[6]
    user_potential_triple_set = data_info[7]
    item_origin_triple_set = data_info[8]
    user_entity_freq = data_info[9]

    model, optimizer, loss_func = _init_model(args, data_info)

    best_eval_auc = -1.0
    best_result = {
        'best_epoch': -1,
        'best_eval_auc': 0.0,
        'best_eval_f1': 0.0,
        'best_test_auc': 0.0,
        'best_test_f1': 0.0,
    }
    if getattr(args, 'eval_topk', False):
        best_result.update({
            'best_hit@1': 0.0,
            'best_ndcg@1': 0.0,
            'best_hit@5': 0.0,
            'best_ndcg@5': 0.0,
            'best_hit@10': 0.0,
            'best_ndcg@10': 0.0,
        })

    for step in range(args.n_epoch):
        np.random.shuffle(train_data)
        start = 0
        model.train()

        while start < train_data.shape[0]:
            labels = _get_feed_label(args, train_data[start:start + args.batch_size, 2])

            scores, CL_loss = model(
                *_get_feed_data(
                    args, train_data, user_triple_set, item_triple_set,
                    user_potential_triple_set, item_origin_triple_set,
                    start, start + args.batch_size, user_entity_freq
                )
            )

            loss = loss_func(scores, labels) + CL_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += args.batch_size

        eval_auc, eval_f1 = ctr_eval(
            args, model, eval_data, user_triple_set, item_triple_set,
            user_potential_triple_set, item_origin_triple_set, user_entity_freq
        )
        test_auc, test_f1 = ctr_eval(
            args, model, test_data, user_triple_set, item_triple_set,
            user_potential_triple_set, item_origin_triple_set, user_entity_freq
        )

        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f'
        logging.info(ctr_info, step, eval_auc, eval_f1, test_auc, test_f1)

        if getattr(args, 'eval_topk', False):
            train_record = _get_user_record(args, train_data, True)
            test_record = _get_user_record(args, test_data, False)
            item_set = set(train_data[:, 1].tolist() + test_data[:, 1].tolist())

            k_list = [1, 5, 10]
            hit, ndcg = topk_eval(
                args=args,
                model=model,
                test_record=test_record,
                item_set=item_set,
                k_list=k_list,
                batch_size=args.batch_size,
                device=torch.device('cuda' if args.use_cuda else 'cpu'),
                user_triple_set=user_triple_set,
                item_triple_set=item_triple_set,
                user_potential_triple_set=user_potential_triple_set,
                item_origin_triple_set=item_origin_triple_set,
                user_entity_freq=user_entity_freq,
                user_sample_size=1000,
                n_items=50
            )

            for k, h, n in zip(k_list, hit, ndcg):
                logging.info(f"[Evaluation] Hit@{k}: {h:.4f}   NDCG@{k}: {n:.4f}")
        else:
            hit, ndcg = None, None

        if eval_auc > best_eval_auc:
            best_eval_auc = eval_auc
            torch.save(model.state_dict(), 'best_model.pth')
            best_result = {
                'best_epoch': step,
                'best_eval_auc': float(eval_auc),
                'best_eval_f1': float(eval_f1),
                'best_test_auc': float(test_auc),
                'best_test_f1': float(test_f1),
            }
            if getattr(args, 'eval_topk', False):
                best_result.update({
                    'best_hit@1': float(hit[0]),
                    'best_ndcg@1': float(ndcg[0]),
                    'best_hit@5': float(hit[1]),
                    'best_ndcg@5': float(ndcg[1]),
                    'best_hit@10': float(hit[2]),
                    'best_ndcg@10': float(ndcg[2]),
                })


    if getattr(args, 'eval_topk', False):
        logging.info(
            "Best top-k | Hit@1: %.4f NDCG@1: %.4f | Hit@5: %.4f NDCG@5: %.4f | Hit@10: %.4f NDCG@10: %.4f",
            best_result['best_hit@1'], best_result['best_ndcg@1'],
            best_result['best_hit@5'], best_result['best_ndcg@5'],
            best_result['best_hit@10'], best_result['best_ndcg@10'])

    return best_result

def ctr_eval(args, model, data, user_triple_set, item_triple_set, user_potential_triple_set, item_origin_triple_set, user_entity_freq):
    model.eval()
    all_labels = []
    all_scores = []

    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        scores, _ = model(
            *_get_feed_data(
                args, data, user_triple_set, item_triple_set,
                user_potential_triple_set, item_origin_triple_set,
                start, start + args.batch_size, user_entity_freq
            )
        )
        scores = scores.detach().cpu().numpy()

        all_labels.extend(labels.tolist())
        all_scores.extend(scores.tolist())
        start += args.batch_size

    predictions = [1 if s >= 0.5 else 0 for s in all_scores]
    auc = roc_auc_score(y_true=all_labels, y_score=all_scores)
    f1 = f1_score(y_true=all_labels, y_pred=predictions)

    model.train()
    return float(auc), float(f1)


def _init_model(args, data_info):
    n_entity = data_info[3]
    n_relation = data_info[4]
    model = DPRec(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func


def _get_feed_data(args, data, user_triple_set, item_triple_set, user_potential_triple_set, item_origin_triple_set,
                   start, end,user_entity_freq):
    # origin item
    items = torch.LongTensor(data[start:end, 1])
    if args.use_cuda:
        items = items.cuda()
    # kg propagation embeddings
    users_triple = _get_triple_tensor(args, data[start:end, 0], user_triple_set)
    items_triple = _get_triple_tensor(args, data[start:end, 1], item_triple_set)
    user_potential_triple = _get_triple_tensor(args, data[start:end, 0], user_potential_triple_set)
    item_origin_triple = _get_triple_tensor(args, data[start:end, 1], item_origin_triple_set)
    user_ids = data[start:end, 0]
    user_freq_tensor = _get_user_freq_tensor(user_ids, users_triple, user_entity_freq)
    if args.use_cuda:
        user_freq_tensor = user_freq_tensor.cuda()
    return items, users_triple, items_triple, user_potential_triple, item_origin_triple,user_freq_tensor


def _get_feed_label(args, labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels


def _get_triple_tensor(args, objs, triple_set):
    h, r, t = [], [], []
    for i in range(args.n_layer):
        h_i = np.array([triple_set[obj][i][0] for obj in objs])
        r_i = np.array([triple_set[obj][i][1] for obj in objs])
        t_i = np.array([triple_set[obj][i][2] for obj in objs])

        h_tensor = torch.from_numpy(h_i).long()
        r_tensor = torch.from_numpy(r_i).long()
        t_tensor = torch.from_numpy(t_i).long()

        if args.use_cuda:
            h_tensor = h_tensor.cuda()
            r_tensor = r_tensor.cuda()
            t_tensor = t_tensor.cuda()

        h.append(h_tensor)
        r.append(r_tensor)
        t.append(t_tensor)

    return [h, r, t]



def _get_user_record(args, data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user, item])
    return np.array(res)


def _show_recall_info(recall_zip):
    res = ""
    for i, j in recall_zip:
        res += "K@%d:%.4f  " % (i, j)
    logging.info(res)



def _get_user_freq_tensor(user_ids, triple_tensor, user_entity_freq):
    B, T = triple_tensor[0][0].shape
    freq_tensor = np.zeros((B, T), dtype=np.float32)
    for i in range(B):
        uid = user_ids[i]
        entities = triple_tensor[0][0][i].tolist()
        freqs = np.array([user_entity_freq[uid].get(e, 1.0) for e in entities])
        freqs = freqs / (freqs.sum() + 1e-8)
        freq_tensor[i] = freqs
    return torch.from_numpy(freq_tensor).cuda()
def topk_eval(
    args, model, test_record, item_set, k_list, batch_size, device,
    user_triple_set, item_triple_set, user_potential_triple_set, item_origin_triple_set, user_entity_freq,
    user_sample_size=1000, n_items=50
):

    import random
    import numpy as np
    import torch

    hit_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}
    model.eval()

    user_list = list(test_record.keys())
    if len(user_list) > user_sample_size:
        user_list = random.sample(user_list, user_sample_size)

    with torch.no_grad():
        for user in user_list:
            gt_items = list(test_record[user])
            if len(gt_items) == 0:
                continue
            gt_item = random.choice(gt_items)


            neg_items = list(item_set - test_record[user])
            if len(neg_items) < n_items - 1:
                continue
            sampled_neg_items = random.sample(neg_items, n_items - 1)

            eval_items = [gt_item] + sampled_neg_items
            random.shuffle(eval_items)

            item_score_map = dict()
            for i in range(0, len(eval_items), batch_size):
                items_batch = eval_items[i:i + batch_size]
                if len(items_batch) < batch_size:
                    items_batch += [items_batch[-1]] * (batch_size - len(items_batch))


                input_data = _get_topk_feed_data(user, items_batch)  # shape: [B, 2]
                scores, _ = model(
                    *_get_feed_data(
                        args, input_data,
                        user_triple_set, item_triple_set,
                        user_potential_triple_set, item_origin_triple_set,
                        0, batch_size, user_entity_freq
                    )
                )
                scores = scores.detach().cpu().numpy().tolist()
                for it, sc in zip(items_batch, scores):
                    item_score_map[it] = float(sc)

            ranked_items = [x[0] for x in sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)]

            for k in k_list:
                topk_items = ranked_items[:k]
                hit = 1.0 if gt_item in topk_items else 0.0
                hit_list[k].append(hit)

                if gt_item in topk_items:
                    idx = topk_items.index(gt_item)
                    ndcg = 1 / np.log2(idx + 2)
                else:
                    ndcg = 0.0
                ndcg_list[k].append(ndcg)

    hit_result = [float(np.mean(hit_list[k])) if len(hit_list[k]) else 0.0 for k in k_list]
    ndcg_result = [float(np.mean(ndcg_list[k])) if len(ndcg_list[k]) else 0.0 for k in k_list]
    model.train()
    return hit_result, ndcg_result

