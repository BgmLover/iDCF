import bottleneck as bn
import numpy as np
import random
import os
import ray
import json
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_index(data_size, split_ratio, random_split, seed=1234):
    all_user_id = np.arange(data_size)
    np.random.seed(seed)
    if random_split:
        np.random.shuffle(all_user_id)
    validation_index = all_user_id[:int(data_size * split_ratio)]
    test_index = all_user_id[int(data_size * split_ratio):]
    return validation_index, test_index


def split_by_item(df, ratio, seed=1234):
    data_group_by_user = df.groupby("user_id")
    train_list, test_list = list(), list()
    np.random.seed(seed)
    for i, (_, group) in enumerate(data_group_by_user):
        n_items = len(group)
        sampled_idx = np.zeros(n_items, dtype="bool")
        sampled_idx[np.random.choice(n_items, size=int(ratio * n_items), replace=False)] = True
        train_list.append(group[np.logical_not(sampled_idx)])
        test_list.append(group[sampled_idx])
    data_train = pd.concat(train_list)
    data_test = pd.concat(test_list)
    return data_train, data_test


def split_by_user(df, ratio, seed=1234):
    np.random.seed(seed)
    unique_uids = df["user_id"].unique()
    test_users = np.random.choice(unique_uids, size=int(unique_uids.size * ratio), replace=False)
    val_users = np.setdiff1d(unique_uids, test_users)
    df_val = df.loc[df["user_id"].isin(val_users)]
    df_test = df.loc[df["user_id"].isin(test_users)]
    return df_val, df_test, val_users, test_users


def df_to_csr(df, shape):
    rows = df["user_id"]
    cols = df["item_id"]
    values = df["rating"]
    mat = csr_matrix((values, (rows, cols)))
    # mat = mat[mat.getnnz(axis=1) > 0]
    # assert mat.shape == shape
    return mat


def np_to_csr(array):
    rows = array[:, 0].astype(int)
    cols = array[:, 1].astype(int)
    values = array[:, 2]
    mat = csr_matrix((values, (rows, cols)))
    return mat


def construct_rating_dataset(train_df_path, random_df_path, test_ratio, split_index=False):
    train_df = pd.read_csv(train_df_path)
    # train_df = train_df.loc[train_df["user_id"] < 5400]
    random_df = pd.read_csv(random_df_path)

    # val_df, test_df = split_by_item(random_df, validation_ratio)
    val_df, test_df, val_users, test_users = split_by_user(random_df, test_ratio)
    if split_index:
        return train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy(), val_users, test_users
    else:
        return train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()


def construct_vae_dataset(df_path, train_ratio, split_test=False, test_test_ratio=0.5, seed=1234):
    df = pd.read_csv(df_path)
    unique_users = df["user_id"].unique()

    n_users = unique_users.shape[0]
    n_items = df["item_id"].max() + 1
    if train_ratio == 1:
        return df_to_csr(df, shape=(n_users, n_items)).toarray()
    n_train_users = int(train_ratio * n_users)

    np.random.seed(seed)
    train_user_index = np.random.choice(unique_users, size=n_train_users, replace=False)
    train_user_index = np.sort(train_user_index)
    test_user_index = np.setdiff1d(unique_users, train_user_index)

    if split_test:
        pass
        # index = df["user_id"].isin(train_user_index)
        # train_df = df.loc[index]
        # test_df = df.loc[~index]
        # test_train, test_test = split_by_item(test_df, test_test_ratio)
        # train_matrix = df_to_csr(train_df, shape=(n_train_users, n_items))
        # test_tr_matrix = df_to_csr(test_train, shape=(n_users - n_train_users, n_items))
        # test_te_matrix = df_to_csr(test_test, shape=(n_users - n_train_users, n_items))
        # return train_matrix.toarray(), test_tr_matrix.toarray(), test_te_matrix.toarray(), train_user_index
    else:
        matrix = df_to_csr(df, (n_users, n_items))
        train_matrix = matrix[train_user_index]
        test_matrix = matrix[test_user_index]
        return train_matrix.toarray(), test_matrix.toarray(), train_user_index, test_user_index


def load_coat_by_ui_pair(path="data_process/coat/", validation_ratio=0.3):
    train_data_raw = pd.read_table(path + "train.ascii").to_numpy()
    test_data_raw = pd.read_table(path + "test.ascii").to_numpy()
    user_feature = pd.read_table(path + "user_item_features/user_features.ascii", sep=" ", header=None).to_numpy()

    val_data = np.zeros_like(test_data_raw)
    test_data = np.zeros_like(test_data_raw)
    for i, row in enumerate(test_data_raw):
        nonzero_items = row.nonzero()[0]
        val_iid = np.random.choice(nonzero_items, size=int(len(nonzero_items) * validation_ratio), replace=False)
        test_iid = np.setdiff1d(nonzero_items, val_iid)
        val_data[i][val_iid] = test_data_raw[val_iid]
        test_data[i][test_iid] = test_data_raw[test_iid]
    train_matrix = csr_matrix(train_data_raw)
    val_matrix = csr_matrix(val_data)
    test_matrix = csr_matrix(test_data)
    return train_matrix, val_matrix, test_matrix, user_feature


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)])
    valid_index = np.nonzero(IDCG)
    return DCG[valid_index] / IDCG[valid_index]


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = heldout_batch > 0
    hit = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    total_size = X_true_binary.sum(axis=1)
    valid_index = np.nonzero(total_size)
    recall = hit[valid_index] / total_size[valid_index]
    return recall


@ray.remote
def NDCG_RECALL_at_k_batch_parallel(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in np.count_nonzero(heldout_batch, axis=1)])
    valid_index = np.nonzero(IDCG)
    ndcg = DCG[valid_index] / IDCG[valid_index]

    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]] = True

    X_true_binary = heldout_batch > 0
    hit = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    total_size = X_true_binary.sum(axis=1)
    valid_index = np.nonzero(total_size)
    recall = hit[valid_index] / total_size[valid_index]
    return np.concatenate((ndcg.reshape(-1, 1), recall.reshape(-1, 1)), axis=1)


def cal_ndcg_recall_parallel(num_workers, X_pred, heldout_batch, k=100):
    prediction = X_pred
    labels = heldout_batch
    lens = X_pred.shape[0]
    piece_lens = int(lens / num_workers)
    task = []
    rounds = num_workers if lens % num_workers == 0 else num_workers + 1
    for i in range(rounds):
        start = i * piece_lens
        end = min((i + 1) * piece_lens, lens)
        x = prediction[start:end]
        y = labels[start:end]
        task.append(NDCG_RECALL_at_k_batch_parallel.remote(x, y, k))
    res = ray.get(task)
    return np.concatenate(res, axis=0)


def mf_evaluate(metric, data_loader, test_model, device="cpu", params=None):
    test_model.eval()
    with torch.no_grad():
        if metric == "mse":
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = test_model.predict(uid, iid)
                predict = params["min_val"] + predict * (params["max_val"] - params["min_val"])
                labels.extend(rating.tolist())
                predicts.extend(predict.tolist())
            mse = mean_squared_error(predicts, labels)
            return mse
        elif metric == "ndcg":
            uids, iids, predicts, labels = list(), list(), list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(device), iid.to(device), rating.to(device)
                predict = test_model.predict(uid, iid)
                uids.extend(uid.cpu())
                iids.extend(iid.cpu())
                predicts.extend(predict.cpu())
                labels.extend(rating.cpu())
            label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids))))
            label_matrix.eliminate_zeros()
            valid_rows = np.unique(label_matrix.nonzero()[0])
            label_matrix = label_matrix[valid_rows].toarray()

            predict_matrix = csr_matrix((np.array(predicts), (np.array(uids), np.array(iids))))
            predict_matrix = predict_matrix[valid_rows]
            predict_matrix.data += 1 << 10
            predict_matrix = predict_matrix.toarray()

            # ndcg = NDCG_binary_at_k_batch(predict_matrix1, label_matrix1, k=params["k"]).mean()
            # recall = Recall_at_k_batch(predict_matrix1, label_matrix1, k=params["k"]).mean()
            if device == "cpu":
                ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=params["k"]).mean()
                recall = Recall_at_k_batch(predict_matrix, label_matrix, k=params["k"]).mean()
                return ndcg, recall
            else:
                res = cal_ndcg_recall_parallel(2, predict_matrix, label_matrix, params["k"]).mean(axis=0)
                return res[0], res[1]


class MFRatingDataset(Dataset):
    def __init__(self, uid, iid, rating, require_index=False):
        self.uid = uid
        self.iid = iid
        self.rating = rating
        self.index = None
        if require_index:
            self.index = np.arange(0, self.uid.shape[0])

    def __getitem__(self, index):
        if self.index is None:
            return self.uid[index], self.iid[index], self.rating[index]
        else:
            return self.uid[index], self.iid[index], self.rating[index], self.index[index]

    def __len__(self):
        return len(self.rating)


def construct_mf_dataloader(config, device, require_index=False):
    data_params = config["data_params"]

    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    min_val, max_val = data_params["min_val"], data_params["max_val"]
    threshold = data_params["threshold"]

    if config["metric"] == "mse":
        train_ratings = ((train_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
        evaluation_params = {
            "min_val": min_val,
            "max_val": max_val,
            "n_items": n_items
        }
    else:
        train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
        val_mat[:, 2] = val_mat[:, 2] >= threshold
        test_mat[:, 2] = test_mat[:, 2] >= threshold

        evaluation_params = {
            "k": config["topk"]
        }
    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"],
                                                           require_index=require_index)
    return train_loader, val_loader, test_loader, evaluation_params, n_users, n_items


def get_dataloader(train_mat, train_ratings, val_mat, test_mat, batch_size, require_index=False, num_workers=5,
                   pin_memory=True):
    train_dataset = MFRatingDataset(train_mat[:, 0].astype(int),
                                    train_mat[:, 1].astype(int),
                                    train_ratings,
                                    require_index)
    val_dataset = MFRatingDataset(val_mat[:, 0].astype(int),
                                  val_mat[:, 1].astype(int),
                                  val_mat[:, 2])
    test_dataset = MFRatingDataset(test_mat[:, 0].astype(int),
                                   test_mat[:, 1].astype(int),
                                   test_mat[:, 2])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def load_uniform_data_from_np(ratio, array, shape):
    size = int(ratio * array.shape[0])
    index = np.random.permutation(np.arange(array.shape[0])[:size])
    rows, cols, rating = array[index, 0], array[index, 1], array[index, 2]
    return csr_matrix(
        (rating, (rows, cols)), shape=shape
    ), index


def construct_ips_dataloader(config, device):
    data_params = config["data_params"]
    train_mat, val_mat, test_mat = construct_rating_dataset(data_params["train_path"],
                                                            data_params["random_path"],
                                                            test_ratio=data_params["test_ratio"])
    n_users = train_mat[:, 0].astype(int).max() + 1
    n_items = train_mat[:, 1].astype(int).max() + 1

    min_val, max_val = data_params["min_val"], data_params["max_val"]
    threshold = data_params["threshold"]

    if config["metric"] == "mse":
        train_ratings = ((train_mat[:, 2] - min_val) / (max_val - min_val)).astype(np.float32)
        evaluation_params = {
            "min_val": min_val,
            "max_val": max_val,
            "n_items": n_items
        }
    else:
        train_ratings = (train_mat[:, 2] >= threshold).astype(np.float32)
        val_mat[:, 2] = val_mat[:, 2] >= threshold
        test_mat[:, 2] = test_mat[:, 2] >= threshold

        evaluation_params = {
            "k": config["topk"]
        }
    uniform_data, index = load_uniform_data_from_np(0.166, val_mat, shape=(n_users, n_items))
    val_mat = np.delete(val_mat, index, axis=0)

    train_loader, val_loader, test_loader = get_dataloader(train_mat,
                                                           train_ratings,
                                                           val_mat,
                                                           test_mat,
                                                           config["batch_size"])

    def Naive_Bayes_Propensity(train, unif):
        # follow [1] Jiawei Chen et, al, AutoDebias: Learning to Debias for Recommendation 2021SIGIR and
        # [2] Tobias Schnabel, et, al, Recommendations as Treatments: Debiasing Learning and Evaluation
        P_Oeq1 = train.getnnz() / (train.shape[0] * train.shape[1])
        train.data[train.data < threshold] = 0
        train.data[train.data >= threshold] = 1
        # unif.data[unif.data < threshold] = 0
        # unif.data[unif.data > threshold] = 1

        y_unique = np.unique(train.data)
        P_y_givenO = np.zeros(y_unique.shape)
        P_y = np.zeros(y_unique.shape)

        for i in range(len(y_unique)):
            P_y_givenO[i] = np.sum(train.data == y_unique[i]) / np.sum(
                np.ones(train.data.shape))
            P_y[i] = np.sum(unif.data == y_unique[i]) / np.sum(np.ones(unif.data.shape))
        Propensity = P_y_givenO * P_Oeq1 / P_y
        Propensity = Propensity * (np.ones((n_items, 2)))

        return y_unique, Propensity

    y_unique, Propensity = Naive_Bayes_Propensity(np_to_csr(train_mat), uniform_data)
    InvP = torch.reciprocal(torch.tensor(Propensity, dtype=torch.float)).to(device)

    return train_loader, val_loader, test_loader, evaluation_params, n_users, n_items, y_unique, InvP


def read_best_params(model, key_name, sr=0.1, cr=2.0, tr=0.0):
    dir_prefix = os.getcwd()
    file_path = "/res/ndcg/sim_{}.json".format(key_name)
    if key_name == "sr":
        key = sr
    elif key_name == "cr":
        key = cr
    else:
        key = tr
    with open(dir_prefix + file_path, "r") as f:
        config = json.load(f)
        for model_config in config["models"]:
            if model == model_config["name"]:
                for param in model_config["params"]:
                    if param[key_name] == key:
                        return param
    raise Exception("invalid ")
