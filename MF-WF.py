from torch import nn
from utils import *
from ray.air import session
from argparser import *
from tune_script import *
from evaluator import Evaluator, mf_evaluate
from seeds import test_seeds
from models.mf import MFwithFeature

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(metric, data_loader, test_model, user_feature, device="cpu", params=None):
    test_model.eval()
    with torch.no_grad():
        if metric == "mse":
            labels, predicts = list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(DEVICE), iid.to(DEVICE), rating.float().to(DEVICE)
                u_feat = user_feature[uid]
                predict = test_model(uid, iid, u_feat).view(-1)
                predict = params["min_val"] + predict * (params["max_val"] - params["min_val"])
                labels.extend(rating.tolist())
                predicts.extend(predict.tolist())
            mse = mean_squared_error(predicts, labels)
            return mse
        elif metric == "ndcg":
            uids, iids, predicts, labels = list(), list(), list(), list()
            for index, (uid, iid, rating) in enumerate(data_loader):
                uid, iid, rating = uid.to(DEVICE), iid.to(DEVICE), rating.float().to(DEVICE)
                u_feat = user_feature[uid]
                predict = test_model(uid, iid, u_feat).view(-1)
                uids.extend(uid.cpu())
                iids.extend(iid.cpu())
                predicts.extend(predict.cpu())
                labels.extend(rating.cpu())
            predict_matrix = -np.inf * np.ones((max(uids) + 1, max(iids) + 1))
            predict_matrix[uids, iids] = predicts
            label_matrix = csr_matrix((np.array(labels), (np.array(uids), np.array(iids)))).toarray()
            ndcg = NDCG_binary_at_k_batch(predict_matrix, label_matrix, k=params["k"]).mean().item()
            recall = Recall_at_k_batch(predict_matrix, label_matrix, k=params["k"]).mean()
            return ndcg, recall


def train_eval(config):
    metric = config["metric"]
    data_params = config["data_params"]
    train_loader, val_loader, test_loader, evaluation_params, n_users, n_items = construct_mf_dataloader(config, DEVICE)

    user_feat = torch.tensor(pd.read_csv(data_params["user_feature_label"]).to_numpy()).to(DEVICE)
    seed_everything(config["seed"])

    model = MFwithFeature(n_users, n_items, args.data_params["user_feature_dim"], config["embedding_dim"], device=DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr_rate"], weight_decay=config["weight_decay"])
    if metric == "mse":
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.MSELoss()

    evaluator = Evaluator(metric, patience_max=config["patience"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_len = 0
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(DEVICE), iid.to(DEVICE), rating.float().to(DEVICE)
            u_feat = user_feat[uid]

            predict = model(uid, iid, u_feat).view(-1)

            loss = loss_func(predict, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(rating)
            total_len += len(rating)

        evaluator.record_training(total_loss / total_len)

        model.eval()
        validation_performance = evaluate(metric, val_loader, model, user_feat, device=DEVICE,
                                          params=evaluation_params)
        early_stop = evaluator.record_val(validation_performance, model.state_dict())
        if not config["tune"]:
            test_performance = evaluate(metric, test_loader, model, user_feat, device=DEVICE, params=evaluation_params)
            evaluator.record_test(test_performance)
        if config["show_log"]:
            evaluator.epoch_log(epoch)

        if early_stop:
            if config["show_log"]:
                print("reach max patience {}, current epoch {}".format(evaluator.patience_max, epoch))
            break

    print("best val performance = {}".format(evaluator.get_val_best_performance()))
    model.load_state_dict(evaluator.get_best_model())
    test_performance = evaluate(metric, test_loader, model, user_feat, device=DEVICE, params=evaluation_params)

    if config["tune"]:
        if config["metric"] == "mse":
            session.report({
                "mse": evaluator.get_val_best_performance(),
                "test_mse": test_performance
            })
        else:
            session.report({
                "ndcg": evaluator.get_val_best_performance(),
                "test_ndcg": test_performance[0],
                "test_recall": test_performance[1]
            })
    print("test performance is {}".format(test_performance))


if __name__ == '__main__':
    args = parse_args()
    model_name = "mfwf"

    if args.tune:
        config = {
            "tune": True,
            "show_log": False,
            "patience": args.patience,
            "data_params": args.data_params,
            "metric": args.metric,
            "batch_size": args.data_params["batch_size"],
            "lr_rate": tune.grid_search([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
            "epochs": 100,
            "weight_decay": tune.grid_search([1e-5, 1e-6]),
            "embedding_dim": 64,
            "topk": args.topk,
            "seed": args.seed,

        }
        name_suffix = ""
        if args.test_seed:
            name_suffix = "_seed"
            if args.data_params["name"] == "coat":
                lr = 1e-3
                wd = 1e-5
            elif args.data_params["name"] == "yahoo":
                lr = 5e-4
                wd = 1e-6
            elif args.data_params["name"] == "kuai_rand":
                lr = 5e-5
                wd = 1e-6
            elif args.data_params["name"] == "sim":
                r_list = args.sim_suffix.split("_")
                sr = eval(r_list[2])
                cr = eval(r_list[4])
                tr = eval(r_list[-1])
                param = read_best_params(model_name, args.key_name, sr, cr, tr)
                lr = param["lr"]
                wd = param["wd"]

            config["lr_rate"] = lr
            config["weight_decay"] = wd
            config["seed"] = tune.grid_search(test_seeds)
        res_name = model_name + name_suffix
        if args.data_params["name"] == "sim":
            res_name = res_name + args.sim_suffix
        tune_param_rating(train_eval, config, args, res_name)
    else:
        sample_config = {
            "metric": args.metric,
            "data_params": args.data_params,
            "tune": False,
            "show_log": True,
            "patience": args.patience,
            "lr_rate": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 30,
            "batch_size": args.data_params["batch_size"],
            "embedding_dim": 64,
            "topk": args.topk,
            "seed": args.seed,
        }

        train_eval(sample_config)
