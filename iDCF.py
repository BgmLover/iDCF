import torch
from torch import nn
from models.mf import MF

from utils import *
from argparser import *
from ray.air import session
from tune_script import *
from evaluator import Evaluator, mf_evaluate
from seeds import test_seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MFMF(nn.Module):
    def __init__(self, num_users, num_items,
                 embedding_size, dropout,
                 ivae_mean, ivae_std,
                 device="cpu"):
        super(MFMF, self).__init__()
        self.mf_layer = MF(num_users, num_items, embedding_size)
        self.ivae_mean = ivae_mean
        self.ivae_std = ivae_std

        self.item_emb = nn.Embedding(num_items, self.ivae_mean.shape[1])
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.sample_size = 10
        self.drop = nn.Dropout(dropout)
        self.device = device
        self.z_linear = nn.Linear(self.ivae_mean.shape[1], self.ivae_mean.shape[1])

        self.to(device)

    def forward(self, uid, iid, sample=False):
        mf_output = self.mf_layer(uid, iid)
        i_emb = self.drop(self.item_emb(iid))
        mean = self.ivae_mean[uid]
        if sample:
            std = self.ivae_std[uid]
            samples_z = torch.randn((self.sample_size, mean.shape[0], mean.shape[1])).to(self.device)
            samples_z = samples_z * std + mean
            latent_regression = (i_emb * samples_z).sum(-1).mean(0)
        else:
            z = mean
            latent_regression = (i_emb * z).sum(1)
        return latent_regression + mf_output

    def predict(self, uid, iid):
        return self.forward(uid, iid, sample=False)


def train_eval(config):
    metric = config["metric"]

    data_params = config["data_params"]

    train_loader, val_loader, test_loader, evaluation_params, n_users, n_items = construct_mf_dataloader(config, DEVICE)

    ivae_mean = torch.load(data_params["ivae_path"] + "mean.pt").to(DEVICE)
    ivae_std = torch.load(data_params["ivae_path"] + "std.pt").to(DEVICE)

    seed_everything(config["seed"])

    model = MFMF(num_users=n_users, num_items=n_items,
                 ivae_mean=ivae_mean, ivae_std=ivae_std,
                 embedding_size=config["embedding_dim"], dropout=config["dropout"],
                 device=DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr_rate"],
                                 weight_decay=config["weight_decay"])

    loss_func = nn.MSELoss()

    evaluator = Evaluator(metric, patience_max=config["patience"])

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_len = 0
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(DEVICE), iid.to(DEVICE), rating.float().to(DEVICE)

            predict = model(uid, iid, sample=True).view(-1)

            loss = loss_func(predict, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(rating)
            total_len += len(rating)
        evaluator.record_training(total_loss / total_len)

        model.eval()
        validation_performance = mf_evaluate(metric, val_loader, model, device=DEVICE, params=evaluation_params)
        early_stop = evaluator.record_val(validation_performance, model.state_dict())
        if not config["tune"]:
            test_performance = mf_evaluate(metric, test_loader, model, device=DEVICE, params=evaluation_params)
            evaluator.record_test(test_performance)
        if config["show_log"]:
            evaluator.epoch_log(epoch)

        if early_stop:
            if config["show_log"]:
                print("reach max patience {}, current epoch {}".format(evaluator.patience_max, epoch))
            break

    print("best val performance = {}".format(evaluator.get_val_best_performance()))
    model.load_state_dict(evaluator.get_best_model())
    test_performance = mf_evaluate(metric, test_loader, model, device=DEVICE, params=evaluation_params)

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
    model_name = "iDCF"
    if args.tune:
        config = {
            "tune": True,
            "show_log": False,
            "patience": args.patience,
            "data_params": args.data_params,
            "metric": args.metric,
            "lr_rate": tune.grid_search([5e-5, 1e-5, 1e-3, 5e-4, 1e-4, ]),
            "epochs": 100,
            "weight_decay": tune.grid_search([1e-5, 1e-6]),
            "dropout": 0.,
            "batch_size": args.data_params["batch_size"],
            "embedding_dim": 64,
            "topk": args.topk,
            "seed": args.seed,

        }
        name_suffix = ""
        if args.test_seed:
            name_suffix = "_seed"
            if args.data_params["name"] == "coat":
                lr = 5e-4
                wd = 1e-6
            elif args.data_params["name"] == "yahoo":
                lr = 5e-5
                wd = 1e-5
            elif args.data_params["name"] == "kuai_rand":
                lr = 1e-4
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
            "tune": False,
            "show_log": True,
            "patience": args.patience,
            "data_params": args.data_params,
            "metric": args.metric,
            "lr_rate": 5e-4,
            "weight_decay": 1e-6,
            "epochs": 100,
            "l2_penalty": 0.0,
            "dropout": 0.,
            "batch_size": args.data_params["batch_size"],
            "embedding_dim": 64,
            "topk": args.topk,
            "seed": args.seed,
        }

        train_eval(sample_config)
