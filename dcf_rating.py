from torch import nn
from models.mf import MF
from ray.air import session

from argparser import *
from tune_script import *
from evaluator import Evaluator
from utils import *
from seeds import test_seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DCF(nn.Module):
    def __init__(self, num_users, num_items,
                 a_hat, embedding_size,
                 device="cpu"):
        super(DCF, self).__init__()
        self.mf_layer = MF(num_users, num_items, embedding_size)
        self.a_hat = a_hat

        self.user_coe = nn.Embedding(num_users, 1)
        self.user_coe.weight.data.uniform_(-0.01, 0.01)

        self.mf_layer.to(device)
        self.to(device)

    def forward(self, uid, iid):
        mf_output = self.mf_layer(uid, iid)
        latent_regression = self.user_coe(uid).view(-1) * self.a_hat[uid, iid]

        return mf_output + latent_regression

    def predict(self, uid, iid):
        return self.forward(uid, iid)


def train_eval(config):
    metric = config["metric"]
    params = config["data_params"]

    A_hat = pd.read_csv(params["dcf_A_hat_path"] + "A_hat_{}.csv".format(config["A_hat_dim"]), header=None, sep=",")

    A_hat = torch.tensor(A_hat.to_numpy(), dtype=torch.float).to(DEVICE)
    train_loader, val_loader, test_loader, evaluation_params, n_users, n_items = construct_mf_dataloader(config, DEVICE)
    seed_everything(config["seed"])

    model = DCF(n_users, n_items, A_hat, config["embedding_dim"],device=DEVICE)
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

            predict = model(uid, iid).view(-1)

            loss = loss_func(predict, rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(rating)
            total_len += len(rating)

        evaluator.record_training(total_loss / total_len)

        model.eval()
        validation_performance = mf_evaluate(metric, val_loader, model, device=DEVICE,
                                             params=evaluation_params)
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
    model_name = "dcf"
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
            "A_hat_dim": tune.grid_search([1, 2, 5, 10, 20, 50, 100]),
            "topk": args.topk,
            "seed": args.seed,

        }
        name_suffix = ""
        if args.test_seed:
            name_suffix = "_seed"
            if args.data_params["name"] == "coat":
                lr = 1e-3
                wd = 1e-6
                A_hat_dim = 20
            elif args.data_params["name"] == "yahoo":
                lr = 1e-3
                wd = 1e-6
                A_hat_dim = 20
            elif args.data_params["name"] == "kuai_rand":
                lr = 1e-4
                wd = 1e-6
                A_hat_dim = 100
            elif args.data_params["name"] == "sim":
                r_list = args.sim_suffix.split("_")
                sr = eval(r_list[2])
                cr = eval(r_list[4])
                tr = eval(r_list[-1])
                param = read_best_params(model_name, args.key_name, sr, cr, tr)
                lr = param["lr"]
                wd = param["wd"]
                A_hat_dim = param["A_hat_dim"]
                
            config["lr_rate"] = lr
            config["weight_decay"] = wd
            config["A_hat_dim"] = A_hat_dim 
            config["seed"] = tune.grid_search(test_seeds)

        res_name = model_name + name_suffix
        if args.data_params["name"] == "sim":
            res_name = res_name + args.sim_suffix
        tune_param_rating(train_eval, config, args, res_name)
    else:
        # params of outcome model
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
            "A_hat_dim": 100,
            "topk": args.topk,
            "seed": args.seed,

        }

        # params of pretrained vae

        train_eval(sample_config)
