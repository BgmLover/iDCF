from torch import nn
from utils import *
from ray.air import session
from argparser import *
from scipy.sparse import csr_matrix
from evaluator import Evaluator
from tune_script import *
from seeds import test_seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MF_ips(nn.Module):
    """
    RD module for matrix factorization.
    """

    def __init__(self, num_users, num_items, InverP, y_unique, embedding_size=100, dropout=0):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(-0.01, 0.01)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        self.mean = nn.Parameter(torch.FloatTensor([0]), False)

        self.corY = y_unique

        self.drop = nn.Dropout(dropout)

        self.invP = InverP

        self.device = DEVICE

    def forward(self, u_id, i_id):
        U = self.drop(self.user_emb(u_id))
        b_u = self.user_bias(u_id).squeeze()
        I = self.drop(self.item_emb(i_id))
        b_i = self.item_bias(i_id).squeeze()
        return (U * I).sum(1) + b_u + b_i + self.mean

    def predict(self, uid, iid):
        return self.forward(uid, iid)

    def base_model_loss(self, u_id, i_id, y_train, loss_f):
        U = self.drop(self.user_emb(u_id))
        b_u = self.user_bias(u_id).squeeze()
        I = self.drop(self.item_emb(i_id))
        b_i = self.item_bias(i_id).squeeze()

        preds = (U * I).sum(1) + b_u + b_i + self.mean

        weight = torch.ones_like(y_train).to(self.device)

        for i in range(len(self.corY)):
            weight[y_train == self.corY[i]] = self.invP[i_id][y_train == self.corY[i], i]

        cost = loss_f(preds, y_train)
        loss = torch.mean(weight * cost)
        return loss

    def l2_norm(self, users, items):
        users = torch.unique(users)
        items = torch.unique(items)

        l2_loss = (torch.sum(self.user_emb(users) ** 2) + torch.sum(self.item_emb(items) ** 2)) / 2
        return l2_loss


def load_uniform_data(ratio, path, shape):
    df = pd.read_csv(path)
    size = int(ratio * len(df))
    index = np.random.permutation(np.arange(len(df)))[:size]
    rows, cols, rating = df["user_id"][index], df["item_id"][index], df["rating"][index]
    return csr_matrix(
        (rating, (rows, cols)), shape=shape
    )


def train_eval(config):
    metric = config["metric"]
    weight_decay = config["weight_decay"]
    epochs = config["epochs"]

    train_loader, val_loader, test_loader, evaluation_params, n_users, n_items, y_unique, InvP = \
        construct_ips_dataloader(
            config,
            DEVICE)

    seed_everything(config["seed"])

    model = MF_ips(n_users, n_items, InvP, y_unique, config["embedding_dim"]).to(DEVICE)

    optimizer_base = torch.optim.Adam(params=model.parameters(), lr=config["base_lr"], weight_decay=weight_decay)
    loss_func = nn.MSELoss(reduction="none")

    evaluator = Evaluator(metric, patience_max=config["patience"])

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_len = 0
        for index, (uid, iid, rating) in enumerate(train_loader):
            uid, iid, rating = uid.to(DEVICE), iid.to(DEVICE), rating.float().to(DEVICE)

            loss = model.base_model_loss(uid, iid, rating, loss_func)
            optimizer_base.zero_grad()
            loss.backward()
            optimizer_base.step()

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
    torch.save(evaluator.get_best_model(), "{}_ips.pt".format(config["data_params"]["name"]))

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
    model_name = "IPS"
    if args.tune:
        config = {
            "tune": True,
            "show_log": False,
            "patience": args.patience,
            "data_params": args.data_params,
            "metric": args.metric,
            "batch_size": args.data_params["batch_size"],
            "base_lr": tune.grid_search([5e-5, 1e-5, 1e-3, 5e-4, 1e-4, ]),
            "epochs": 100,
            "weight_decay": tune.grid_search([1e-5, 1e-6]),
            "embedding_dim": 64,
            "seed": args.seed,
            "topk": args.topk
        }
        name_suffix = ""
        if args.test_seed:
            name_suffix = "_seed"
            if args.data_params["name"] == "coat":
                lr = 1e-4
                wd = 1e-5
            elif args.data_params["name"] == "yahoo":
                lr = 1e-4
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

            config["base_lr"] = lr
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
            "patience": 20,
            "base_lr": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 100,
            "batch_size": args.data_params["batch_size"],
            "embedding_dim": 64,
            "topk": args.topk,
            "seed": args.seed,

        }

        train_eval(sample_config)
