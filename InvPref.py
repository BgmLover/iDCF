import math

from torch import nn

from utils import *
from ray.air import session
from argparser import *
from tune_script import *

from evaluator import Evaluator
from seeds import test_seeds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RatingDataset(Dataset):
    def __init__(self, feature, rating, envs):
        self.feature = feature
        self.rating = rating
        self.envs = envs

    def __getitem__(self, index):
        if self.envs is None:
            return self.feature[index], self.rating[index]
        return self.feature[index], self.rating[index], self.envs[index]

    def __len__(self):
        return len(self.rating)


class LinearLogSoftMaxEnvClassifier(nn.Module):
    def __init__(self, factor_dim, env_num):
        super(LinearLogSoftMaxEnvClassifier, self).__init__()
        self.linear_map: nn.Linear = nn.Linear(factor_dim, env_num)
        self.classifier_func = nn.LogSoftmax(dim=1)
        self._init_weight()
        self.elements_num: float = float(factor_dim * env_num)
        self.bias_num: float = float(env_num)

    def forward(self, invariant_preferences):
        result: torch.Tensor = self.linear_map(invariant_preferences)
        result = self.classifier_func(result)
        return result

    def get_L1_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 1) / self.elements_num \
               + torch.norm(self.linear_map.bias, 1) / self.bias_num

    def get_L2_reg(self) -> torch.Tensor:
        return torch.norm(self.linear_map.weight, 2).pow(2) / self.elements_num \
               + torch.norm(self.linear_map.bias, 2).pow(2) / self.bias_num

    def _init_weight(self):
        torch.nn.init.xavier_uniform_(self.linear_map.weight)


class ReverseLayerF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class InvPref(nn.Module):
    def __init__(self, num_users, num_items, num_envs, embedding_size, device="cpu"):
        super(InvPref, self).__init__()
        self.user_emb_inv = nn.Embedding(num_users, embedding_size)
        self.user_bias_inv = nn.Embedding(num_users, 1)
        self.item_emb_inv = nn.Embedding(num_items, embedding_size)
        self.item_bias_inv = nn.Embedding(num_items, 1)

        self.user_emb_env = nn.Embedding(num_users, embedding_size)
        self.user_bias_env = nn.Embedding(num_users, 1)
        self.item_emb_env = nn.Embedding(num_items, embedding_size)
        self.item_bias_env = nn.Embedding(num_items, 1)

        self.env_emb = nn.Embedding(num_envs, embedding_size)
        self.env_bias = nn.Embedding(num_envs, 1)

        self.user_emb_inv.weight.data.uniform_(-0.01, 0.01)
        self.user_bias_inv.weight.data.uniform_(-0.01, 0.01)
        self.item_emb_inv.weight.data.uniform_(-0.01, 0.01)
        self.item_bias_inv.weight.data.uniform_(-0.01, 0.01)
        self.user_emb_env.weight.data.uniform_(-0.01, 0.01)
        self.user_bias_env.weight.data.uniform_(-0.01, 0.01)
        self.item_emb_env.weight.data.uniform_(-0.01, 0.01)
        self.item_bias_env.weight.data.uniform_(-0.01, 0.01)
        self.env_emb.weight.data.uniform_(-0.01, 0.01)
        self.env_bias.weight.data.uniform_(-0.01, 0.01)

        self.env_classifier = LinearLogSoftMaxEnvClassifier(embedding_size, num_envs)

    def forward(self, users_id, items_id, envs_id, alpha=0):
        users_embed_invariant = self.user_emb_inv(users_id)
        items_embed_invariant = self.item_emb_inv(items_id)

        users_embed_env_aware = self.user_emb_env(users_id)
        items_embed_env_aware = self.item_emb_env(items_id)

        envs_embed = self.env_emb(envs_id)

        invariant_preferences = users_embed_invariant * items_embed_invariant
        env_aware_preferences = users_embed_env_aware * items_embed_env_aware * envs_embed

        invariant_score = torch.sum(invariant_preferences, dim=1) \
                          + self.user_bias_inv(users_id).view(-1) \
                          + self.item_bias_inv(items_id).view(-1)

        env_aware_mid_score = torch.sum(env_aware_preferences, dim=1) \
                              + self.user_bias_env(users_id).view(-1) \
                              + self.item_bias_env(items_id).view(-1) \
                              + self.env_bias(envs_id).view(-1)

        env_aware_score = invariant_score + env_aware_mid_score

        reverse_invariant_preferences = ReverseLayerF.apply(invariant_preferences, alpha)
        env_outputs = self.env_classifier(reverse_invariant_preferences)

        return invariant_score.view(-1), env_aware_score.view(-1), env_outputs.view(-1, self.env_emb.num_embeddings)

    def predict(self, users_id, items_id):
        users_embed_invariant = self.user_emb_inv(users_id)
        items_embed_invariant = self.item_emb_inv(items_id)
        invariant_preferences = users_embed_invariant * items_embed_invariant

        invariant_score = torch.sum(invariant_preferences, dim=1) \
                          + self.user_bias_inv(users_id).view(-1) \
                          + self.item_bias_inv(items_id).view(-1)

        return invariant_score


def update_dataloader(original, new_envs):
    dataset = copy.deepcopy(original.dataset)
    dataset.envs = new_envs
    new_random_loader = DataLoader(dataset, original.batch_size, shuffle=True, num_workers=original.num_workers,
                                   pin_memory=original.pin_memory)
    new_sequential_loader = DataLoader(dataset, original.batch_size, shuffle=True, num_workers=original.num_workers,
                                       pin_memory=original.pin_memory)
    return new_random_loader, new_sequential_loader


def train_eval(config):
    metric = config["metric"]

    n_envs = config["num_envs"]
    alpha = 0
    inv_coe = config["inv_coe"]
    env_coe = config["env_coe"]
    cluster_interval = config["cluster_interval"]

    random_train_loader, val_loader, test_loader, evaluation_params, n_users, n_items = construct_mf_dataloader(
        config,
        DEVICE,
        require_index=True)
    envs = torch.randint(n_envs, size=(len(random_train_loader.dataset),)).to(DEVICE)
    sequential_train_loader = DataLoader(random_train_loader.dataset, random_train_loader.batch_size, shuffle=False,
                                         num_workers=random_train_loader.num_workers,
                                         pin_memory=random_train_loader.pin_memory)
    seed_everything(config["seed"])

    model = InvPref(n_users, n_items, n_envs, config["embedding_dim"]).to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["lr_rate"], weight_decay=config["weight_decay"])
    loss_func = nn.MSELoss()
    loss_func_clf = nn.NLLLoss()
    cluster_distance_func = nn.MSELoss(reduction="none")

    evaluator = Evaluator(metric, patience_max=config["patience"])

    batch_num = math.ceil(len(sequential_train_loader.dataset) / config["batch_size"])
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        total_len = 0
        for batch_index, (uid, iid, y, i) in enumerate(random_train_loader):
            uid, iid, y, i = uid.to(DEVICE), iid.to(DEVICE), y.to(DEVICE), i.to(DEVICE)
            e = envs[i]
            p = float(batch_index + (epoch + 1) * batch_num) / float((epoch + 1) * batch_num)
            alpha = 2. / (1. + np.exp(-10. * p)) - 1.

            inv_score, env_score, env_out = model(uid, iid, e, alpha)

            inv_loss = loss_func(inv_score, y)
            env_loss = loss_func(env_score, y)
            clf_loss = loss_func_clf(env_out, e)

            loss = inv_loss * inv_coe + env_loss * env_coe + clf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_len += len(y)
        evaluator.record_training(total_loss / total_len)

        model.eval()
        if (epoch + 1) % cluster_interval == 0:
            new_envs = []
            for index, (uid, iid, y, _) in enumerate(sequential_train_loader):
                uid, iid, y, = uid.to(DEVICE), iid.to(DEVICE), y.to(DEVICE)
                all_distances = []
                for env in range(n_envs):
                    env_tensor = torch.full((uid.shape[0],), env).to(DEVICE)
                    _, env_score, _ = model(uid, iid, env_tensor, 0)
                    distances = cluster_distance_func(env_score, y)
                    all_distances.append(distances.view(-1, 1))

                env_distances = torch.cat(all_distances, dim=1)
                new_envs_batch = torch.argmin(env_distances, dim=1)
                new_envs.append(new_envs_batch)
            envs = torch.cat(new_envs, dim=0).to(DEVICE)

        validation_performance = mf_evaluate(metric, val_loader, model, device=DEVICE,
                                             params=evaluation_params)
        early_stop = evaluator.record_val(validation_performance, model.state_dict())
        if not config["tune"]:
            test_performance = 0
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
    model_name = "invPref"
    if args.tune:
        config = {
            "tune": True,
            "show_log": False,
            "patience": args.patience,
            "data_params": args.data_params,
            "metric": args.metric,
            "batch_size": args.data_params["batch_size"],
            "lr_rate": tune.grid_search([1e-5, 5e-5, 1e-3, 5e-4, 1e-4, ]),
            "epochs": 100,
            "weight_decay": tune.grid_search([1e-5, 1e-6]),
            "embedding_dim": 64,
            "num_envs": tune.randint(2, 8),
            "inv_coe": tune.uniform(0.01, 10),
            "env_coe": tune.uniform(0.01, 10),
            "cluster_interval": tune.randint(1, 5),
            "topk": args.topk,
            "seed": args.seed,

        }
        name_suffix = ""
        num_samples = 20
        if args.test_seed:
            name_suffix = "_seed"
            num_samples = 1
            if args.data_params["name"] == "coat":
                lr = 1e-4
                wd = 1e-6
                num_envs = 6
                inv_coe = 4.964938856429313
                env_coe = 5.585819157533456
                cluster_interval = 2
            elif args.data_params["name"] == "yahoo":
                lr = 5e-4
                wd = 1e-6
                num_envs = 4
                inv_coe = 1.7033849755015773
                env_coe = 9.098100207266121
                cluster_interval = 2
            elif args.data_params["name"] == "kuai_rand":
                lr = 5e-4
                wd = 1e-5
                num_envs = 4
                inv_coe = 9.4056259844272
                env_coe = 9.602621604513832
                cluster_interval = 2
            elif args.data_params["name"] == "sim":
                r_list = args.sim_suffix.split("_")
                sr = eval(r_list[2])
                cr = eval(r_list[4])
                tr = eval(r_list[-1])
                param = read_best_params(model_name, args.key_name, sr, cr, tr)
                lr = param["lr"]
                wd = param["wd"]
                num_envs = param["num_envs"]
                inv_coe = param["inv_coe"]
                env_coe = param["env_coe"]
                cluster_interval = param["cluster_interval"]
            config["lr_rate"] = lr
            config["weight_decay"] = wd
            config["num_envs"] = num_envs
            config["inv_coe"] = inv_coe
            config["env_coe"] = env_coe
            config["cluster_interval"] = cluster_interval
            config["seed"] = tune.grid_search(test_seeds)

        res_name = model_name + name_suffix
        if args.data_params["name"] == "sim":
            res_name = res_name + args.sim_suffix
        tune_param_rating(train_eval, config, args, res_name, num_samples=num_samples)
    else:
        sample_config = {
            "metric": args.metric,
            "data_params": args.data_params,
            "tune": False,
            "show_log": True,
            "patience": args.patience,
            "lr_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 100,
            "batch_size": args.data_params["batch_size"],
            "embedding_dim": 64,
            "num_envs": 2,
            "alpha": 1.9053711444718746,
            "inv_coe": 3.351991776096847,
            "env_coe": 9.988658447411407,
            "cluster_interval": 5,
            "topk": args.topk,
            "seed": args.seed,

        }

        train_eval(sample_config)
