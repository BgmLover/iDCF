from models.vae import *
from torch import nn
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
from utils import *
from argparser import *
from ray.air import session
from torch.utils.data import Dataset, DataLoader
from tune_script import *
from seeds import test_seeds


def loss_function(x, x_hat, mean, log_var, anneal=1., mask=None):
    if mask is None:
        reproduction_loss = torch.mean(
            torch.sum(nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="none"), dim=1))
    else:
        entropy = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
        make_sure = x * mask
        assert make_sure.sum() == 0
        valid_mask = torch.logical_not(torch.logical_and(torch.logical_not(x), mask))
        filter_entropy = entropy * valid_mask
        reproduction_loss = torch.mean(
            filter_entropy.sum(dim=1)
        )
    kld = -0.5 * torch.mean(torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1))
    # print(reproduction_loss.item(), kld.item(), anneal)
    return reproduction_loss + kld * anneal


class TrainingDataset(Dataset):
    def __init__(self, data, mask=None):
        self.data = data
        self.mask = mask

    def __getitem__(self, index):
        if self.mask is None:
            return self.data[index]
        else:
            return self.data[index], self.mask[index]

    def __len__(self):
        return len(self.data)


def train_eval(config):
    params = config["data_params"]
    train_ratio = params["train_ratio"]
    train_matrix, val_matrix, train_user_index, test_user_index = construct_vae_dataset(params["train_path"],
                                                                                        train_ratio=train_ratio,
                                                                                        split_test=False, )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = torch.tensor(train_matrix > 0, dtype=torch.float).to(device)
    val_data = torch.tensor(val_matrix > 0, dtype=torch.float).to(device)

    n_users = train_matrix.shape[0] + val_matrix.shape[0]
    n_items = train_matrix.shape[1]

    train_mask = None
    val_mask = None
    train_dataset = TrainingDataset(data=train_data, mask=train_mask)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    hidden_dim = config["hidden_dim"]
    latent_dim = config["latent_dim"]

    lr = config["lr"]
    epochs = config["epochs"]
    seed_everything(config["seed"])

    model = VAE(input_dim=n_items, latent_dim=latent_dim, hidden_dim=hidden_dim, n_layers=config["n_layers"],
                device=device)

    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = np.inf
    best_training_loss = np.inf
    val_loss_list = []
    training_loss_list = []
    patience_counter = 0
    patience = config["patience"]

    use_anneal = config["anneal"]
    anneal_max = config["beta_max"]
    anneal_count = 0
    total_batches = int(epochs * train_data.shape[0] / config["batch_size"])
    anneal_max_count = int(0.2 * total_batches / anneal_max)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_len = 0
        for x in train_dataloader:
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)

            if use_anneal:
                anneal = min(anneal_max, 1. * anneal_count / anneal_max_count)
            else:
                anneal = anneal_max

            l2_reg = torch.tensor([0]).to(device)
            for param in model.parameters():
                l2_reg = l2_reg + torch.norm(param)
            loss = loss_function(x, x_hat, mean, log_var, anneal, None) + l2_reg * config["l2_penalty"]
            loss.backward()
            optimizer.step()
            anneal_count += 1
            total_loss += loss.item() * len(x)
            total_len += len(x)

        predict_test_x, mean_val, log_var_val = model(val_data)
        val_loss = loss_function(val_data, predict_test_x, mean_val, log_var_val, anneal=anneal_max, mask=None).item()
        training_loss = total_loss / total_len

        patience_counter += 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "{}_{}_vae_exposure_best.pt".format(params["name"], "val"))
            patience_counter = 0
        if training_loss < best_training_loss:
            best_training_loss = training_loss
            torch.save(model.state_dict(), "{}_{}_vae_exposure_best.pt".format(params["name"], "train"))

        if config["show_log"]:
            val_loss_list.append(val_loss)
            training_loss_list.append(training_loss)
            print("Epoch {}, Training Loss = {}, Val Loss = {} ".format(epoch, training_loss, val_loss))

        if patience_counter >= patience:
            if config["show_log"]:
                print("reach max patience {}, current epoch {}".format(patience, epoch))
            break

    if config["tune"]:
        session.report({
            "training_loss": best_training_loss,
            "val_loss": best_val_loss
        })

    if config["show_log"]:
        plt.plot(val_loss_list, label="Val Loss")
        plt.plot(training_loss_list, label="Training Loss")
        plt.title("VAE")
        plt.legend()
        plt.show()
        print("Best Training loss = {}, Val loss = {}".format(best_training_loss, best_val_loss))


if __name__ == '__main__':
    args = parse_args()
    if args.tune:
        config = {
            "tune": True,
            "show_log": False,
            "patience": args.patience,
            "anneal": True,
            "batch_size": args.data_params["batch_size"],
            "lr": tune.grid_search([1e-2, 1e-3, 1e-4]),
            "epochs": 1000,
            "latent_dim": 2 if args.data_params["name"] == "sim" else tune.grid_search([16, 32]),
            "hidden_dim": tune.grid_search([32, 64]),
            "weight_decay": tune.grid_search([1e-5, 1e-6]),
            "l2_penalty": tune.grid_search([0, 0.01, 0.05, 0.1]),
            "n_layers": 3,
            "beta_max": args.data_params["beta_max"],
            "data_params": args.data_params,
            "seed": args.seed,
        }

        res_name = "vae_{}".format(args.seed)
        if args.data_params["name"] == "sim":
            res_name = res_name + args.sim_suffix
        tune_param_exposure(train_eval, config, args, res_name)
    else:
        sample_config = {
            "tune": False,
            "lr": 1e-3,
            "epochs": 1000,
            "latent_dim": 32,
            "hidden_dim": 32,
            "beta_max": args.data_params["beta_max"],
            "l2_penalty": 0.0,
            "anneal": True,
            "data_params": args.data_params,
            "patience": args.patience,
            "weight_decay": 1e-5,
            "batch_size": args.data_params["batch_size"],
            "show_log": True,
            "n_layers": 3,
            "seed": args.seed,
        }

        train_eval(sample_config)
