import torch.cuda
from ray import tune
import json
import ray
import multiprocessing
from ray import air

def tune_param_rating(train_eval, config, args, model_name, num_samples=1):
    print(multiprocessing.cpu_count())
    ray.init(ignore_reinit_error=True, num_cpus=20)
    print(ray.available_resources())
    print("success")

    tuner = tune.Tuner(
        train_eval,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=num_samples
        )
    )
    if torch.cuda.is_available():
        tuner = tune.Tuner(
            tune.with_resources(train_eval,
                                resources=tune.PlacementGroupFactory(
                                    [{'CPU': 1.0, 'GPU': 1.0}] + [{'CPU': 1.0}] * 2
                                )),
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=num_samples
            )
        )

    out_dir = "res/{}/".format(config["metric"])
    results = tuner.fit()
    best_result = results.get_best_result(metric=config["metric"],
                                          mode="min" if config["metric"] == "mse" else "max")
    best_config = best_result.config  # Get best trial's hyperparameters
    best_metrics = best_result.metrics  # Get best trial's last results
    df_results = results.get_dataframe()
    df_results.to_csv(out_dir + "{}_{}.csv".format(args.dataset, model_name))
    with open(out_dir + "{}_best_{}.json".format(args.dataset, model_name), "w") as f:
        json.dump({**best_config, **best_metrics}, f)


def tune_param_exposure(train_eval, config, args, model_name):
    print(multiprocessing.cpu_count())
    ray.init(ignore_reinit_error=True, num_cpus=50)
    tuner = tune.Tuner(
        train_eval,
        param_space=config,
        run_config=air.RunConfig(name="{}_{}".format(config["data_params"]["name"],model_name))
    )
    if torch.cuda.is_available():
        tuner = tune.Tuner(
            tune.with_resources(train_eval,
                                resources=tune.PlacementGroupFactory(
                                    [{'CPU': 4.0, 'GPU': 1.0}] + [{'CPU': 1.0}] * 2
                                )),
            param_space=config
        )

    out_dir = "res/exposure/"
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_loss", mode="min")
    best_config = best_result.config  # Get best trial's hyperparameters
    best_logdir = best_result.log_dir  # Get best trial's logdir
    best_metrics = best_result.metrics  # Get best trial's last results
    df_results = results.get_dataframe()
    df_results.to_csv(out_dir + "{}_{}.csv".format(args.dataset, model_name))
    best_config['logdir']=str(best_logdir)
    with open(out_dir + "{}_best_{}.json".format(args.dataset, model_name), "w") as f:
        json.dump({**best_config, **best_metrics}, f)
