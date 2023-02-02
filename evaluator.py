import copy
from matplotlib import pyplot as plt

from utils import *


class Evaluator:
    def __init__(self, metric, patience_max):
        self.metric = metric

        self.training_loss = []
        self.val_loss = []
        self.test_loss = []

        self.training_best = np.inf
        self.val_best = np.inf if self.metric == "mse" else -np.inf

        self.best_val_model = None

        self.patience_max = patience_max
        self.patience_counter = 0

    def record_training(self, loss):
        self.training_loss.append(loss)
        if loss < self.training_best:
            self.training_best = loss

    def record_val(self, performance, state_dict):
        self.patience_counter += 1
        self.val_loss.append(performance)

        if self.metric == "mse":
            if performance < self.val_best:
                self.val_best = performance
                self.best_val_model = copy.deepcopy(state_dict)
                self.patience_counter = 0
        elif self.metric == "ndcg":
            if performance[0] > self.val_best:
                self.val_best = performance[0]
                self.best_val_model = copy.deepcopy(state_dict)
                self.patience_counter = 0
        else:
            raise Exception("invalid metric")

        if self.patience_counter >= self.patience_max:
            return True
        return False

    def record_test(self, performance):
        self.test_loss.append(performance)

    def get_best_model(self):
        return self.best_val_model

    def epoch_log(self, epoch):
        print("epoch:{}, loss:{:.5}, val performance:{}; {} ".format(epoch,
                                                                     self.training_loss[epoch],
                                                                     self.val_loss[epoch],
                                                                     self.test_loss[epoch]))

    def get_val_best_performance(self):
        return self.val_best

    def plot(self):
        if self.metric == "mse":
            plt.plot(self.val_loss, label="val MSE")
            plt.plot(self.test_loss, label="test MSE")
        elif self.metric == "ndcg":
            plt.plot(self.val_loss, label="val NDCG")
            plt.plot(self.test_loss, label="test NDCG")
        plt.legend()
        plt.show()
