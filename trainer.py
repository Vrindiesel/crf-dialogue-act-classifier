
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim

import dill
from progress.bar import Bar
from torch.nn.utils import clip_grad_norm

try:
    import Dataset
except ModuleNotFoundError as e:
    import DA_Classifier.Dataset as Dataset


class Trainer(object):

    def __init__(self, model, optim, loss_fn, update_interval=100, cuda=False,
                 max_grad_norm=None):
        self.model = model
        self.optimizer = optim
        self.update_interval = update_interval
        self.cuda = cuda
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm


    def train_on_epoch(self, epoch, train_iter):
        train_iter.init_epoch()
        self.model.train()

        losses = []
        all_stats = []
        num_batches = len(train_iter)

        print("\n Training:")
        for b, batch in enumerate(train_iter):
            self.model.zero_grad()
            #optimizer.zero_grad()

            conv, conv_seq_len, utt_len = batch.conversation
            labels = batch.labels.view(conv.size(0), -1)

            #print("utt_len:", utt_len.size())
            #print(utt_len)

            utt_len = utt_len.to(torch.device('cpu'))

            if self.cuda:
                conv = conv.to(torch.device('cuda'))
                #utt_len = utt_len.to(torch.device('cuda'))
                labels = labels.to(torch.device('cuda'))


            output, crf_input = self.model(conv, utt_len)
            loss, stats = self.loss_fn(crf_input, labels, conv, output)
            losses.append(stats)
            all_stats.append(stats)

            loss.backward()
            if self.max_grad_norm is not None:
                clip_grad_norm(self.model.rnn_params(), self.max_grad_norm)

            self.optimizer.step()

            if b % self.update_interval == 0:
                num_examples = sum(ba["batch_size"] for ba in losses)
                loss_val = sum(ba["batch_size"]*ba["loss"] for ba in losses)
                n_correct = sum(ba["num_correct"] for ba in losses)
                tot = sum(ba["total"] for ba in losses)
                acc = n_correct / tot

                print("Epoch {} batch {}/{} loss: {:.4f}\tacc: {:.4f}".format(
                    epoch, b, num_batches, loss_val/num_examples, acc))
                losses = []

        stats = {
            "num_batches": sum(ba["batch_size"] for ba in all_stats),
            "total_loss": sum(ba["batch_size"]*ba["loss"] for ba in all_stats),
            "num_correct": sum(ba["num_correct"] for ba in all_stats),
            "num_utterances": sum(ba["total"] for ba in all_stats)
        }
        stats["accuracy"] = stats["num_correct"] / stats["num_utterances"]
        stats["utt_avg_loss"] = stats["total_loss"] / stats["num_utterances"]
        print("\nTraining Stats:")
        for k, v in stats.items():
            print(" * {}: {:.4f}".format(k, v))

        return stats

    def evaluate(self, eval_iter):
        eval_iter.init_epoch()
        self.model.eval()

        all_stats = []
        num_batches = len(eval_iter)

        bar = Bar("Evaluating ", max=num_batches)

        for b, batch in enumerate(eval_iter):
            conv, conv_seq_len, utt_len = batch.conversation
            labels = batch.labels.view(conv.size(0), -1)
            utt_len = utt_len.to(torch.device("cpu"))

            if self.cuda:
                conv = conv.to(torch.device('cuda'))
                labels = labels.to(torch.device('cuda'))


            output, crf_input = self.model(conv, utt_len)
            loss, stats = self.loss_fn(crf_input, labels, conv, output)
            all_stats.append(stats)
            bar.next()
        bar.finish()


        stats = {
            "num_batches": sum(ba["batch_size"] for ba in all_stats),
            "total_loss": sum(ba["batch_size"]*ba["loss"] for ba in all_stats),
            "num_correct": sum(ba["num_correct"] for ba in all_stats),
            "num_utterances": sum(ba["total"] for ba in all_stats)
        }
        stats["accuracy"] = stats["num_correct"] / stats["num_utterances"]
        stats["utt_avg_loss"] = stats["total_loss"] / stats["num_utterances"]

        print("\nEvaluation Stats:")
        for k, v in stats.items():
            print(" * {}: {:.4f}".format(k, v))

        return stats


    def drop_checkpoint(self, opt, epoch, fields, stats):

        acc = stats["accuracy"]
        avg_loss = stats["utt_avg_loss"]

        save_path = "{}_acc{:.2f}_loss{:.2f}_e{}.pt".format(opt.save_model, acc * 100, avg_loss, epoch + 1)
        print(" * saving model", save_path)

        checkpoint = {
            "model": self.model.state_dict(),
            "fields": fields,
            "opt": opt,
            "epoch": epoch,
            "optim": self.optimizer
        }
        torch.save(checkpoint, save_path, pickle_module=dill)



def main():
    pass


if __name__ == "__main__":
    main()
