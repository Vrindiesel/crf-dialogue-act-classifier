
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import argparse
import random


#try:
import Dataset
#import train
import args
from bcrf import BiLSTM_CRF
from trainer import Trainer

#except ModuleNotFoundError as e:
#    import DA_Classifier.Dataset as Dataset
#    import DA_Classifier.args as args
#    from DA_Classifier.bcrf import BiLSTM_CRF
#    from DA_Classifier.trainer import Trainer


PAD = Dataset.PAD


def setup_args(arglist=None):
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args.add_args(parser)
    if arglist is None:
        opt = parser.parse_args()
    else:
        opt = parser.parse_args(arglist)

    print("Options:")
    print(opt)

    return opt


def setup_iterator2(data, batch_size=16, cuda=False):
    print("Building data iterators")
    device = torch.device('cuda:0') if cuda else None
    train_iter = torchtext.data.Iterator(dataset=data["train"], batch_size=batch_size,
                                         sort=False, repeat=False,
                                         device=device, shuffle=True)
    dev_iter = torchtext.data.Iterator(dataset=data["dev"], batch_size=batch_size, sort=False, repeat=False,
                                       device=device, shuffle=False)

    return train_iter, dev_iter



def build_embeddings(opt, field, embedding_size, pretrained=None):
    vocab_size = len(field.vocab)
    pad_id = field.vocab.stoi[PAD]

    if pretrained is None:
        embeddings = nn.Embedding(vocab_size, embedding_size,  padding_idx=pad_id)
    else:
        embeddings = nn.Embedding.from_pretrained(pretrained, ) #padding_idx=pad_id)

    if not opt.fix_word_vecs:
        embeddings.weight.requires_grad = True

    return embeddings


def build_model(opt, fields, embeddings):
    vocab_size = len(fields["conversation"].vocab)
    embedding_size = embeddings.weight.size(1)

    num_tags = len(fields["labels"].vocab)
    pad_id = fields["conversation"].vocab.stoi[PAD]


    model = BiLSTM_CRF(embeddings, embedding_size, opt.rnn_size, num_tags,
                       num_layers=opt.layers, dropout=opt.dropout)


    return model


def print_conv(conv, field):
    print("\nconv:", conv.size())
    pad_id = field.vocab.stoi[PAD]
    #print("utt_seq:", utt_seq.size())

    #print(conv)

    for j, utt_seq in enumerate(conv):
        #print(utt_seq)
        u = []
        for w in utt_seq:
            #print(w)
            u.append(field.vocab.itos[w.item()])
        print(" * ", " ".join(u))


def build_criterion(pad_id, normalization="utterance", cuda=False, loss_function=None):
    """

    :param pad_id:
    :param normalization: one of {"batch_size", "num_events"}
    :return: loss tensor
    """
    #loss_function = nn.BCEWithLogitsLoss(reduction="none")
    #loss_function = nn.NLLLoss(reduction="none")
    #loss_function = nn.BCELoss(reduction="none")
    #if cuda:
    #    loss_function = loss_function.to(torch.device('cuda:0'))

    def loss_fn(crf_in, target, inputs, output):
        loss = loss_function(crf_in, target)

        if isinstance(output, list):
            pred = torch.tensor(output).type_as(target)

        num_correct = pred.eq(target).sum().item()
        num_utts = target.size(0) * target.size(1)

        if normalization == "batch_size":
            loss = loss.div(inputs.size(0)).sum()
        elif normalization == "num_events":
            loss = loss.div(inputs.size(0) * inputs.size(1)).sum()
        else:
            loss = loss.sum()

        d = {
            "loss": loss.item(),
            "num_correct": num_correct,
            "total": num_utts,
            "batch_size": inputs.size(0),
        }

        return loss, d

    return loss_fn


def setup_optim(opt, model):
    print("Model Parameters:")
    for k, param in model.named_parameters():
        print(" * parameter {} requires_grad: {}".format(k, param.requires_grad))

    #params = filter(lambda p: p.requires_grad, model.parameters())

    if opt.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate)
    elif opt.optim == "adam":
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    elif opt.optim == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=.0001)
    elif opt.optim == "adagrad":
        raise NotImplementedError

    return optimizer




def main(arglist=None):

    opt = setup_args(arglist)

    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    _data = Dataset.get_dataset(opt)
    if len(_data) == 3:
        fields, dataset, embeddings = _data
    else:
        fields, dataset = _data
        embeddings = None

    pad_id = fields["conversation"].vocab.stoi[PAD]
    train_iter, dev_iter = setup_iterator2(dataset, batch_size=1, cuda=opt.cuda)

    if embeddings is None:
        embedding_size = opt.word_vec_size
    else:
        embedding_size = embeddings.size(1)
        assert embedding_size == opt.word_vec_size

    embeddings = build_embeddings(opt, fields["conversation"], embedding_size, pretrained=embeddings)
    model = build_model(opt, fields, embeddings)

    optimizer = setup_optim(opt, model)

    print("Model:")
    print(model)


    loss_fn = build_criterion(pad_id, loss_function=model.crf_score)

    if opt.cuda:
        #optimizer.to(torch.device('cuda:0'))
        #print("z1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    #print("z2")
    trainer = Trainer(model, optimizer, loss_fn, update_interval=opt.report_every,
                      cuda=(opt.cuda and torch.cuda.is_available()))

    print("Training")
    for epoch in range(opt.epochs):
        train_stats = trainer.train_on_epoch(epoch+1, train_iter)
        dev_stats = trainer.evaluate(dev_iter)

        if epoch+1 >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(opt, epoch, fields, dev_stats)




if __name__ == "__main__":
    glove_path = "glove.840B.300d.txt"
    ar = [
        "-cuda",
        "-batch_size", "64",
        "-epochs", "6",
        "-max_grad_norm", "5",
        "-glove_path", glove_path,
        "-optim", "adadelta",
        "-word_vec_size", "300",
        #"-fix_word_vecs",
        "-layers", "1",
        "-rnn_size", "600",
        "-data_path", "swda/ready_data/swda-corpus.json",
        "-word_min_frequency", "1",
        "-save_model", "models/da-5",
        "-save_data", "data/",
        "-load_data",
        "-start_checkpoint_at", "1",
        # dropout not implimented "-dropout", "0.2",
        "-lower",
        "-learning_rate", "0.25"
    ]

    main(ar)




