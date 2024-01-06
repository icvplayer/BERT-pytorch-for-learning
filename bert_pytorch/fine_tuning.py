import torch
from torch import nn
import utils as ut
import ClassifierModel as cm
from d2l import torch as d2l
import argparse


def startFineTuning(batch_size, max_len, lr, num_epochs):
    if torch.cuda.is_available():
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    else:
        devices = torch.device('cpu')

    # 加载预训练的模型
    bert, vocab = ut.load_pretrained_model(
        'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
        num_layers=2, dropout=0.1, max_len=512, devices=devices)

    data_dir = ut.download_extract('SNLI')
    # 构造可迭代的数据集对象
    train_set = ut.SNLIBERTDataset(ut.read_snli(data_dir, True), max_len, vocab)
    test_set = ut.SNLIBERTDataset(ut.read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=0)

    net = cm.BERTClassifier(bert)

    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--max_len', type=int, default=128, help='max length of sentence')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    args = parser.parse_args()

    startFineTuning(args.batch_size, args.max_len, args.lr, args.num_epochs)
