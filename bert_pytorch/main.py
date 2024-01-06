import torch
import torch.nn as nn
import argparse
import utils as ut
import BertModel as bt


# @save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)  # 把无效的pad都去掉，不计算损失
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps, weight_path, lr):
    # cuda_condition = torch.cuda.is_available()
    # device = torch.device("cuda:0" if cuda_condition else "cpu")
    # net.to(device)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    step, timer = 0, ut.Timer()
    # 画图用的类
    animator = ut.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = ut.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X, \
                mlm_weights_X, mlm_Y, nsp_y in train_iter:  # dataloader中的trainiter
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)  # vocab_size代表词表中总的词的数量
            print("epoch: {0:.3f}\tmlm_loss: {1:.3f}\tnsp_loss: {2:.3f}\tall_loss: {3:.3f}".format(step+1, mlm_l, nsp_l, l))
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1

            if step >= num_steps-5:
                torch.save(net, weight_path)

            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--max_len', type=int, default=64, help='max length of sequence')
    parser.add_argument('--num_hiddens', type=int, default=128, help='number of hidden units')
    parser.add_argument('--norm_shape', type=int, default=128, help='number of norm layers')
    parser.add_argument('--ffn_num_input', type=int, default=128, help='number of ffn layers')
    parser.add_argument('--ffn_num_hiddens', type=int, default=256, help='number of ffn hidden layers')
    parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--key_size', type=int, default=128, help='key size')
    parser.add_argument('--query_size', type=int, default=128, help='query size')
    parser.add_argument('--value_size', type=int, default=128, help='value size')
    parser.add_argument('--hid_in_features', type=int, default=128, help='Hidden layer input for adjacent sentence '
                                                                         'prediction task')
    parser.add_argument('--mlm_in_features', type=int, default=128, help='MLM task hidden layer input size')
    parser.add_argument('--nsp_in_features', type=int, default=128, help='NSP task hidden layer input')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--weight_path', type=str, default="./bert_weight.pth", help='Path to save model weights')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')

    args = parser.parse_args()

    # 加载维基百科数据，返回可迭代数据，以及vocab类
    train_iter, vocab = ut.load_data_wiki(args.batch_size, args.max_len, num_workers=0)
    with open('vocab.txt', 'w', encoding='utf-8') as f:
        f.write(str(vocab.idx_to_token))

    # for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
    #      mlm_Y, nsp_y) in train_iter:
    #     print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
    #           pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
    #           nsp_y.shape)
    #     break

    # 创建bert模型
    net = bt.BERTModel(len(vocab), num_hiddens=args.num_hiddens, norm_shape=[args.norm_shape],
                       ffn_num_input=args.ffn_num_input, ffn_num_hiddens=args.ffn_num_hiddens,
                       num_heads=args.num_heads, num_layers=args.num_layers, dropout=args.dropout,
                       key_size=args.key_size, query_size=args.query_size, value_size=args.value_size,
                       hid_in_features=args.hid_in_features, mlm_in_features=args.mlm_in_features,
                       nsp_in_features=args.nsp_in_features)

    # 指定cuda设备
    devices = ut.try_all_gpus()

    loss = nn.CrossEntropyLoss()

    train_bert(train_iter, net, loss, len(vocab), devices, args.epochs, args.weight_path, args.learning_rate)


if __name__ == '__main__':
    train()
