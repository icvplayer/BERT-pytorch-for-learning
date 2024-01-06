import torch
from torch import nn
import model as d2l


# @save
class BERTEncoder(nn.Module):
    """
    BERT编码器\n
    vocab_size: 词向量的大小\n
    num_hiddens: 隐藏层大小\n
    norm_shape: layer_norm的形状\n
    ffn_num_input: 前馈全连接层隐层的输入\n
    ffn_num_hiddens: 前馈全连接层隐层的大小\n
    num_heads: 多头的个数\n
    num_layers: encoder有多少层进行堆叠\n
    max_len: 句子的最大长度\n
    """

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        """
        self.token_embedding: 词向量编码\n
        self.segment_embedding: 句子分段编码\n
        self.pos_embedding: 词向量的位置编码\n
        """
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


# @save
class MaskLM(nn.Module):
    """
    BERT的掩蔽语言模型任务
    将mask后的单词对应位置的token找出来，进行预测
    """

    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        """

        :param X: 经过encoder后的输出
        :param pred_positions: x中mask位置的索引
        :return: 预测的输出
        """
        num_pred_positions = pred_positions.shape[1]  # 每一行掩码的个数
        pred_positions = pred_positions.reshape(-1)  # 调整成一行
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]  # 根据索引找到x对应的值
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))  # 变成 batch_size * num of masks * num_niddens
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


# @save
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""

    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)


# @save
class BERTModel(nn.Module):
    """
    BERT模型\n
    :param vocab_size: 词向量的大小
    :param num_hiddens: 编码器隐藏层大小
    :param norm_shape: layer_norm的形状
    :param ffn_num_input: 前馈全连接层隐层的输入
    :param ffn_num_hiddens: 前馈全连接层隐层的大小
    :param num_heads: 多头的个数
    :param num_layers: encoder有多少层进行堆叠
    :param max_len: 句子的最大长度
    :param hid_in_features: 下一个句子预测隐层的输入
    :param mlm_in_features: 最后mask预测层的输入大小
    :param nsp_in_features: 下一个句子预测层的输入大小
    :return 编码器输出, mask预测输出, 下一个句子预测输出
    """

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                                   dropout, max_len=max_len, key_size=key_size,
                                   query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)  # valid_lens代表的是每个句子pad的长度组成的列表
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
