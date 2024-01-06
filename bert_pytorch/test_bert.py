import torch
import argparse
import utils as ut


def get_bert_encoding(devices, net, tokens_a, tokens_b=None, vocab=None):
    tokens, segments = ut.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices).unsqueeze(0)
    segments = torch.tensor(segments, device=devices).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices).unsqueeze(0)
    encoded_X, _, nsp_Y_hat = net(token_ids, segments, valid_len)
    return encoded_X, nsp_Y_hat


def test_bert(model_path, batch_size, max_len):

    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    model = torch.load(model_path, map_location=device)
    # print("----------model:-----------\n{}".format(model))
    _, vocab = ut.load_data_wiki(batch_size, max_len, num_workers=0)
    model.eval()
    with torch.no_grad():
        tokens_a = ['a', 'crane', 'is', 'flying']
        tokens_b = ['he', 'just', 'left']
        encoded_text, nsp_Y_hat = get_bert_encoding(device, model, tokens_a, tokens_b, vocab=vocab)
        # 词元：'<cls>','a','crane','is','flying','<sep>'
        encoded_text_cls = encoded_text[:, 0, :]
        encoded_text_crane = encoded_text[:, 2, :]
        print(encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default="./bert_weight.pth", help='Path to save model weights')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--max_len', type=int, default=64, help='max length of sequence')
    args = parser.parse_args()

    test_bert(args.weight_path, args.batch_size, args.max_len)
