# 本部分代码为自己补充，目的是使用训练好的模型进行对输入文本分类
import torch
from models import Transformer
import pickle as pkl

dataset = 'THUCNews'
embedding = 'embedding_SougouNews.npz'
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

config = Transformer.Config(dataset, embedding)
vocab = pkl.load(open(config.vocab_path, 'rb'))
pkl.dump(vocab, open(config.vocab_path, 'wb'))
config.n_vocab = len(vocab)

model = Transformer.Model(config).to(config.device) 
model.load_state_dict(torch.load(config.save_path)) # 加载训练好的模型参数
model.eval()

class_dict = {0:'财经',
            1:'房产',
            2:'股票',
            3:'教育',
            4:'科技',
            5:'社会',
            6:'时政',
            7:'体育',
            8:'游戏',
            9:'娱乐',
}

def text2tensor(text, pad_size=32):
    tokenizer = lambda x: [y for y in x]  # 将每个字母分开，因为是以单个字建立的词向量
    lin = text.strip()
    content = lin
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:  # 建立向量
        words_line.append(vocab.get(word, vocab.get(UNK)))
    x = torch.LongTensor([words_line]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    return (x, seq_len)  

def output(model, text):
    model.eval()
    with torch.no_grad():
        output = model(text)
    return output

while(True):
    text = input('请输入待分类的文本(ctrl+c终止)：\n')
    result = output(model, text2tensor(text))
    predic = torch.max(result.data, 1)[1].cpu().item()
    print(f"该文本属于“{class_dict[predic]}”类 \n")
