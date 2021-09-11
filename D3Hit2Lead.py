# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : D3Hit2Lead.py
@Function : Automatic compound modification based on molecular and protein sequences
@Project  : molecule_generation_1014
@Time     : 2021/9/2 19:22
@Author   : Xuelian li
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/9/2 19:22        1.0             None
2021/9/8 22:24        2.0             None
"""

import unicodedata
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from datetime import datetime

# 判断是否是GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
teacher_forcing_ratio = 0.5
embedding_size = 256
embedding_pro_size = 64
encoder_hidden_size = 512

learning_rate = 0.005
print_every = 5000
plot_every = 500
n_epoches = 3
pro_hidden_size = 128
n_layers  = 3
dropout = 0.1

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 70
MIN_LENGTH = 25

decoder_hidden_size = encoder_hidden_size + pro_hidden_size

class Seq:
    def __init__(self, name):
        self.name = name
        self.char2index = {'SOS': 0, 'EOS': 1}  # vocabulary
        self.index2char = {0: 'SOS', 1: 'EOS'}
        self.n_chars = 2  # 字符数目

    def Sequenceadd(self, sequence):
        for char in sequence:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1

def prepareData(com_seq, incom_seq, protein):
    """

    :param com_seq: 完整分子
    :param incom_seq: 切割分子
    :param protein:
    :return:
    """
    ################## initialize the classes
    input_seq = Seq(incom_seq)
    output_seq = Seq(com_seq)
    protein_seq = Seq(protein)
    ##################initialize the lines
    # Read the file and split into lines
    lines = open('0525ligand_pro_15cutways.txt', encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[re.sub(r'\\', '_', unicodeToAscii(s.strip())) for s in l.split('\t')] for l in lines]
    num = 0
    for p in pairs:
        tmp = p[0]
        p[0] = p[1]
        p[1] = tmp
    # delete the sequences that are too long or too short
    pairs = [pair for pair in pairs if (len(pair[0]) < MAX_LENGTH
                                        and len(pair[1]) < MAX_LENGTH)
             and (len(pair[0]) > MIN_LENGTH and len(pair[1]) > MIN_LENGTH)]

    ################## add sequences into the classes
    for pair in pairs:
        input_seq.Sequenceadd(pair[0])
        output_seq.Sequenceadd(pair[1])
        protein_seq.Sequenceadd(pair[2])

    return input_seq, output_seq, protein_seq, pairs

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers, dropout):
        super(LSTMEncoder, self).__init__()
        # Parameters
        self.input_size = input_size  # 不完整分子序列字典词汇量
        self.embedding_size = embedding_size  # 嵌入层维数
        self.hidden_size = hidden_size  # LSTM单元的隐状态维数
        self.n_layers = n_layers  # LSTM单元层数
        self.dropout = dropout  # LSTM单元的dropout值

        # Layers Definition
        self.embedding = nn.Embedding(input_size, embedding_size)
        # 定义嵌入层
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=False,
                            batch_first=True)
        # 定义n_layers层的LSTM单元，bidirectional=False表示该单元为单向LSTM单元

    # 定义前向传播函数forward
    def forward(self, input, hidden):
        # input为当前时间步输入的字符编码，为一维tensor
        # hidden为当前时间步LSTM单元的隐状态
        # 对于LSTM单元来说，由（hn,cn）构成，其维度均为(batch, num_layers*num_directions, hidden_size)
        # 由于设置batch_first=1，因此batch为输入输出的第一维
        # 在本模型中，每次喂Encoder和Decoder一个字符，batch = 1
        # 在本模型中，LSTM单元为单向，num_directions = 1

        embedded = self.embedding(input).view(1, 1, -1)
        # 通过嵌入层将输入input嵌入为embedding_size维的tensor
        # 再通过view将其转换为三维tensor（1，1，embedding_size），以作为LSTM的输入
        output = embedded
        output, hidden = self.lstm(output, hidden)
        # output = （batch = 1, seq_length = 1, num_directions*hidden_size = hidden_size)
        # hidden: hn,cn = (batch = 1, num_layers*num_directions, hidden_size)
        return output, hidden

    # initHidden函数用于初始化LSTM单元的hidden state
    def initHidden(self):
        return torch.zeros(self.n_layers , 1, self.hidden_size)
        # hidden = (num_layers*num_directions, batch_size = 1, hidden_size)

class LSTMProEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers, dropout):
        super(LSTMProEncoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers  # LSTM单元层数
        self.dropout = dropout  # LSTM单元的dropout值

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=False,
                            batch_first=True)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        # |hidden| = (num_layers*num_directions, batch_size, hidden_size)
        return torch.zeros(self.n_layers, 1, self.hidden_size)



# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, n_layers, dropout):
        super(LSTMDecoder, self).__init__()
        # Parameters
        self.output_size = output_size  # 完整分子序列字典词汇量
        self.embedding_size = embedding_size  # 嵌入层维数
        self.hidden_size = hidden_size  # LSTM单元的隐状态维数
        self.n_layers = n_layers  # LSTM单元层数
        self.dropout = dropout  # LSTM单元的dropout值

        # Layers Definition
        self.embedding = nn.Embedding(output_size, embedding_size)
        # 定义嵌入层
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=False,
                            batch_first=True)
        # 定义n_layers层的LSTM单元，bidirectional=False表示该单元为单向LSTM单元
        self.out = nn.Linear(hidden_size, output_size)
        # 定义全连接层out
        self.softmax = nn.LogSoftmax(dim=1)
        # 定义softmax层，选择LogSoftmax函数和后续train函数中计算loss的NLLLoss函数相组合

    def forward(self, input, hidden):
        # input为当前时间步输入的字符编码，为一维tensor
        # hidden为当前时间步LSTM单元的隐状态
        # 对于LSTM单元来说，由（hn,cn）构成，其维度均为(batch, num_layers*num_directions, hidden_size)
        # 在本模型中，每次喂Encoder和Decoder一个字符，batch = 1
        # 在本模型中，LSTM单元为单向，num_directions = 1
        output = self.embedding(input).view(1, 1, -1)
        # 通过嵌入层将输入input嵌入为embedding_size维的tensor
        # 再通过view将其转换为三维tensor（1，1，embedding_size），以作为LSTM的输入
        output = F.relu(output)
        # 选用ReLU激活函数
        output, hidden = self.lstm(output, hidden)
        # output = （batch = 1, seq_length = 1, num_directions*hidden_size = hidden_size)
        # hidden: hn,cn = (batch = 1, num_layers*num_directions, hidden_size）
        output = self.softmax(self.out(output[0]))
        # out层将output转换为tensor（1，output_size）
        # 其中，output_size为输出序列字典字符数
        return output, hidden

def tensorFromSequence(lang, sentence):
    indexes = [lang.char2index[word] for word in sentence]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def merge_hiddens(encoder_mol_hidden, encoder_pro_hidden):
    # encoder_mol_hidden为编码不完整分子序列的encoder最后一个时间步输出的隐状态
    # encoder_pro_hidden为编码蛋白质序列的encoder最后一个时间步输出的隐状态
    hiddens_n, cells_n = encoder_mol_hidden
    hiddens_n_pro, cells_n_pro = encoder_pro_hidden
    new_hiddens = torch.cat((hiddens_n_pro,hiddens_n),dim=-1)
    new_cells = torch.cat((cells_n_pro,cells_n),dim=-1)
    # 将隐状态中的hn和cn分别拼接起来，得到拼接后的（new_hiddens, new_cells），
    # 将其作为Decoder的初始隐状态
    return (new_hiddens, new_cells)

def evaluate(sentence, protein, encoder, encoder_pro, decoder, max_length=MAX_LENGTH):
    encoder.eval()
    encoder_pro.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensorFromSequence(input_seq, sentence)

        input_length = input_tensor.size(0)

        input_protein = tensorFromSequence(protein_seq, protein)
        pro_length = input_protein.size(0)

        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))

        encoder_pro_hidden = (encoder_pro.initHidden().to(device), encoder_pro.initHidden().to(device))

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)


        ii = 0
        for pi in range(input_length, input_length + pro_length):
            encoder_pro_output, encoder_pro_hidden = encoder_pro(input_protein[ii], encoder_pro_hidden)
            ii = ii + 1


        decoder_input = torch.tensor([[SOS_token]]).to(device)

        decoder_hidden = merge_hiddens(encoder_hidden, encoder_pro_hidden)


        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)


            topv, topi = decoder_output.data.topk(1)  # top-1 value, index
            # |topv|, |topi| = (1, 1)

            if topi.item() == EOS_token:

                break
            else:
                decoded_words.append(output_seq.index2char[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words

def FlitBack(s): #转换回最初的SMILES分子式，将'_'转换回'\'
    b = re.sub('_', r'\\', s)
    return b

def translate(smi, output):
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_object = open('./Pro-LSTM-Seq2Seq_2/results/LSTMresults_' + date_time + '.txt', 'w', encoding='utf-8')
    #file_object.writelines("原始输入|预测输出\n")
    file_object.writelines("The input hit|predicted lead\n")
    file_object.writelines(FlitBack(smi) + '\t')
    file_object.writelines(FlitBack(output) + '\n')
    # print('{}|{}'.format(pair[1], output))

if __name__ == "__main__":

    #M = input("请输入需要预测的分子数(整数)：")
    M = input("Please input the number of the molecues to be modified (integer, e.g., 1): ")
    for i in range(int(M)):
        input_seq, output_seq, protein_seq, pairs = prepareData('com', 'incom', 'protein')
        # 通过加载LSTM-Pro/r1下的model进行预测。
        #print("请输入分子以及蛋白质序列，并以空格隔开:")
        print("Please input the compound smiles and protein fasta. The smiles and fasta should be separated by space:")
        input_line = input().split()
        if len(input_line) > 2:
            #print("错误的输入")
            print("The input is error, the right one is like this: C(/C=C(/F)\C=C(\S(=O)(=O)/C(=C/C)/C=C\CCNC(=O)/C=C/NC)/C)F	MNAAAEAEFNILLATDSYKVTHYKQYPPNTSKVYSYFECREKKTENSKVRKVKYEETVFYGLQYILNKYLKGKVVTKEKIQEAKEVYREHFQDDVFNERGWNYILEKYDGHLPIEVKAVPEGSVIPRGNVLFTVENTDPECYWLTNWIETILVQSWYPITVATNSREQKKILAKYLLETSGNLDGLEYKLHDFGYRGVSSQETAGIGASAHLVNFKGTDTVAGIALIKKYYGTKDPVPGYSVPAAEHSTITAWGKDHEKDAFEHIVTQFSSVPVSVVSDSYDIYNACEKIWGEDLRHLIVSRSTEAPLIIRPDSGNPLDTVLKVLDILGKKFPVSENSKGYKLLPPYLRVIQGDGVDINTLQEIVEGMKQKKWSIENVSFGSGGALLQKLTRDLLNCSFKCSYVVTNGLGVNVFKDPVADPNKRSKKGRLSLHRTPAGTFVTLEEGKGDLEEYGHDLLHTVFKNGKVTKSYSFDEVRKNAQLNMEQDVAPH")
        else:
            smi, protein = input_line[:]
            smi = re.sub(r'\\', '_', unicodeToAscii(smi.strip()))
            # 加载模型
            encoder_path = "Pro-LSTM-Seq2Seq_2/full_encoder.pth"
            encoder_pro_path = "Pro-LSTM-Seq2Seq_2/full_length_encoder_pro.pth"
            decoder_path = "Pro-LSTM-Seq2Seq_2/full_decoder.pth"
            encoder = LSTMEncoder(input_size=input_seq.n_chars,
                                  embedding_size=embedding_size,
                                  hidden_size=encoder_hidden_size, n_layers=n_layers, dropout=dropout
                                  ).to(device)

            encoder_pro = LSTMProEncoder(input_size=protein_seq.n_chars,
                                         embedding_size=embedding_pro_size,
                                         hidden_size=pro_hidden_size, n_layers=n_layers, dropout=dropout
                                         ).to(device)

            decoder = LSTMDecoder(output_size=output_seq.n_chars,
                                  embedding_size=embedding_size,
                                  hidden_size=decoder_hidden_size, n_layers=n_layers, dropout=dropout
                                  ).to(device)
            encoder_model = encoder.load_state_dict(torch.load(encoder_path, map_location=device))
            encoder_pro_model = encoder_pro.load_state_dict(torch.load(encoder_pro_path, map_location=device))
            decoder_model = decoder.load_state_dict(torch.load(decoder_path, map_location=device))

            output_words = evaluate(smi, protein, encoder, encoder_pro, decoder)
            output_sentence = ''.join(output_words)

            #print("预测输出结果：", output_sentence)
            print("The predicted lead (modified compound) is：", output_sentence)
            # for print
            translate(smi, output_sentence)
