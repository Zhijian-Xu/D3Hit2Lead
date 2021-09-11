import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import time
import math
import random
import numpy as np
from torch import optim
import os.path
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token, EOS_token = 0, 1
MAX_LENGTH = 27
MIN_LENGTH = 25

embedding_size=3
hidden_size=4
n_epoches=10
learning_rate=0.005
plot_every=500

print('==========parameters===========')
print('learning rate : %s' % (learning_rate))
print('epoches : %s' % (n_epoches))
print('embedding size : %s' % (embedding_size))
print('hidden size : %s' % (hidden_size))
print('plot_every : %s' % (plot_every))
print('max length : %s' % (MAX_LENGTH))
print('min length : %s' % (MIN_LENGTH))
print('说明')
print('===============================')

class Seq:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}  # vocabulary
        self.word2count = {}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # SOS, EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):

    s = unicodeToAscii(s.strip())
    s = re.sub(r'\\', '_', s)  # 把序列里的'\'转义字符变成'_'

    return s


def readLangs(lang1, lang2, reverse=False):
    print('Reading lines..')
    # Read the file and split into lines
    lines = open('../0507ligand_15cutways.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Seq(lang2)
        output_lang = Seq(lang1)
    else:
        input_lang = Seq(lang1)
        output_lang = Seq(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return (len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH) and (len(p[0]) > MIN_LENGTH and len(p[1]) > MIN_LENGTH)



def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print('Read {} pairs'.format(len(pairs)))

    pairs = filterPairs(pairs)
    print('Trimmed to {} pairs'.format(len(pairs)))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('com', 'incom', True)
print(random.choice(pairs))
print(input_lang.index2word)
print("Cong! We have processed the data successfully!")


################################################################
#======================seq2seq model===========================#

###########################Encoder##############################


class GRUEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):

        super(GRUEncoder, self).__init__()

        self.input_size = input_size  # 字典里一共有多少个单词
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size  # RNN隐藏层的维度，也是输出的特征维度

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        # embedding层的输出为[seq_len,batch_size,embedding_size]
        self.gru = nn.GRU(embedding_size, hidden_size,
                          bidirectional=False,
                          batch_first=True)

    def forward(self, input, hidden):

        # (seq_len=1,batch_size=1,embedding_dim)
        # (seq_len, batch, input_size)这是RNN的输入数据格式，这里只有1个时间步，seq_len是一个句子的最大长度
        embedded = self.embedding(input).view(1, 1, -1) #这里的input是一个单词（即一个batch的数据）
        output = embedded

        output, hidden = self.gru(output, hidden)
        #print(output)
        #print(hidden)

        # num_layers指有几层gru叠加，num_directions指是单向还是双向rnn，单向为1，双向为2
        return output, hidden

    def initHidden(self):
        # |hidden| = (num_layers*num_directions, batch_size, hidden_size)
        return torch.zeros(1, 1, self.hidden_size)


class GRUDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        super(GRUDecoder, self).__init__()

        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size,
                          bidirectional=False,
                          batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)


        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



####################################################################
##################train model#####################


teacher_forcing_ratio = 0.5


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(loss, axix):

    plt.title('Loss')
    #host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    # par1 = host.twinx()  # 共享x轴

    #plt.title('Loss')
    plt.plot(axix, loss)

    #plt.plot(axix, val_loss, label='Validate Loss')

    #plt.legend()  # 显示图例

    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()
    #plt.close()
    plt.savefig('results/GRU-loss')
    plt.close()



def showPlot_val(loss, val_loss, axix):
    plt.title('Train Loss & Validate Loss')
    #host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    #par1 = host.twinx()  # 共享x轴



    plt.plot(axix, loss, label='Train Loss')

    plt.plot(axix, val_loss, label='Validate Loss')

    plt.legend()  # 显示图例

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig('results/GRUloss-val_loss')
    plt.close()



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length, target_length = input_tensor.size(0), target_tensor.size(0)

    encoder_hidden = encoder.initHidden().to(device)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)


    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]]).to(device)

    decoder_hidden = encoder_hidden


    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input.
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            topv, topi = decoder_output.topk(1)  # top-1 value, index


            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])

            if decoder_input.item() == EOS_token:

                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def velidate(input_tensor,target_tensor, encoder, decoder, criterion, max_length=MAX_LENGTH):

    target_length = target_tensor.size(0)

    loss = 0
    with torch.no_grad():
        #input_tensor = tensorFromSentence(input_lang, sentence)

        input_length = input_tensor.size(0)


        encoder_hidden = encoder.initHidden().to(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)


        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden



        #decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            #decoder_attentions[di] = decoder_attention.data

            topv, topi = decoder_output.data.topk(1)  # top-1 value, index

            loss += criterion(decoder_output, target_tensor[di])

            if topi.item() == EOS_token:
                # decoded_words.append('<EOS>')
                break

            decoder_input = topi.squeeze().detach()

    return loss.item() / target_length



def trainiters(pairs, encoder, decoder,  epoches, plot_every=500, train_pairs_seed=1, learning_rate=.01):

    start = time.time()
    plot_losses = []
    plot_losses1 = []
    plot_val_losses = []
    #plot_losses1=
    axix = []
    bxix = []

    print_loss_total, plot_loss_total, plot_loss_total1 = 0, 0, 0
    print_loss_val_total, plot_loss_val_total = 0, 0
    #best_valid_loss = 1000000

    pairs_tra_val, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    train_pairs, val_pairs = train_test_split(pairs_tra_val, test_size=0.15, random_state=train_pairs_seed)

    n_iters = epoches * len(train_pairs)
    iter_num = 0


    train_pairs = [tensorsFromPair(pair) for pair in train_pairs]

    val_pairs = [tensorsFromPair(pair) for pair in val_pairs]

    print('length of train pairs')
    print(len(train_pairs))
    print('length of test pairs')
    print(len(test_pairs))
    print('length of validate pairs')
    print(len(val_pairs))




    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, epoches + 1):
        axix.append(epoch)
        i = len(train_pairs)
        for i in range(i):
            pair = train_pairs[i - 1]
            input_tensor, target_tensor = pair[0], pair[1]
            loss = train(input_tensor, target_tensor, encoder, decoder,
                         encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            plot_loss_total1 += loss
            iter_num = iter_num + 1  # 迭代训练的总次数


            if iter_num % plot_every == 0:
                plot_loss_avg1 = plot_loss_total1 / plot_every
                plot_losses1.append(plot_loss_avg1)
                plot_loss_total1 = 0
                bxix.append(iter_num)



        print_loss_avg = print_loss_total / len(train_pairs)
        print_loss_total = 0

        plot_loss_avg = plot_loss_total / len(train_pairs)
        plot_losses.append(plot_loss_avg)

        j=len(val_pairs)
        for j in range(j):
            pair_val = val_pairs[j - 1]
            input_tensor_val, target_tensor_val = pair_val[0], pair_val[1]
            loss_val = velidate(input_tensor_val, target_tensor_val, encoder, decoder, criterion)
            print_loss_val_total += loss_val
            plot_loss_val_total += loss_val

        print_loss_val_avg = print_loss_val_total / len(val_pairs)
        print_loss_val_total = 0

        plot_loss_val_avg = plot_loss_val_total / len(val_pairs)
        plot_val_losses.append(plot_loss_val_avg)

        plot_loss_total = 0
        plot_loss_val_total = 0

        print('Epoch: %d : %s (%d %d%%) Loss: %.4f | Val: %.4f' % (epoch, timeSince(start, iter_num / n_iters),
                                                       iter_num, iter_num / n_iters * 100, print_loss_avg, print_loss_val_avg))






    showPlot_val(plot_losses, plot_val_losses, axix)
    showPlot(plot_losses1, bxix)



    torch.save(encoder.state_dict(), 'results/encoder.pth')
    torch.save(decoder.state_dict(), 'results/decoder.pth')


######################################################################
############参数设置
#teacher_forcing_ratio = 0.5



encoder = GRUEncoder(input_size=input_lang.n_words,
                             embedding_size=embedding_size,
                             hidden_size=hidden_size
                             ).to(device)

decoder = GRUDecoder(output_size=output_lang.n_words,
                             embedding_size=embedding_size,
                             hidden_size=hidden_size
                             ).to(device)

trainiters(pairs, encoder, decoder, epoches=n_epoches, plot_every=plot_every,learning_rate=learning_rate)




##############################################################
####################evaluate model

def FlitBack(s): #转换回最初的SMILES分子式，将'_'转换回'\'
    b = re.sub('_', r'\\', s)
    return b

def translate(pair, output):
    print('Source:\t{}\nAnswer:\t{}'.format(FlitBack(pair[0]), FlitBack(pair[1])))
    print('Translate: {}'.format(FlitBack(output)), end='\n\n')
    file_object = open('GRUresults.txt', 'a', encoding='utf-8')
    file_object.writelines(FlitBack(pair[0]) + '\t')
    file_object.writelines(FlitBack(pair[1]) + '\t')
    file_object.writelines(FlitBack(output) + '\n')
    # print('{}|{}'.format(pair[1], output))


def evaluate(sentence, encoder, decoder, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        input_length = input_tensor.size(0)

        encoder_hidden = encoder.initHidden().to(device)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)


        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)

            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]]).to(device)

        decoder_hidden = encoder_hidden


        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

            topv, topi = decoder_output.data.topk(1)  # top-1 value, index


            if topi.item() == EOS_token:

                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

    return decoded_words


def evaluateiters(pairs, encoder, decoder, train_pairs_seed=1):
    start = time.time()
    cc = SmoothingFunction()
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)

    for pi, pair in enumerate(test_pairs):
        output_words = evaluate(pair[0], encoder, decoder)
        output_sentence = ''.join(output_words)

        # for print
        translate(pair, output_sentence)


    print('evaluate successfully')




encoder = GRUEncoder(input_size=input_lang.n_words,
                             embedding_size=embedding_size,
                             hidden_size=hidden_size
                             ).to(device)

decoder = GRUDecoder(output_size=output_lang.n_words,
                             embedding_size=embedding_size,
                             hidden_size=hidden_size
                             ).to(device)



evaluateiters(pairs, encoder, decoder)