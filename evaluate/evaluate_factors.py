import pandas as pd
import numpy as np
import pickle as pkl
from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
import logging
from rdkit.Chem.Crippen import MolLogP
from rdkit import rdBase, Chem
from rdkit.Chem import PandasTools, QED, Descriptors, rdMolDescriptors

#smi = 'CCCc1nn(C)c2C(=O)NC(=Nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4' #sildenafil
#m = Chem.MolFromSmiles(smi)


lines = open('results/0730attnProLSTMresults.txt' ).read().strip().split('\n')

pairs = [[s for s in l.split('\t')] for l in lines]

#打开并读取训练集的
lines_dataset = open('dataset/0512pro_train_dataset.txt' ).read().strip().split('\n')
pairs_dataset = [[s for s in l.split('\t')] for l in lines_dataset]

length_dataset=len(pairs_dataset)
length=len(pairs)
print(length)  # 测试集数量
p=0 # valid num
q = 0
novel_num=0
logP = 0
qed = 0
min_qed = 10000
max_qed = 0
qed_new = 0
qed_dataset=0
logp_dataset=0
dataset_qed_value=[]
dataset_logp_value=[]

sm_all=0 #总的相似度和

jj=0
for ii in range(length_dataset):
    if(Chem.MolFromSmiles(pairs_dataset[ii][0])):
        aa = QED.default(Chem.MolFromSmiles(pairs_dataset[ii][0]))
        qed_dataset=qed_dataset+aa
        dataset_qed_value.append(aa)
        bb_logp=MolLogP(Chem.MolFromSmiles(pairs_dataset[ii][0]))
        logp_dataset = logp_dataset + bb_logp
        dataset_logp_value.append( bb_logp)
        jj = jj+1
    else:
        print(ii)
print(qed_dataset/length_dataset)  # 训练集的平均QED
print(length_dataset)
print(jj) #valid分子个数
print('=============================================')
novel_molecules =[]
valid_molecules=[]
similar_value=[]
qed_value=[]
logp_value=[]

for i in range(length): # 对整个结果文件循环
    n = 0
    if(Chem.MolFromSmiles(pairs[i][2])):
        valid_molecules.append(pairs[i][2])
        logp_a=MolLogP(Chem.MolFromSmiles(pairs[i][2]))
        logP = logP + logp_a
        a = QED.default(Chem.MolFromSmiles(pairs[i][2]))
        qed = qed + a
        qed_value.append(a)
        logp_value.append(logp_a)

        if(a>max_qed):
            max_qed = a
        if(a < min_qed):
            min_qed = a
        p=p+1
        n=n+1
        for j in range(length_dataset):
            if(pairs[i][2]==pairs_dataset[j][0]):
                n=n+1

        if(n==1):
            #novel_num = novel_num+1
            #print(a)
            #qed_new = qed_new + a

            novel_molecules.append(pairs[i][2])

        # 计算相似度
        mols = []
        m1 = Chem.MolFromSmiles(pairs[i][2])
        mols.append(m1)
        m2 = Chem.MolFromSmiles(pairs[i][1])
        mols.append(m2)
        fps = [Chem.RDKFingerprint(x) for x in mols]
        sm01 = DataStructs.FingerprintSimilarity(fps[0], fps[1])
        #print(sm01)
        sm_all = sm_all + sm01
        similar_value.append(sm01)



    if(pairs[i][1]==pairs[i][2]):
        q=q+1

print(q)

unique_novel_molecules=list(set(novel_molecules))
novel_num=len(unique_novel_molecules)

for x in range(novel_num):
    a = QED.default(Chem.MolFromSmiles(unique_novel_molecules[x]))
    qed_new = qed_new + a

unique_valid_molecules=list(set(valid_molecules))
unique_valid_num=len(unique_valid_molecules)

#print(p)
unique = unique_valid_num  / length
valid = p / length
logp = logP / p
QED = qed / p
similar = sm_all / p
acc = q / length
novel = novel_num / length
qed_new = qed_new / novel_num  #  计算生成的valid的新分子的qed均值
similar_std=np.std(similar_value) #  计算similarity标准差
QED_std = np.std(qed_value)  # 计算qed标准差
logp_std=np.std(logp_value) # 计算 logp标准差

print('length of test set: ')
print(length)

print('length of valid set: ')
print(p)

print( 'length of similar set:')
print(len(similar_value))


print('standard deviation of similarity :')
print(similar_std)

print('standard deviation of QED :')
print(QED_std)

#print('number of all the novel molecules:')
#print(len(novel_molecules)) #生成的所有novel分子数量

#print('number of the unique novel molecules:')
#print(novel_num) #去重复后的novel分子数量



#print('Num of unique valid molecules:')
#print(unique_valid_num)

#print('Num of novel molecules:')
#print(novel_num)

print('similarity of valid molecules : ') # 模型生成valid分子与正确分子的平均相似度
print(similar)

print('Valid:')  # 模型生成的valid分子个数/测试集分子数
print(valid)

#print('Accuracy:')  # 模型生成的与目标完整分子完全一致的分子数目/测试集分子数
#print(acc)

#print('Novel:')  # 模型生成的原始数据集中未出现过的有效分子数(去重复后的)/测试集分子数
#print(novel)

#print('Unique:') #模型生成的独一无二的有效分子数/测试集分子数
#print(unique)

print('logP:')  # 计算所有valid分子的logp均值
print(logp)

print('std of logp: ')
print(logp_std)

print('QED:')  # 所有valid分子的qed均值
print(QED)

#print("QED of new molecules:") #计算生成的valid新分子的qed均值
#print (qed_new)

#print('Max QED:')
#print(max_qed)  #所有生成的valid分子中，最大的qed值

#print('Min QED:')
#print(min_qed)  #所有生成的valid分子中，最小的qed值

print('The average QED of training set:')
print(qed_dataset/length_dataset)  # 训练集的平均QED
print(length_dataset)
print('The average logP of training set:')
print(logp_dataset/length_dataset)  # 训练集的平均QED

print('std of dataset QED:')
print(np.std(dataset_qed_value))

print('std of dataset logp:')
print(np.std(dataset_logp_value))


