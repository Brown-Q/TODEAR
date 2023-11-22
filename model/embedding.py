import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.datasets import Planetoid
import json
import torch.nn as nn
import random
import math
import numpy as np
import copy
import time
import tqdm
import apex

def data_loader(file):
    file_train = file + "train.txt"
    file_name_train = file + "entity2id.txt"
    entity_set = set()  
    name_set = set()
    relation_set = set()
    time_set = set()
    quadruple_list = []
    with open(file_name_train, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            s_ = quadruple[0]
            name_set.add(s_)
    with open(file_train, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            # if len(quadruple) != 4:
            #     continue
            # e.g. South Korea  Criticize or denounce   North Korea 2014-05-13
            if quadruple[0].isdigit():
                s_ = int(quadruple[0])
                r_ = int(quadruple[1])
                o_ = int(quadruple[2])
                t_ = int(quadruple[3])
            quadruple_list.append([s_, r_, o_, t_])  
            entity_set.add(s_)
            relation_set.add(int(r_))
            entity_set.add(o_)
            time_set.add(int(t_))
    rel_num = 600
    with open(file_train, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            if len(quadruple) == 5:
                s_ = int(quadruple[2])
                r_ = int(quadruple[1]) + rel_num
                o_ = int(quadruple[0])
                t_ = int(quadruple[3])
            quadruple_list.append([s_, r_, o_, t_])  
            entity_set.add(s_)
            relation_set.add(int(r_))
            entity_set.add(o_)
            time_set.add(int(t_))    
    # print(entity_set)  
    relation_set = list(relation_set)  
    time_set = list(time_set)
    time_set = sorted(time_set)
    return entity_set, relation_set, time_set, quadruple_list, name_set

class GCN(torch.nn.Module):
    def __init__(self,feature, hidden, classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature, hidden)
        self.conv2 = GCNConv(hidden, classes)
        self.mlp_l1 = nn.Linear(160, 100, bias=True)
        self.mlp_l2 = nn.Linear(100, 100, bias=True)
        self.mlp_l3 = nn.Linear(20, 20, bias=True)
        self.mlp_l4 = nn.Linear(20, 20, bias=True)
        self.mlp_l5 = nn.Linear(100, 100, bias=True)
    def forward(self, features, edges,edge_triple,relations,relation_embedding,time_embedding):
        features = self.conv1(features, edges)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.conv2(features, edges)
        for i in relations:
            id = torch.where(edge_triple[:,1] == i)
            relation_embedding[i,:] = self.mlp_l2(self.mlp_l1(torch.cat((torch.mean(features[edge_triple[:,0][id]],dim=0), torch.mean(features[edge_triple[:,2][id]],dim=0)))))
        relation_embedding_new = self.mlp_l5(relation_embedding).detach()
        time_embedding = self.mlp_l4(self.mlp_l3(time_embedding))
        features = F.normalize(features, dim=1, p=2)
        relation_embedding_new = F.normalize(relation_embedding_new, dim=1, p=2)
        time_embedding = F.normalize(time_embedding, dim=1, p=2)
        return features, relation_embedding_new, time_embedding

def distanceL2(s, r, o, t):
    return torch.sum(torch.abs(torch.cat((s,2*t),1)+r-torch.cat((o,t),1)),dim=-1)

class TTransE:
    def __init__(self, entity_set, relation_set, time_set, quadruple_list, name_set,file,hour,
                 embedding_dim=80, learning_rate=0.01, margin=1):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.time = time_set
        self.name = name_set
        self.quadruple_list = quadruple_list
        self.file = file
        self.hour = hour
        self.loss = 0
        self.embs = {}
        with open("../data/glove.6B.100d.txt") as f:
            for line in tqdm.tqdm(f.readlines()):
                line = line.strip().split()
                self.embs[line[0]] = torch.tensor([float(x) for x in line[1:]])

    def emb_initialize(self):
        entity_dict = {}
        x = []
        for name in self.name:
            ent_x = []
            for word in name.split():
                    word = word.lower()
                    if word in self.embs.keys():
                        ent_x.append(self.embs[word][:80])
            if len(ent_x) > 0:
                x.append(torch.stack(ent_x, dim=0).mean(dim=0))
            else:
                x.append(torch.rand(80))
        x = torch.stack(x, dim=0).contiguous()
        x = F.normalize(x, dim=1, p=2)
        for entity in self.entity:
            e_emb_temp = torch.randn(80)
            entity_dict[entity] = F.normalize(e_emb_temp, dim=0, p=2)
        self.embedding = x
        self.entity = entity_dict
        
    def Corrupt(self, quadruple):
        corrupted_quadruple = copy.deepcopy(quadruple)
        seed = random.random()
        if seed > 0.5:
            head = quadruple[0]
            rand_head = head
            while (rand_head == head):
                rand_head = random.sample(self.entity.keys(), 1)[0] 
            corrupted_quadruple[0] = rand_head
        else:
            tail = quadruple[2]
            rand_tail = tail
            while (rand_tail == tail):
                rand_tail = random.sample(self.entity.keys(), 1)[0] 
            corrupted_quadruple[2] = rand_tail
        return corrupted_quadruple

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, torch.mean(F.relu(dist_correct - dist_corrupt + self.margin)))    
   
    def train(self, file,epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        quadruple_list = np.array(self.quadruple_list)
        Tbatch = []
        for quadruple in quadruple_list:
            corrupted_quadruple = self.Corrupt(quadruple)
            Tbatch.append(corrupted_quadruple)
        Tbatch = np.array(Tbatch)
        relation = torch.tensor(self.relation,dtype=torch.int64).to(device)
        relation_num = torch.max(relation)
        entity_embedding = self.embedding.to(device).requires_grad_()
        relation_embedding = torch.randn(relation_num.item()+1,100)
        relation_embedding = F.normalize(relation_embedding, dim=1, p=2).to(device)
        file_test = file + "test.txt"
        with open(file_test, 'r', encoding='utf-8') as f:
            content = f.readlines()
            time_num = int(content[-1].strip().split()[-2])//self.hour
            time_embedding = torch.randn(time_num+1,20)
            time_embedding = F.normalize(time_embedding, dim=1, p=2).to(device).requires_grad_()
        edges=[]
        edge_triple = []
        for strs in quadruple_list:
            edges.append([int(strs[0]),int(strs[2])])
            edge_triple.append([int(strs[0]),int(strs[1]),int(strs[2])])
        edge_triple = torch.tensor(edge_triple,dtype=torch.int64).to(device)
        edges = torch.tensor(edges, dtype=torch.int64).T.to(device)
        model = GCN(80, 80, 80).to(device)
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)                                            
        model.train()
        for epoch in range(epochs):
            start = time.time()
            entity_embedding1, relation_embedding1, time_embedding1 = model(entity_embedding,edges,edge_triple,relation,relation_embedding,time_embedding) 
            loss = self.hinge_loss(distanceL2(entity_embedding1[quadruple_list[:,0],:],relation_embedding1[quadruple_list[:,1],:],entity_embedding1[quadruple_list[:,2],:],time_embedding1[quadruple_list[:,3]//self.hour,:]),
                                  distanceL2(entity_embedding1[Tbatch[:,0],:],relation_embedding1[Tbatch[:,1],:],entity_embedding1[Tbatch[:,2],:],time_embedding1[Tbatch[:,3]//self.hour,:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", loss)
        print("写入文件...")
        entity_embedding1 = entity_embedding1.cpu().detach().numpy()
        relation_embedding1 = relation_embedding1.cpu().detach().numpy()
        time_embedding1 = time_embedding1.cpu().detach().numpy()
        with open(file+"res/entity_50dim_batch400.txt", 'w', encoding='utf-8') as f1:
            for e in self.entity.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(entity_embedding1[int(e),:])))
                f1.write("\n")
        with open(file+"res/relation_50dim_batch400.txt", 'w', encoding='utf-8') as f2:
            for r in self.relation:
                f2.write(str(r) + "\t")
                f2.write(str(list(relation_embedding1[r,:])))
                f2.write("\n")
        with open(file+"res/time_50dim_batch400.txt", 'w', encoding='utf-8') as f3:
            for t in self.time:
                f3.write(str(t) + "\t")
                f3.write(str(list(time_embedding1[t//self.hour,:])))
                f3.write("\n")
        print("写入完成")

    
if __name__ == '__main__':
    file = "../data/ICEWS14/"
    print("load file...")
    entity_set, relation_set, time_set, quadruple_list, name_set = data_loader(file)
    hour = 24
    print("Complete load. entity : %d , relation : %d , time : %d , quadruple : %d" % (len(entity_set), len(relation_set), len(time_set), len(quadruple_list)))
    TTransE = TTransE(entity_set, relation_set, time_set, quadruple_list, name_set,file,hour, embedding_dim=80 ,learning_rate=0.001, margin=5)
    TTransE.emb_initialize()
    TTransE.train(file,epochs=1100)
