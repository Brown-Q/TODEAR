import random
import math
import numpy as np
import copy
import time
import torch


def data_loader(file):
    file_train = file + "train.txt"
    entity_set = set()   
    relation_set = set()
    time_set = set()
    quadruple_list = []
    with open(file_train, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            # if len(quadruple) != 4:
            #     continue
            # e.g. South Korea  Criticize or denounce   North Korea 2014-05-13
            s_ = int(quadruple[0])
            r_ = int(quadruple[1])
            o_ = int(quadruple[2])
            t_ = int(quadruple[3]) // 24
            quadruple_list.append([s_, r_, o_, t_])  
            entity_set.add(s_)
            relation_set.add(r_)
            entity_set.add(o_)
            time_set.add(t_)
    rel_num = len(relation_set)
    with open(file_train, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            s_ = int(quadruple[2])
            r_ = int(quadruple[1]) + rel_num
            o_ = int(quadruple[0])
            t_ = int(quadruple[3]) // 24
            quadruple_list.append([s_, r_, o_, t_]) 
            entity_set.add(s_)
            relation_set.add(r_)
            entity_set.add(o_)
            time_set.add(t_)    
            
    return list(entity_set), list(relation_set), list(time_set), quadruple_list


def distanceL2(s, r, o, t):
    return torch.sum(torch.square(torch.mul(s,t)+ r  - torch.mul(o,t))).cuda()

def distanceL1(s, r, o, t):
    return torch.sum(torch.abs(torch.mul(s,t) + r  - torch.mul(o,t))).cuda()

class TTransE:
    def __init__(self, entity_set, relation_set, time_set, quadruple_list,
                 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.entity = entity_set
        self.relation = relation_set
        self.time = time_set
        self.quadruple_list = quadruple_list
        self.L1 = L1
        self.loss = 0

    def emb_initialize(self):
        self.relation = torch.randn(len(relation_set), 100).cuda()
        self.entity = torch.randn(len(entity_set), 100).cuda()
        self.time = torch.rand(len(time_set), 100).cuda()

    def train(self, file, epochs):
        nbatches = 400
        batch_size = len(self.quadruple_list) // nbatches
        print("batch size: ", batch_size)
        quadruple_num = len(entity_set)
        for epoch in range(0,epochs):
            self.loss = 0
            for k in range(0, nbatches-1):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    Sbatch = self.quadruple_list[k*batch_size:(k+1)*batch_size]
                    Tbatch = []
                    for quadruple in Sbatch:
                        corrupted_quadruple = self.Corrupt(quadruple, quadruple_num)
                        Tbatch.append([quadruple, corrupted_quadruple])
                    Tbatch = torch.tensor(Tbatch)
                    self.update_embeddings(Tbatch)
            print("epoch: ", epoch)
            print("loss: ", self.loss)

        print("写入文件...")
        with open(file+"res/entity_50dim_batch400.txt", 'w', encoding='utf-8') as f1:
            self.entity = self.entity.tolist()
            for i, j in enumerate(self.entity):
                f1.write(str(i) + "\t")
                f1.write(str(j))
                f1.write("\n")
        with open(file+"res/relation_50dim_batch400.txt", 'w', encoding='utf-8') as f2:
            self.relation = self.relation.tolist()
            for i, j in enumerate(self.relation):
                f2.write(str(i) + "\t")
                f2.write(str(j))
                f2.write("\n")
        with open(file+"res/time_50dim_batch400.txt", 'w', encoding='utf-8') as f3:
            self.time = self.time.tolist()
            for i, j in enumerate(self.time):
                f3.write(str(i) + "\t")
                f3.write(str(j))
                f3.write("\n")
        print("写入完成")

    def Corrupt(self, quadruple, quadruple_num):
        corrupted_quadruple = copy.deepcopy(quadruple)
        seed = random.random()
        if seed > 0.5:
        # print(seed)
            head = quadruple[0]
            rand_head = head
            if rand_head == head:
                rand_head = entity_set[int(seed*quadruple_num)]
            corrupted_quadruple[0] = rand_head
        else:
            tail = quadruple[2]
            rand_tail = tail
            if rand_tail == tail:
                rand_tail = entity_set[int(seed*quadruple_num)] 
            corrupted_quadruple[2] = rand_tail
       
        return corrupted_quadruple

    def update_embeddings(self, Tbatch):
        s_correct = torch.nn.functional.normalize(self.entity[Tbatch[:,0][:,0]], p=2.0, dim=1)
        o_correct = torch.nn.functional.normalize(self.entity[Tbatch[:,0][:,2]], p=2.0, dim=1)
        s_corrupt = torch.nn.functional.normalize(self.entity[Tbatch[:,1][:,0]], p=2.0, dim=1)
        o_corrupt = torch.nn.functional.normalize(self.entity[Tbatch[:,1][:,2]], p=2.0, dim=1)
        # print(o_corrupt)
        relation = torch.nn.functional.normalize(self.relation[Tbatch[:,0][:,1]], p=2.0, dim=1)
        time = torch.nn.functional.normalize(self.time[Tbatch[:,0][:,3]], p=2.0, dim=1)
        if self.L1:
            dist_correct = distanceL1(s_correct, relation, o_correct, time)
            dist_corrupt = distanceL1(s_corrupt, relation, o_corrupt, time)
        else:
            dist_correct = distanceL2(s_correct, relation, o_correct, time)
            dist_corrupt = distanceL2(s_corrupt, relation, o_corrupt, time)
        err = self.hinge_loss(dist_correct, dist_corrupt)

        if err > 0:
            self.loss += err
            grad_pos = 2 * torch.nn.functional.normalize((torch.mul(s_correct,time) + relation  - torch.mul(o_correct,time)),p=2.0, dim=1)
            grad_neg = 2 * torch.nn.functional.normalize((torch.mul(s_corrupt,time) + relation  - torch.mul(o_corrupt,time)),p=2.0, dim=1)
            self.entity[Tbatch[:,0][:,0]] -= self.learning_rate * grad_pos
            self.entity[Tbatch[:,0][:,2]] -= (-1) * self.learning_rate * grad_pos
            self.entity[Tbatch[:,1][:,0]] -= (-1) * self.learning_rate * grad_neg
            self.entity[Tbatch[:,1][:,2]] -= self.learning_rate * grad_neg
            self.relation[Tbatch[:,0][:,1]] -= self.learning_rate * grad_pos
            self.relation[Tbatch[:,0][:,1]] -= (-1) * self.learning_rate * grad_neg
            self.time[Tbatch[:,0][:,3]] -= self.learning_rate * grad_pos
            self.time[Tbatch[:,0][:,3]] -= (-1) * self.learning_rate * grad_neg

    def hinge_loss(self, dist_correct, dist_corrupt):
        return torch.max(torch.tensor(0), dist_correct - dist_corrupt + self.margin)

if __name__ == '__main__':
    file = "../data/ICEWS14/"
    print("load file...")
    entity_set, relation_set, time_set, quadruple_list = data_loader(file)
    print("Complete load. entity : %d , relation : %d , time : %d , quadruple : %d" % (len(entity_set), len(relation_set), len(time_set), len(quadruple_list)))
    TTransE = TTransE(entity_set, relation_set, time_set, quadruple_list, embedding_dim=80, learning_rate = 0.001, margin = 1, L1 = True)
    TTransE.emb_initialize()
    TTransE.train(file, epochs = 200)
