import random
import math
import numpy as np
import copy
import time


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
            s_ = quadruple[0]
            r_ = quadruple[1]
            o_ = quadruple[2]
            t_ = quadruple[3]

            quadruple_list.append([s_, r_, o_, t_])  

            entity_set.add(s_)
            relation_set.add(r_)
            entity_set.add(o_)
            time_set.add(t_)
    rel_num = 600
    with open(file_train, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            quadruple = line.strip().split("\t")
            s_ = quadruple[2]
            r_ = str(int(quadruple[1]) + rel_num)
            o_ = quadruple[0]
            t_ = quadruple[3]

            quadruple_list.append([s_, r_, o_, t_]) 

            entity_set.add(s_)
            relation_set.add(r_)
            entity_set.add(o_)
            time_set.add(t_)    
        
        
    return entity_set, relation_set, time_set, quadruple_list


def distanceL2(s, r, o, t):
    return np.sum(np.square(np.concatenate((s,t),axis=0) + r  - np.concatenate((o,t),axis=0)))


def distanceL1(s, r, o, t):
    return np.sum(np.fabs(np.concatenate((s,t),axis=0) + r  - np.concatenate((o,t),axis=0)))


class TTransE:
    def __init__(self, entity_set, relation_set, time_set, quadruple_list,
                 embedding_dim=80, learning_rate=0.01, margin=1, L1=True):
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
        relation_dict = {}
        entity_dict = {}
        time_dict = {}
        for relation in self.relation:
            r_emb_temp = np.random.uniform(-6 / math.sqrt(100),
                                           6 / math.sqrt(100),
                                           100)
            relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

        for entity in self.entity:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(80),
                                           6 / math.sqrt(80),
                                           80)
            entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

        for time in self.time:
            e_emb_temp = np.random.uniform(-6 / math.sqrt(20),
                                           6 / math.sqrt(20),
                                           20)
            time_dict[time] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)
        
        
        self.relation = relation_dict
        self.entity = entity_dict
        self.time = time_dict

    def train(self, file,epochs):
        nbatches = 400
        batch_size = len(self.quadruple_list) // nbatches
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0

            for k in range(nbatches):
                Sbatch = random.sample(self.quadruple_list, batch_size)
                Tbatch = []
                for quadruple in Sbatch:
                    corrupted_quadruple = self.Corrupt(quadruple)
                    Tbatch.append((quadruple, corrupted_quadruple))

                self.update_embeddings(Tbatch)

            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("loss: ", self.loss)

            if epoch % 10 == 0:
                with open(file+"res/entity_temp.txt", 'w', encoding='utf-8') as f_e:
                    for e in self.entity.keys():
                        f_e.write(str(e) + "\t")
                        f_e.write(str(list(self.entity[e])))
                        f_e.write("\n")
                with open(file+"res/relation_temp.txt", 'w', encoding='utf-8') as f_r:
                    for r in self.relation.keys():
                        f_r.write(str(r) + "\t")
                        f_r.write(str(list(self.relation[r])))
                        f_r.write("\n")
                with open(file+"res/time_temp.txt", 'w', encoding='utf-8') as f_t:
                    for t in self.time.keys():
                        f_t.write(str(t) + "\t")
                        f_t.write(str(list(self.time[t])))
                        f_t.write("\n")
                with open(file+"res/result_temp.txt", 'a', encoding='utf-8') as f_s:
                    f_s.write("epoch: %d\tloss: %s\n" % (epoch, self.loss))

        print("写入文件...")
        with open(file+"res/entity_50dim_batch400.txt", 'w', encoding='utf-8') as f1:
            for e in self.entity.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entity[e])))
                f1.write("\n")

        with open(file+"res/relation_50dim_batch400.txt", 'w', encoding='utf-8') as f2:
            for r in self.relation.keys():
                f2.write(str(r) + "\t")
                f2.write(str(list(self.relation[r])))
                f2.write("\n")
        with open(file+"res/time_50dim_batch400.txt", 'w', encoding='utf-8') as f3:
            for t in self.time.keys():
                f3.write(str(t) + "\t")
                f3.write(str(list(self.time[t])))
                f3.write("\n")
        print("写入完成")

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

    def update_embeddings(self, Tbatch):
        entity_updated = {}
        relation_updated = {}
        time_updated = {}
        for quadruple, corrupted_quadruple in Tbatch:
            s_correct = self.entity[quadruple[0]]
            o_correct = self.entity[quadruple[2]]
            s_corrupt = self.entity[corrupted_quadruple[0]]
            o_corrupt = self.entity[corrupted_quadruple[2]]
            # print(o_corrupt)
            relation = self.relation[quadruple[1]]
            time = self.time[quadruple[3]]


            if quadruple[0] in entity_updated.keys():
                pass
            else:
                entity_updated[quadruple[0]] = copy.copy(self.entity[quadruple[0]])
            if quadruple[2] in entity_updated.keys():
                pass
            else:
                entity_updated[quadruple[2]] = copy.copy(self.entity[quadruple[2]])
            if corrupted_quadruple[0] in entity_updated.keys():
                pass
            else:
                entity_updated[corrupted_quadruple[0]] = copy.copy(self.entity[corrupted_quadruple[0]])
            if corrupted_quadruple[2] in entity_updated.keys():
                pass
            else:
                entity_updated[corrupted_quadruple[2]] = copy.copy(self.entity[corrupted_quadruple[2]])
            if quadruple[1] in relation_updated.keys():
                pass
            else:
                relation_updated[quadruple[1]] = copy.copy(self.relation[quadruple[1]])
            if quadruple[3] in time_updated.keys():
                pass
            else:
                time_updated[quadruple[3]] = copy.copy(self.time[quadruple[3]])


            if self.L1:
                dist_correct = distanceL1(s_correct, relation, o_correct, time)
                dist_corrupt = distanceL1(s_corrupt, relation, o_corrupt, time)
            else:
                dist_correct = distanceL2(s_correct, relation, o_correct, time)
                dist_corrupt = distanceL2(s_corrupt, relation, o_corrupt, time)

            err = self.hinge_loss(dist_correct, dist_corrupt)

            if err > 0:
                self.loss += err
                grad_pos = 2 * (np.concatenate((s_correct,time),axis=0) + relation  - np.concatenate((o_correct,time),axis=0))
                grad_neg = 2 * (np.concatenate((s_corrupt,time),axis=0) + relation  - np.concatenate((o_corrupt,time),axis=0))
                if self.L1:
                    for i in range(len(grad_pos)):
                        if (grad_pos[i] > 0):
                            grad_pos[i] = 1
                        else:
                            grad_pos[i] = -1

                    for i in range(len(grad_neg)):
                        if (grad_neg[i] > 0):
                            grad_neg[i] = 1
                        else:
                            grad_neg[i] = -1

                entity_updated[quadruple[0]] -= self.learning_rate * grad_pos[:80]
                entity_updated[quadruple[2]] -= (-1) * self.learning_rate * grad_pos[:80]

                entity_updated[corrupted_quadruple[0]] -= (-1) * self.learning_rate * grad_neg[:80]
                entity_updated[corrupted_quadruple[2]] -= self.learning_rate * grad_neg[:80]

                relation_updated[quadruple[1]] -= self.learning_rate * grad_pos
                relation_updated[quadruple[1]] -= (-1) * self.learning_rate * grad_neg

                time_updated[quadruple[3]] -= self.learning_rate * grad_pos[80:]
                time_updated[quadruple[3]] -= (-1) * self.learning_rate * grad_neg[80:]


        # batch norm
        for i in entity_updated.keys():
            entity_updated[i] /= np.linalg.norm(entity_updated[i])
            self.entity[i] = entity_updated[i]
        for i in relation_updated.keys():
            relation_updated[i] /= np.linalg.norm(relation_updated[i])
            self.relation[i] = relation_updated[i]
        for i in time_updated.keys():
            time_updated[i] /= np.linalg.norm(time_updated[i])
            time_updated[i] /= np.linalg.norm(time_updated[i])
            self.time[i] = time_updated[i]
        return

    def hinge_loss(self, dist_correct, dist_corrupt):
        return max(0, dist_correct - dist_corrupt + self.margin)

if __name__ == '__main__':
    file = "../data/ICEWS14/"
    print("load file...")
    entity_set, relation_set, time_set, quadruple_list = data_loader(file)
    print("Complete load. entity : %d , relation : %d , time : %d , quadruple : %d" % (len(entity_set), len(relation_set), len(time_set), len(quadruple_list)))
    TTransE = TTransE(entity_set, relation_set, time_set, quadruple_list, embedding_dim=80 ,learning_rate=0.01, margin=1, L1=True)
    TTransE.emb_initialize()
    TTransE.train(file,epochs=20)