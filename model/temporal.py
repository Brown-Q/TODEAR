import numpy as np
import math
import operator
import json


def dataloader(file,entity_file, relation_file, time_file,valid_file, test_file):
    entity_dict = {}
    relation_dict = {}
    time_dict = {}
    with open(entity_file, 'r', encoding='utf-8') as e_f:
        lines = e_f.readlines()
        for line in lines:
            entity, embedding = line.strip().split('\t')
            embedding = np.array(json.loads(embedding))
            entity_dict[entity] = embedding

    with open(relation_file, 'r', encoding='utf-8') as r_f:
        lines = r_f.readlines()
        for line in lines:
            relation, embedding = line.strip().split('\t')
            embedding = np.array(json.loads(embedding))
            relation_dict[relation] = embedding

    with open(time_file, 'r', encoding='utf-8') as t_f:
        lines = t_f.readlines()
        for line in lines:
            time, embedding = line.strip().split('\t')
            embedding = np.array(json.loads(embedding))
            time_dict[time] = embedding

    with open(test_file, 'r', encoding='utf-8') as test_f:
        lines = test_f.readlines()
        for line in lines:
            quadruple = line.strip().split('\t')

            s_ = quadruple[0]
            if s_ not in entity_dict:
                entity_dict[s_] = np.random.uniform(-6 / math.sqrt(80), 6 / math.sqrt(80),80)
            r_ = quadruple[1]
            if r_ not in relation_dict:
                relation_dict[r_] = np.random.uniform(-6 / math.sqrt(100), 6 / math.sqrt(100), 100)
            o_ = quadruple[2]
            if o_ not in entity_dict:
                entity_dict[o_] = np.random.uniform(-6 / math.sqrt(80), 6 / math.sqrt(80), 80)
            t_ = quadruple[3]
            if t_ not in time_dict:
                time_dict[t_] = np.random.uniform(-6 / math.sqrt(20), 6 / math.sqrt(20), 20)
                
    with open(valid_file, 'r', encoding='utf-8') as test_f:
        lines = test_f.readlines()
        for line in lines:
            quadruple = line.strip().split('\t')

            s_ = quadruple[0]
            if s_ not in entity_dict:
                entity_dict[s_] = np.random.uniform(-6 / math.sqrt(80), 6 / math.sqrt(80),80)
            r_ = quadruple[1]
            if r_ not in relation_dict:
                relation_dict[r_] = np.random.uniform(-6 / math.sqrt(100), 6 / math.sqrt(100), 100)
            o_ = quadruple[2]
            if o_ not in entity_dict:
                entity_dict[o_] = np.random.uniform(-6 / math.sqrt(80), 6 / math.sqrt(80), 80)
            t_ = quadruple[3]
            if t_ not in time_dict:
                time_dict[t_] = np.random.uniform(-6 / math.sqrt(20), 6 / math.sqrt(20), 20)
                
    entity_dict=dict(sorted(entity_dict.items(),key = lambda x:int(x[0])))
    relation_dict=dict(sorted(relation_dict.items(),key = lambda x:int(x[0])))
    time_dict=dict(sorted(time_dict.items(),key = lambda x:int(x[0])))
    i = 0
    with open(file+"res/entity_emb.txt", 'w', encoding='utf-8') as f_e:
        for e in entity_dict.keys():
            f_e.write(str(e) + "\t")
            f_e.write(str(list(entity_dict[e])))
            f_e.write("\n")
        for j in range(500):
            padding = np.random.uniform(-6 / math.sqrt(80), 6 / math.sqrt(80), 80) 
            f_e.write(str(int(e)+1) + "\t")
            f_e.write(str(list(padding)))
            f_e.write("\n")
    with open(file+"res/relation_emb.txt", 'w', encoding='utf-8') as f_r:
        for r in relation_dict.keys():
            if i == 24:
                padding = np.random.uniform(-6 / math.sqrt(100), 6 / math.sqrt(100), 100) 
                f_r.write(str(int(r)+1) + "\t")
                f_r.write(str(list(padding)))
                f_r.write("\n") 
            else:
                f_r.write(str(r) + "\t")
                f_r.write(str(list(relation_dict[r])))
                f_r.write("\n")
            i = i+1
        for j in range(500):
            padding = np.random.uniform(-6 / math.sqrt(100), 6 / math.sqrt(100), 100) 
            f_r.write(str(int(j)+1) + "\t")
            f_r.write(str(list(padding)))
            f_r.write("\n") 
    with open(file+"res/time_emb.txt", 'w', encoding='utf-8') as f_t:
        for t in time_dict.keys():
            f_t.write(str(t) + "\t")
            f_t.write(str(list(time_dict[t])))
            f_t.write("\n") 
        for j in range(500):      
            padding = np.random.uniform(-6 / math.sqrt(20), 6 / math.sqrt(20), 20) 
            f_t.write(str(int(t)+1) + "\t")
            f_t.write(str(list(padding)))
            f_t.write("\n")    
    # embeddings = np.empty(shape=[0, 80])       
    # with open(file+"res/entity_emb.txt", 'r', encoding='utf-8') as e_f:
    #     lines = e_f.readlines()
    #     for line in lines:
    #         entity, embedding = line.strip().split('\t')
    #         embedding = np.array([json.loads(embedding)])
    #         embeddings = np.concatenate([embeddings,embedding],axis=0)
    # print(embeddings)               
    return entity_dict, relation_dict, time_dict           
                
    
            
if __name__ == '__main__':
    file = "../data/ICEWS14/"
    _, _, train_quadruple = dataloader(file,file+"res/entity_50dim_batch400.txt",file+"res/relation_50dim_batch400.txt",file+"res/time_50dim_batch400.txt",file+"valid.txt",file+"test.txt")
  
