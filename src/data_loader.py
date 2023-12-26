import collections
import os
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set, n_user, n_item = load_rating(args)
    n_entity, n_relation, kg = load_kg(args)
    logging.info("contructing users' kg triple sets ...")
    user_triple_sets = kg_propagation(args, kg, user_init_entity_set, args.user_triple_set_size, True)
    logging.info("contructing items' kg triple sets ...")
    item_triple_sets = kg_propagation(args, kg, item_init_entity_set, args.item_triple_set_size, False)
    return train_data, eval_data, test_data, n_entity, n_relation, user_triple_sets, item_triple_sets, n_user, n_item


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final'
    logging.info("load rating file: %s.npy", rating_file)
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
        np.save(rating_file + '.npy', rating_np)
    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    return dataset_split(rating_np, n_user, n_item)


def dataset_split(rating_np, n_user, n_item):
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    
    user_init_entity_set, item_init_entity_set = collaboration_propagation(rating_np, train_indices)
    
    train_indices = [i for i in train_indices if rating_np[i][0] in user_init_entity_set.keys()]
    eval_indices = [i for i in eval_indices if rating_np[i][0] in user_init_entity_set.keys()]
    test_indices = [i for i in test_indices if rating_np[i][0] in user_init_entity_set.keys()]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]
    
    return train_data, eval_data, test_data, user_init_entity_set, item_init_entity_set, n_user, n_item
    
    
def collaboration_propagation(rating_np, train_indices):
    logging.info("contructing users' initial entity set ...")
    user_history_item_dict = dict()
    item_history_user_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        rating = rating_np[i][2]
        if rating == 1:
            if user not in user_history_item_dict:
                user_history_item_dict[user] = []
            user_history_item_dict[user].append(item)
            if item not in item_history_user_dict:
                item_history_user_dict[item] = []
            item_history_user_dict[item].append(user)
    '''
    user_coll_user_dict = dict()
    for user in user_history_item_dict.keys():
        if user not in user_coll_user_dict:
            user_coll_user_dict[user] = []
        for item in user_history_item_dict[user]:
            for coll_user in item_history_user_dict[item]:
                jiao_length = len(
                    list(set(user_history_item_dict[user]).intersection(set(user_history_item_dict[coll_user]))))
                bing_length = len(list(set(user_history_item_dict[user]).union(set(user_history_item_dict[coll_user]))))
                if bing_length != 0 and jiao_length/bing_length > 0.5 and coll_user not in user_coll_user_dict[user]:
                    user_coll_user_dict[user].append(coll_user)
    for user in user_history_item_dict.keys():
        if len(user_history_item_dict[user]) < 5:
            # print(user, user_history_item_dict[user])
            for coll_user in user_coll_user_dict[user]:
                for item in user_history_item_dict[coll_user]:
                    if item not in user_history_item_dict[user]:
                        user_history_item_dict[user].append(item)
            # print(user, user_history_item_dict[user])
    '''
    '''
    item_coll_item_dict = dict()
    for item in item_history_user_dict.keys():
        item_coll_item_dict[item] = []
        for coll_item in item_history_user_dict.keys():
            jiao_length = len(
                list(set(item_history_user_dict[item]).intersection(set(item_history_user_dict[coll_item]))))
            bing_length = len(list(set(item_history_user_dict[item]).union(set(item_history_user_dict[coll_item]))))
            if item != coll_item and jiao_length / bing_length > 0.2 and coll_item not in item_coll_item_dict[item]:
                item_coll_item_dict[item].append(coll_item)
    for user in user_history_item_dict.keys():
        if len(user_history_item_dict[user]) < 5:
            for item in user_history_item_dict[user]:
                for coll_item in item_coll_item_dict[item]:
                    if coll_item not in user_history_item_dict[user]:
                        user_history_item_dict[user].append(coll_item)
    '''
    item_neighbor_item_dict = dict()
    logging.info("contructing items' initial entity set ...")
    for item in item_history_user_dict.keys():
        item_item = []
        item_nerghbor_item = []
        for user in item_history_user_dict[item]:
            for i in user_history_item_dict[user]:
                if i not in item_item:
                    item_item.append(i)
        for i in item_item:
            jiao_length = len(list(set(item_history_user_dict[item]).intersection(set(item_history_user_dict[i]))))
            bing_length = len(list(set(item_history_user_dict[item]).union(set(item_history_user_dict[i]))))
            if bing_length != 0 and jiao_length / bing_length > 0.5 and i not in item_nerghbor_item:
                item_nerghbor_item.append(i)

        item_neighbor_item_dict[item] = list(set(item_nerghbor_item))
        # print(item, item_neighbor_item_dict[item])

    item_list = set(rating_np[:, 1])
    for item in item_list:
        if item not in item_neighbor_item_dict:
            item_neighbor_item_dict[item] = [item]
    return user_history_item_dict, item_neighbor_item_dict


def load_kg(args):
    kg_file = '../data/' + args.dataset + '/kg_final'
    logging.info("loading kg file: %s.npy", kg_file)
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
        np.save(kg_file + '.npy', kg_np)
    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg = construct_kg(kg_np)
    return n_entity, n_relation, kg


def construct_kg(kg_np):
    logging.info("constructing knowledge graph ...")
    kg = collections.defaultdict(list)
    for head, relation, tail in kg_np:
        kg[head].append((tail, relation))
    return kg


def kg_propagation(args, kg, init_entity_set, set_size, is_user):
    # triple_sets: [n_obj][n_layer](h,r,t)x[set_size] 
    triple_sets = collections.defaultdict(list)
    for obj in init_entity_set.keys():
        if is_user and args.n_layer == 0:
            n_layer = 1
        else:
            n_layer = args.n_layer
        for l in range(n_layer):
            h,r,t = [],[],[]
            if l == 0:
                entities = init_entity_set[obj]
            else:
                entities = triple_sets[obj][-1][2]

            for entity in entities:
                for tail_and_relation in kg[entity]:
                    h.append(entity)
                    t.append(tail_and_relation[0])
                    r.append(tail_and_relation[1])
                    
            if len(h) == 0:
                triple_sets[obj].append(triple_sets[obj][-1])
            else:
                indices = np.random.choice(len(h), size=set_size, replace= (len(h) < set_size))
                h = [h[i] for i in indices]
                r = [r[i] for i in indices]
                t = [t[i] for i in indices]
                triple_sets[obj].append((h, r, t))
    return triple_sets
