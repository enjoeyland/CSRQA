import json
from multiprocessing import Pool
import pickle
import networkx as nx
from tqdm import tqdm

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

def load_resources(cpnet_vocab_path):
    print(f'loading conceptnet vocab from {cpnet_vocab_path}...')
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    print(f'loading conceptnet graph from {cpnet_graph_path}...')
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def generate_adj_data_from_grounded_concepts(grounded_path):
    print(f'generating adj data for {grounded_path}...')
    qa_data = []
    statement_path = grounded_path.replace('grounded', 'statement')
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_ground = fin_ground.readlines()
        lines_state  = fin_state.readlines()
        assert len(lines_ground) % len(lines_state) == 0
        n_choices = len(lines_ground) // len(lines_state)
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            qa_intersect = q_ids & a_ids
            q_ids = q_ids - a_ids
            statement_obj = json.loads(lines_state[j//n_choices])
            QAcontext = "{} {}.".format(statement_obj['question']['stem'], dic['ans'])
            qa_data.append((sorted(q_ids), sorted(a_ids), QAcontext))
    return qa_data

def generate_subgraph(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, extract_method, num_processes):
    # load
    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = generate_adj_data_from_grounded_concepts(grounded_path)

    print(f'finding extra node...')
    with Pool(num_processes) as p:
        extra_nodes__all_pair_2hop_data = list(tqdm(p.imap(extract_method, qa_data), total=len(qa_data)))

    with open(output_path, 'w') as fout:
        for (q_ids,a_ids,QAcontext),extra_nodes in zip(qa_data,extra_nodes__all_pair_2hop_data):
            dic = {
                'extra_c_num':len(extra_nodes),
                'qa_context': QAcontext,
                'qc': [id2concept[q_id] for q_id in q_ids],
                'ac': [id2concept[a_id] for a_id in a_ids],
                'extra_c': [id2concept[en] for en in extra_nodes],
            }
            fout.write(json.dumps(dic) + '\n')
    print(f'subgraph data saved to {output_path}')

    draw_hist(extra_nodes__all_pair_2hop_data)


def draw_hist(data):
    import matplotlib.pyplot as plt
    from collections import Counter

    c = Counter(map(len, data))

    plt.bar(c.keys(), c.values())
    plt.savefig("qa_pair_2hop.png")

def pair_2hop(source_nodes, target_nodes):
    extra_nodes = set()
    for s_id in source_nodes:
        for t_id in target_nodes:
            if s_id != t_id and s_id in cpnet_simple.nodes and t_id in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[s_id]) & set(cpnet_simple[t_id])
    extra_nodes -= set(source_nodes) | set(target_nodes)
    return sorted(extra_nodes)

def all_pair_2hop(data, hop=2):
    qc_ids, ac_ids, _ = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    return pair_2hop(qa_nodes, qa_nodes)

def qa_pair_2hop(data, hop=2):
    qc_ids, ac_ids, _ = data
    return pair_2hop(qc_ids, ac_ids)

def q_pair_2hop(data, hop=2):
    qc_ids, _, _ = data
    return pair_2hop(qc_ids, qc_ids)

def a_pair_2hop(data, hop=2):
    _, ac_ids, _ = data
    return pair_2hop(ac_ids, ac_ids)


