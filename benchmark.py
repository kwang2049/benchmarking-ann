import json
import faiss
import pickle
import os
import tqdm
import numpy as np
import argparse
import time
from functools import wraps

parser = argparse.ArgumentParser()
parser.add_argument('--d', type=int, default=768, help='dimension size')
parser.add_argument('--buffer_size', type=int, default=50000)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128, help='for retrieval')
parser.add_argument('--embedded_dir', type=str, default='msmarco-embedded')
parser.add_argument('--output_dir', type=str, default='msmarco-benchmarking')
parser.add_argument('--eval_string', type=str, required=True, help='e.g. pq(384, 8)')
args_cli = parser.parse_args()

path_doc_embedding = os.path.join(args_cli.embedded_dir, 'embeddings.documents.pkl')
path_query_embedding = os.path.join(args_cli.embedded_dir, 'embeddings.queries.pkl')
path_ids = os.path.join(args_cli.embedded_dir, 'ids.txt')
path_qrels = os.path.join(args_cli.embedded_dir, 'qrels.json')

print('>>> Loading query embeddings')
with open(path_query_embedding, 'rb') as f:
    queries = pickle.load(f)

print('>>> Loading qrels')
with open(path_qrels, 'r') as f:
    qrels = json.load(f)

print('>>> Loading document embeddings')
with open(path_doc_embedding, 'rb') as f:
    xb = pickle.load(f)

print('>>> Loading ids')
with open(path_ids, 'r') as f:
    ids = []
    for line in f:
        ids.append(line.strip())

os.makedirs(args_cli.output_dir, exist_ok=True)

def faiss_wrapper(indexing_setup_func):
    @wraps(indexing_setup_func)
    def wrapped_function(*args, **kwargs):
        index, index_name = indexing_setup_func(*args, **kwargs)
        loaded = False
        index_path = os.path.join(args_cli.output_dir, index_name)
        if not os.path.exists(index_path):
            print(f'>>> Doing training for {index_path}')
            for _ in tqdm.trange(1):
                index.train(xb)
            print(f'>>> Adding embeddings to {index_path}')
            for start in tqdm.trange(0, len(xb), args_cli.buffer_size):
                index.add(xb[start : start + args_cli.buffer_size])
            faiss.write_index(index, index_path)
        else:
            index = faiss.read_index(index_path)
            loaded = True
        return index, index_name, loaded
    return wrapped_function


######################## All the candidate methods ########################
@faiss_wrapper
def flat():
    index = faiss.IndexFlatIP(args_cli.d)
    index_name = 'flat.index'
    return index, index_name

@faiss_wrapper
def flat_sq(qname):
    assert qname in dir(faiss.ScalarQuantizer)  # QT_fp16, QT_8bit_uniform, QT_4bit_uniform ...
    index_name = f'flat-{qname}.index'
    qtype = getattr(faiss.ScalarQuantizer, qname)
    index = faiss.IndexScalarQuantizer(args_cli.d, qtype, faiss.METRIC_INNER_PRODUCT)
    return index, index_name

@faiss_wrapper
def flat_pcq_sq(qname, d_target=args_cli.d // 2):
    assert qname in dir(faiss.ScalarQuantizer)  # QT_fp16, QT_8bit_uniform, QT_4bit_uniform ...
    index_name = f'flat-{qname}.index'
    qtype = getattr(faiss.ScalarQuantizer, qname)
    index = faiss.IndexScalarQuantizer(args_cli.d, qtype, faiss.METRIC_INNER_PRODUCT)
    ################
    index_name = index_name.replace('flat-', 'flat-pca-')
    pca_matrix = faiss.PCAMatrix(args_cli.d, d_target, 0, True) 
    index = faiss.IndexPreTransform(pca_matrix, index)
    return index, index_name

@faiss_wrapper
def flat_ivf(qname, nlist, nprobe):
    assert qname in dir(faiss.ScalarQuantizer)  # QT_fp16, QT_8bit_uniform, QT_4bit_uniform ...
    index_name = f'flat-{qname}.index'
    qtype = getattr(faiss.ScalarQuantizer, qname)
    ################
    index_name = index_name.replace('flat-', 'flat-ivf-')
    quantizer = faiss.IndexFlatIP(args_cli.d)
    index = faiss.IndexIVFScalarQuantizer(quantizer, args_cli.d, nlist, qtype, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe
    return index, index_name
    
@faiss_wrapper
def pq(m, nbits):
    # m: How many chunks for splitting each vector
    # nbits: How many clusters (2 ** nbits) for each chunked vectors 
    assert args_cli.d % m == 0
    index_name = f'pq-{m}-{nbits}b.index'
    index = faiss.IndexPQ(args_cli.d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    return index, index_name

@faiss_wrapper
def opq(m, nbits):
    assert args_cli.d % m == 0
    index_name = f'pq-{m}-{nbits}b.index'
    index = faiss.IndexPQ(args_cli.d, m, nbits, faiss.METRIC_INNER_PRODUCT)
    ################
    index_name = index_name.replace('pq-', 'opq-')
    opq_matrix = faiss.OPQMatrix(args_cli.d, m)
    index = faiss.IndexPreTransform(opq_matrix, index)
    return index, index_name

@faiss_wrapper
def hnsw(store_n, ef_search, ef_construction):
    index_name = f'hnsw-{store_n}-{ef_search}-{ef_construction}.index'
    index = faiss.IndexHNSWFlat(args_cli.d, store_n, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.hnsw.efConstruction = ef_construction
    return index, index_name
##########################################################################


def mrr(index):
    mrr = 0
    qids = list(qrels.keys())
    print('>>> Doing retrieval')
    for start in tqdm.trange(0, len(qrels), args_cli.batch_size):
        qid_batch = qids[start : start + args_cli.batch_size]
        qembs = np.vstack([queries[qid] for qid in qid_batch])
        _, I = index.search(qembs, args_cli.topk)  # (batch_size, topk)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                qid = qid_batch[i]
                did = ids[I[i, j]]  # The ids returned by FAISS are just positions!!!
                if did in qrels[qid]:
                    mrr += 1.0 / (j + 1)
                    break
    return mrr / len(qrels)

results = {}
results['batch size'] = args_cli.batch_size
results['eval_string'] = args_cli.eval_string

start = time.time()
index, index_name, loaded = eval(args_cli.eval_string)
end = time.time()
results['indexing (s)'] = end - start
if loaded:
    results['indexing (s)'] = None
results['size (GB)'] = os.path.getsize(os.path.join(args_cli.output_dir, index_name)) / 1024 ** 3

start = time.time()
_mrr = mrr(index)
end = time.time()
results['retrieval (s)'] = end - start
results['per query (s)'] = (end - start) / len(qrels)
results['mrr'] = _mrr

with open(os.path.join(args_cli.output_dir, f'results-{index_name}.json'), 'w') as f:
    json.dump(results, f, indent=4)


