import os
from sentence_transformers import SentenceTransformer
import json
import pickle
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='In the BeIR format.')
parser.add_argument('--model_name', default='msmarco-distilbert-base-tas-b', help='SBERT or HF-Transformer checkpoints')
parser.add_argument('--ndocs', type=int, default=None)
parser.add_argument('--nqueries', type=int, default=None)
parser.add_argument('--embedded_dir', type=str, default='msmarco-embedded')
args = parser.parse_args()

sbert = SentenceTransformer(args.model_name)

os.makedirs(args.embedded_dir, exist_ok=True)
path_doc_embedding = os.path.join(args.embedded_dir, 'embeddings.documents.pkl')
path_query_embedding = os.path.join(args.embedded_dir, 'embeddings.queries.pkl')
path_ids = os.path.join(args.embedded_dir, 'ids.txt')
path_qrels = os.path.join(args.embedded_dir, 'qrels.json')

### document embeddings ###
print('>>> Embedding documents')
documents = {}
with open(os.path.join(args.data_path, 'corpus.jsonl'), 'r') as f:
    for line in tqdm.tqdm(f, total=args.ndocs):
        val_dict = json.loads(line)
        documents[val_dict['_id']] = val_dict['text']

embeddings = sbert.encode(list(documents.values()), show_progress_bar=True, batch_size=128)
with open(path_doc_embedding, 'wb') as f:
    pickle.dump(embeddings, f)

with open(path_ids, 'w') as f:
    for _id in documents.keys():
        f.write(_id + '\n')

### query embeddings ###
print('>>> Embedding queries')
queries = {}
with open(os.path.join(args.data_path, 'queries.jsonl'), 'r') as f:
    for line in tqdm.tqdm(f, total=args.nqueries):
        val_dict = json.loads(line)
        queries[val_dict['_id']] = val_dict['text']

query_embeddings = sbert.encode(list(queries.values()), show_progress_bar=True, batch_size=128)
for _id, query_embedding in zip(queries.keys(), query_embeddings):
    queries[_id] = query_embedding

with open(path_query_embedding, 'wb') as f:
    pickle.dump(queries, f)

### load and save qrels ###
print('>>> Loading and saving qrels')
qrels = {}
with open(os.path.join(args.data_path, 'qrels', 'dev.tsv'), 'r') as f:
    f.readline()
    for line in f:
        qid, did, _ = line.split('\t')
        qrels.setdefault(qid, [])
        qrels[qid].append(did)

with open(path_qrels, 'w') as f:
    json.dump(qrels, f)