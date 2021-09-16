# Benchmarking ANN
This repo benchmarks Approximate Nearest Neighbor algorithms supported by [Faiss](https://github.com/facebookresearch/faiss) for dense text retrieval (powered by [SBERT](https://github.com/UKPLab/sentence-transformers)).

## Usage: Experiments on MS-MARCO
1. Install Python requirements: ([Faiss](https://github.com/facebookresearch/faiss) & [SBERT](https://github.com/UKPLab/sentence-transformers))
    ```
    conda install -c pytorch faiss-cpu
    pip install sentence-transformers
    ```
2. First download the MS-MARCO dataset (in [BeIR](https://github.com/UKPLab/beir) format):
    ```bash
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
    unzip msmarco.zip
    ```
3. Embedd the queries and the documents with the SOTA [TAS-B model](https://arxiv.org/pdf/2104.06967.pdf):
    ```bash
    python embed.py --data_path=./msmarco --ndocs=8841823 --nqueries=509962 --model_name="msmarco-distilbert-base-tas-b" --embedded_dir=./msmarco-embedded
    ```
4. Run the benchmarking:
    ```bash
    bash ./benchmark.sh
    ```
    or run the individual experiment:
    ```bash
    python benchmark.py --eval_string "pq(384, 8)" --embedded_dir=./msmarco-embedded --output_dir=./msmarco-benchmarking
    ```
    where the `eval_string` defines how to build the Faiss index (please refer to the [benchmark.py](benchmark.py) for more details and [benchmark.sh](benchmark.sh) for other experiments).

## Pre-computed embedding files & results
To save the effort of the step 2 and 3, one can also download our pre-computed embedding files:
```bash
mkdir msmarco-embedded
cd msmarco-embedded
wget https://public.ukp.informatik.tu-darmstadt.de/kwang/benchmarking-ann/msmarco-embedded/ids.txt
wget https://public.ukp.informatik.tu-darmstadt.de/kwang/benchmarking-ann/msmarco-embedded/qrels.json
wget https://public.ukp.informatik.tu-darmstadt.de/kwang/benchmarking-ann/msmarco-embedded/embeddings.queries.pkl
wget https://public.ukp.informatik.tu-darmstadt.de/kwang/benchmarking-ann/msmarco-embedded/embeddings.documents.pkl
```

Results are available at:
[Benchmarking-ANN Google sheet](https://docs.google.com/spreadsheets/d/19RieebaLXYHjjBu9uEzkF6EFhy-EBVEtMCTHWleESNA/edit?usp=sharing)
> Note: The testing computational environment is a shared DGX2 machine. So the time metrics may not be absolutely comparable.
