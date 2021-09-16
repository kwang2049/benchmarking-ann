import os
import json

result_files = [fname for fname in os.listdir('./') if fname.endswith('.json')]
results = {}
for result_file in result_files:
    with open(result_file, 'r') as f:
        result = json.load(f)
        index_name = result_file.replace('results-', '').replace('.index.json', '')
        results[index_name] = result

splitter = '\t'
headers = ['method'] + list(list(results.values())[0].keys())
with open('results.tsv', 'w') as f:
    f.write(splitter.join(headers) + '\n')
    for index_name, result in results.items():
        result_row = [index_name] + list(map(str, result.values()))
        f.write(splitter.join(result_row) + '\n')
