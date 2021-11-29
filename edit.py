import json

file_path = 'data/func_freq/deep/unit_uniform.json'
with open(file_path) as f:
    data_lines = f.read()
print(data_lines.find('[[['))

with open(file_path, 'r') as f:
    read_data = json.load(f)
print(len(read_data))