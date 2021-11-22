import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("test", type=str, help='test')

args = parser.parse_args() # ⇦これがNamespace型

print(args.test)

path_file = 'data/func_freq/a.json'
with open(path_file, 'w') as f:
    json.dump([[1,1], [1,2]] , f)
with open(path_file, 'w') as f:
    json.dump([[1,1], [1,2]] , f)



'''

with open('data/func_freq/unit_uniform.json') as f:
    unit_data = json.load(f)  # ファイルオブジェクトfinをデコードする

data = unit_data[1][0]
data_char = ''
for d in data:
    data_char += str(bin(int(d)))[2:]
print(data_char)
'''