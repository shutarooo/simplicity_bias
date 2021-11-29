import argparse
import json

from lz_compression import customized_lz
import random




s = list('0'*21 + '1'*13 + '0'*18 + '1'*24 + '0'*39 + '1'*13)
print(customized_lz(s, False))
    
'''
s = '0'*127 + '1'
s = '0'*100 + '1' + '0'*27
s_list = list(s)
s_list_re = list(s[::-1])

print(s)

comp = lz_compression(s_list)
comp_re = lz_compression(s_list_re)

print(comp)
print(comp_re)
print(7*(comp+comp_re)/2)
'''


'''

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