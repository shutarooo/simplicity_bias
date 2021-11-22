from lz_compression import customized_lz
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help='relative path for json file: ex. data/.../unit_uniform')
args = parser.parse_args() 

file_path = args.path

with open(file_path) as f:
    data = json.load(f)  # ファイルオブジェクトfinをデコードする

prob_and_complexity = [[d[1]/(10**6), customized_lz()] for d in data]
complexity = [customized_lz(d[0]) for d in data]

