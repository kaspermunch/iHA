from itertools import combinations
import argparse
import random
import subprocess
import re, os
import tempfile
import numpy as np
import pandas as pd
from pandas import DataFrame
import msprime, pyslim

random.seed(7)

parser = argparse.ArgumentParser()
parser.add_argument("--selcoef", type=float)
parser.add_argument("--window", type=int)
parser.add_argument("--samples", type=int)
parser.add_argument("slurm_script", type=str)
parser.add_argument("trees_file", type=str)
parser.add_argument("hdf_file", type=str)
args = parser.parse_args()

# slim needs output file to be absolute
if not os.path.isabs(args.trees_file):
    args.trees_file = os.path.abspath(args.trees_file)

# read slim template script file and replace output file
with open(args.slurm_script) as f:
    slurm_script = re.sub('(treeSeqOutput\(")([^(]+)("\))', 
        r'\1{}\3'.format(args.trees_file), f.read())

# write slim script file with the right output name
slurm_script_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
slurm_script_file.write(slurm_script)
slurm_script_file.close()

# run slim
cmd = './slim -d s={} {}'.format(args.selcoef, slurm_script_file.name)
p = subprocess.Popen(cmd.split(), 
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
print(stdout)
print(stderr)

# ts = pyslim.load("./sweep.trees").simplify()
# mutated = msprime.mutate(ts, rate=1e-7, random_seed=1, keep=True)
# mutated.dump("./sweep_overlaid.trees") 

# load trees from slim
ts = pyslim.load(args.trees_file)

# overlay mutations
mutated_ts = msprime.mutate(ts, rate=1e-7, random_seed=7)

window_size = args.window

# table with sampled haplotypes
table = random.sample([map(int, hap) for hap in mutated_ts.haplotypes()], args.samples)

df = DataFrame(table, dtype='int8')
df['start'] = df.index.values // window_size
df.set_index('start', inplace=True)

def pw_dist(df):
    "computes differences bewteen all pairs in a window"
    pairs = list(combinations(df.columns, 2))
    site_diffs = [np.bitwise_xor(df[p[0]], df[p[1]]) for p in pairs]
    return pd.concat(site_diffs, axis=1, keys=pairs).sum()

# make a dataframe with distance for each pair
pw_dist_df = (
    df
    .groupby('start')
    .apply(pw_dist)
    .reset_index()
    .melt(id_vars=['start'], var_name=['indiv_1', 'indiv_2'], value_name='dist')
    )
pw_dist_df['dist'] /= window_size
pw_dist_df['end'] = pw_dist_df.start + window_size

# convert indiv labels from object to int and and write hdf
pw_dist_df['indiv_1'] = pw_dist_df['indiv_1'].astype('int')
pw_dist_df['indiv_2'] = pw_dist_df['indiv_2'].astype('int')
pw_dist_df.to_hdf(args.hdf_file, 'df', format='table', mode='w')