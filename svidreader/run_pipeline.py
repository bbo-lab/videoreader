import svidreader.filtergraph as filtergraph
from svidreader import SVidReader
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Process program arguments.')
parser.add_argument('-i', '--input', nargs='*')
parser.add_argument('-o', '--output')
parser.add_argument('-g', '--filtergraph')
args = parser.parse_args()

files = []
for f in args.input:
    if os.path.isdir(f):
        if args.recursive:
            get_files_recursive(f, files)
    elif os.path.isfile(f):
        files.append(f)
    else:
        raise Exception("File " + f + " not found")

for i in range(len(files)):
    files[i] = SVidReader(files[i])

fg = filtergraph.create_filtergraph_from_string(files, args.filtergraph)
out = fg['out']

if args.output is not None:
    outputfile = open(args.output, 'w')

print(fg)
for i in range(0, out.n_frames):
    data = out.read(index=i)
    outputfile.write(str(i) + ' ' + ' '.join(map(str, np.asarray([data]).flatten())) + '\n')

outputfile.close()