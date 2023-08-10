import hashlib
import imageio.v3 as iio
from svidreader.imagecache import ImageCache
from svidreader.effects import BgrToGray
from svidreader.effects import AnalyzeContrast
from svidreader.effects import FrameDifference
from svidreader.effects import Scale
from svidreader.effects import Crop
from svidreader.effects import DumpToFile
from svidreader.effects import Arange
from svidreader.effects import DumpToFile
from svidreader.viewer import MatplotlibViewer
from svidreader import SVidReader
from svidreader.cameraprojection import PerspectiveCameraProjection
from ccvtools import rawio

def create_filtergraph_from_string(inputs, pipeline):
    filtergraph = {}
    for i in range(len(inputs)):
        filtergraph["input_"+str(i)] = inputs[i]
    sp = pipeline.split(';')
    last = inputs[-1] if len(inputs) != 0 else None
    for line in sp:
        try:
            curinputs = []
            while True:
                line = line.strip()
                if line[0]!='[':
                    break
                br_close= line.find(']')
                curinputs.append(filtergraph[line[1:br_close]])
                line = line[br_close + 1:len(line)]
            noinput = len(curinputs) == 0
            if noinput:
                curinputs.extend(inputs)
            curoutputs = []
            while True:
                line = line.strip()
                if line[len(line) -1]!=']':
                    break
                br_open= line.rfind('[')
                curoutputs.append(line[br_open + 1:len(line) - 1])
                line = line[0:br_open]
            line = line.strip()
            eqindex = line.find('=')
            effectname = line
            if eqindex != -1:
                effectname = line[0:eqindex]
                line = line[eqindex + 1:len(line)]
            line = line.split(':')
            options = {}
            for opt in line:
                eqindex = opt.find('=')
                options[opt[0:eqindex]] = opt[eqindex + 1:len(opt)]

            if effectname == 'cache':
                assert len(curinputs) == 1
                last = ImageCache(curinputs[0],maxcount=1000)
            elif effectname == 'bgr2gray':
                assert len(curinputs) == 1
                last = BgrToGray(curinputs[0])
            elif effectname == 'tblend':
                assert len(curinputs) == 1
                last = FrameDifference(curinputs[0])
            elif effectname == 'reader':
                assert noinput
                last = SVidReader(options['input'],cache=False)
            elif effectname == 'permutate':
                assert len(curinputs) == 1
                last = PermutateFrames(reader = curinputs[0], permutation=options['input'])
            elif effectname == "contrast":
                assert len(curinputs) == 1
                last = AnalyzeContrast(curinputs[0])
            elif effectname == "crop":
                assert len(curinputs) == 1
                sp = options['size'].split('x')
                last = Crop(curinputs[0], width = int(sp[0]), height=int(sp[1]))
            elif effectname == "perprojection":
                assert len(curinputs) == 1
                last = PerspectiveCameraProjection(curinputs[0], config_file=options.get('calibration', None))
            elif effectname == "viewer":
                assert len(curinputs) == 1
                last = MatplotlibViewer(curinputs[0], backend=options['backend'] if 'backend' in options else "matplotlib")
            elif effectname == "dump":
                assert len(curinputs) == 1
                last = DumpToFile(reader=curinputs[0], outputfile=options['output'])
            elif effectname == "arange":
                last = Arange(inputs=curinputs, ncols=int(options['ncols']) if 'ncols' in options else -1)
            elif effectname == "scale":
                assert len(curinputs) == 1
                last = Scale(reader=curinputs[0], scale=float(options['scale']))
            else:
                raise Exception("Effectname " + effectname + " not known")
            for out in curoutputs:
                filtergraph[out] = last
        except Exception as e:
            raise e
    filtergraph['out'] = last
    return filtergraph
