import hashlib
import imageio.v3 as iio
from svidreader.imagecache import ImageCache
from svidreader.effects import BgrToGray
from svidreader.effects import AnalyzeContrast
from svidreader.effects import FrameDifference
from svidreader import SVidReader
from ccvtools import rawio

def create_filtergraph_from_string(inputs, pipeline):
    filtergraph = {}
    sp = pipeline.split(';')
    last = inputs
    for line in sp:
        try:
            curinputs = []
            while True:
                line = line.strip()
                if line[0]!='[':
                    break
                br_close= line.find(']')
                curinputs.append(filtergraph[line[1:br_close]])
                line = line[br_close:len(line)]
            if len(curinputs) == 0:
                curinputs.append(last)
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
            print(line)
            line = line.split(':')
            options = {}
            for opt in line:
                eqindex = opt.find('=')
                options[opt[0:eqindex]] = opt[eqindex + 1:len(opt)]

            if effectname == 'cache':
                assert len(inputs) == 1
                last = ImageCache(inputs[0])
            elif effectname == 'bgr2gray':
                assert len(inputs) == 1
                last = BgrToGray(inputs[0])
            elif effectname == 'tblend':
                assert len(inputs) == 1
                last = FrameDifference(inputs[0])
            elif effectname == 'reader':
                assert len(inputs) == 0
                last = SVidReader(options['input'])
            elif effectname == "contrast":
                assert len(inputs) == 1
                last = AnalyzeContrast(inputs[0])
            elif effectname == "dump":
                assert len(inputs) == 1
                last = DumpToFile(reader=inputs[0], output=options['output'])
            else:
                raise Exception("Effectname " + effectname + " not known")
            for out in curoutputs:
                filtergraph[out] = last
        except Exception as e:
            raise e
    filtergraph['out'] = last
    return filtergraph
