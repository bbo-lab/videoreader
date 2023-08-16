import hashlib
import imageio.v3 as iio
from svidreader.imagecache import ImageCache
from svidreader.videoscraper import VideoScraper
from svidreader.effects import BgrToGray
from svidreader.effects import AnalyzeImage
from svidreader.effects import FrameDifference
from svidreader.effects import Scale
from svidreader.effects import Crop
from svidreader.majorityvote import MajorityVote
from svidreader.effects import DumpToFile
from svidreader.effects import Arange
from svidreader.effects import PermutateFrames
from svidreader.effects import DumpToFile
from svidreader.effects import Math
from svidreader.viewer import MatplotlibViewer
from svidreader import SVidReader
from svidreader.cameraprojection import PerspectiveCameraProjection
from ccvtools import rawio

def find_ignore_escaped(str, tofind):
    single_quotes = False
    double_quotes = False
    escaped = False

    for i in range(len(str)):
        char = str[i]
        if single_quotes:
            if char == "'":
                single_quotes = False
            continue
        if double_quotes:
            if char == '"':
                double_quotes = False
            continue
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == "'":
            single_quotes = True
            continue
        if char == '"':
            double_quotes = True
            continue
        if char == tofind:
            return i
    return -i


def unescape(str):
    single_quotes = False
    double_quotes = False
    escaped = False
    result = ""
    for i in range(len(str)):
        char = str[i]
        if single_quotes:
            if char == "'":
                single_quotes = False
            continue
        if double_quotes:
            if char == '"':
                double_quotes = False
            continue
        if escaped:
            escaped = False
            result += char
            continue
        if char == '\\':
            escaped = True
            continue
        if char == "'":
            single_quotes = True
            continue
        if char == '"':
            double_quotes = True
            continue
        result +=char
    return  result


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
                eqindex = find_ignore_escaped(opt, '=')
                if eqindex == -1:
                    options[opt] = None
                else:
                    options[opt[0:eqindex]] = unescape(opt[eqindex + 1:len(opt)])
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
                last = PermutateFrames(reader = curinputs[0], permutation=options.get('input', None), mapping=options.get('map', None))
            elif effectname == "analyze":
                assert len(curinputs) == 1
                last = AnalyzeImage(curinputs[0])
            elif effectname == "majority":
                assert len(curinputs) == 1
                last = MajorityVote(curinputs[0],  window=int(options.get('window', 10)), scale=float(options.get('scale', 1)), foreground='foreground' in options)
            elif effectname == "math":
                last = Math(curinputs, expression=options.get('exp'))
            elif effectname == "crop":
                assert len(curinputs) == 1
                w = -1
                h = -1
                x = 0
                y = 0
                if "size" in options:
                    sp = options['size'].split('x')
                    w = int(sp[0])
                    h = int(sp[0])
                if "rect" in options:
                    rect = options['rect']
                    print(rect)
                    sp = rect.split('x')
                    w = int(sp[0])
                    h = int(sp[1])
                    x = int(sp[2])
                    y = int(sp[3])
                last = Crop(curinputs[0], x = x, y = y, width = w, height=h)
            elif effectname == "perprojection":
                assert len(curinputs) == 1
                last = PerspectiveCameraProjection(curinputs[0], config_file=options.get('calibration', None))
            elif effectname == "scraper":
                assert len(curinputs) == 1
                last = VideoScraper(curinputs[0], tokens=options['tokens'])
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
