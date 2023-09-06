import hashlib
from svidreader.imagecache import ImageCache
from svidreader.effects import BgrToGray
from svidreader.effects import AnalyzeImage
from svidreader.effects import FrameDifference
from svidreader.effects import Scale
from svidreader.effects import Crop
from svidreader.effects import DumpToFile
from svidreader.effects import Arange
from svidreader.effects import PermutateFrames
from svidreader.effects import DumpToFile
from svidreader.effects import Math
from svidreader.effects import MaxIndex

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
    return -1


def split_ignore_escaped(str, splitChar):
    result = []
    while True:
        index = find_ignore_escaped(str, splitChar)
        if index == -1:
            break
        result.append(str[0:index])
        str = str[index + 1:-1]
    result.append(str)
    return result


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
            else:
                result += char
            continue
        if double_quotes:
            if char == '"':
                double_quotes = False
            else:
                result += char
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


def get_reader(filename, backend="decord", cache=False):
    pipe = filename.find("|")
    pipeline = None
    if pipe >= 0:
        pipeline = filename[pipe + 1:]
        filename = filename[0:pipe]
    if backend == 'iio':
        from svidreader import SVidReader
        res = SVidReader(filename, cache=False)
    elif backend == 'decord':
        from svidreader import decord_video_wrapper
        res = decord_video_wrapper.DecordVideoReader(filename)
    else:
        raise Exception('Unknown videoreader')
    if cache:
        res = ImageCache(res, maxcount=500)
    if pipeline is not None:
        res = create_filtergraph_from_string([res], pipeline)['out']
    return res


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
            line = split_ignore_escaped(line,':')
            options = {}
            for opt in line:
                eqindex = find_ignore_escaped(opt, '=')
                if eqindex == -1:
                    options[opt] = None
                else:
                    options[opt[0:eqindex]] = unescape(opt[eqindex + 1:len(opt)])
            if effectname == 'cache':
                assert len(curinputs) == 1
                last = ImageCache(curinputs[0], maxcount=options.get('cmax',1000), processes=options.get('num_threads',1), preload=options.get('preload',20))
            elif effectname == 'bgr2gray':
                assert len(curinputs) == 1
                last = BgrToGray(curinputs[0])
            elif effectname == 'tblend':
                assert len(curinputs) == 1
                last = FrameDifference(curinputs[0])
            elif effectname == 'reader':
                assert noinput
                last = get_reader(options['input'],cache=False)
            elif effectname == 'permutate':
                assert len(curinputs) == 1
                last = PermutateFrames(reader = curinputs[0], permutation=options.get('input', None), mapping=options.get('map', None))
            elif effectname == "analyze":
                assert len(curinputs) == 1
                last = AnalyzeImage(curinputs[0])
            elif effectname == "majority":
                assert len(curinputs) == 1
                from svidreader.majorityvote import MajorityVote
                last = MajorityVote(curinputs[0],  window=int(options.get('window', 10)), scale=float(options.get('scale', 1)), foreground=options.get('foreground',False))
            elif effectname == "light_detector":
                assert len(curinputs) == 1
                from svidreader.light_detector import LightDetector
                last = LightDetector(curinputs[0], mode=options.get('mode','blinking'))
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
                    sp = rect.split('x')
                    w = int(sp[0])
                    h = int(sp[1])
                    x = int(sp[2])
                    y = int(sp[3])
                last = Crop(curinputs[0], x = x, y = y, width = w, height=h)
            elif effectname == "perprojection":
                assert len(curinputs) == 1
                from svidreader.cameraprojection import PerspectiveCameraProjection
                last = PerspectiveCameraProjection(curinputs[0], config_file=options.get('calibration', None))
            elif effectname == "scraper":
                assert len(curinputs) == 1
                from svidreader.videoscraper import VideoScraper
                last = VideoScraper(curinputs[0], tokens=options['tokens'])
            elif effectname == "argmax":
                assert len(curinputs) == 1
                last = MaxIndex(curinputs[0], count=options.get('count',1), radius=options.get('radius',1))
            elif effectname == "viewer":
                assert len(curinputs) == 1
                from svidreader.viewer import MatplotlibViewer
                last = MatplotlibViewer(curinputs[0], backend=options.get('backend','matplotlib'))
            elif effectname == "dump":
                assert len(curinputs) == 1
                last = DumpToFile(reader=curinputs[0], outputfile=options['output'])
            elif effectname == "arange":
                last = Arange(inputs=curinputs, ncols=int(options.get('ncols','-1')))
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
