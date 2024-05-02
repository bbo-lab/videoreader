import hashlib
import os
from svidreader.imagecache import ImageCache
from svidreader.effects import BgrToGray
from svidreader.effects import GrayToBgr
from svidreader.effects import FrameDifference
from svidreader.effects import Scale
from svidreader.effects import Crop
from svidreader.effects import ConstFrame
from svidreader.effects import DumpToFile
from svidreader.effects import Arange
from svidreader.effects import PermutateFrames
from svidreader.effects import Concatenate
from svidreader.effects import DumpToFile
from svidreader.effects import Math
from svidreader.effects import MaxIndex
from svidreader.effects import ChangeFramerate

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
        str = str[index + 1:]
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


def get_reader(filename, backend="decord", cache=False, options={}):
    pipe = filename.find("|")
    pipeline = None
    res = None
    processes = 1
    if pipe >= 0:
        pipeline = filename[pipe + 1:]
        filename = filename[0:pipe]
    from svidreader import get_imageEndings
    if os.path.isdir(filename) or filename.endswith(get_imageEndings()) or filename.endswith('.zip'):
        from svidreader import ImageReader
        res = ImageReader.ImageRange(filename)
        processes = 10
    elif backend == 'iio':
        from svidreader import SVidReader
        res = SVidReader(filename, cache=False)
    elif backend == 'decord':
        from svidreader import decord_video_wrapper
        res = decord_video_wrapper.DecordVideoReader(filename)
    else:
        raise Exception('Unknown videoreader')
    if cache:
        res = ImageCache(res, maxcount=200, processes=processes)
    if pipeline is not None:
        res = create_filtergraph_from_string([res], pipeline, options=options)['out']
    return res


def create_filtergraph_from_string(inputs, pipeline, gui_callback=None, options={}):
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
            if noinput and last is not None:
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
            line = split_ignore_escaped(line,':')
            effect_options = options.copy()
            for opt in line:
                eqindex = find_ignore_escaped(opt, '=')
                if eqindex == -1:
                    effect_options[opt] = None
                else:
                    effect_options[opt[0:eqindex]] = unescape(opt[eqindex + 1:len(opt)])
            if effectname == 'cache':
                assert len(curinputs) == 1
                last = ImageCache(curinputs[0], maxcount=effect_options.get('cmax',1000), processes=effect_options.get('num_threads',1), preload=effect_options.get('preload',20))
            elif effectname == 'minicache':
                assert len(curinputs) == 1
                last = ImageCache(curinputs[0], maxcount=effect_options.get('cmax', 5),
                                  processes=effect_options.get('num_threads', 1),
                                  preload=effect_options.get('preload', 2))
            elif effectname == 'bgr2gray':
                assert len(curinputs) == 1
                last = BgrToGray(curinputs[0])
            elif effectname == 'gray2bgr':
                assert len(curinputs) == 1
                last = GrayToBgr(curinputs[0])
            elif effectname == 'tblend':
                assert len(curinputs) == 1
                last = FrameDifference(curinputs[0])
            elif effectname == 'reader':
                assert noinput
                last = get_reader(effect_options['input'], backend=effect_options.get("backend", "iio"), cache=False)
            elif effectname == 'flow':
                assert len(curinputs) == 1
                import svidreader.flow as flow
                last = flow.OpticFlow(curinputs[0])
            elif effectname == 'permutate':
                assert len(curinputs) == 1
                last = PermutateFrames(reader=curinputs[0],
                                       permutation=effect_options.get('input', None),
                                       mapping=effect_options.get('map', None),
                                       source=effect_options.get('source','from'),
                                       destination=effect_options.get('destination','to'),
                                       sourceoffset=int(effect_options.get('sourceoffset', 0)),
                                       destinationoffset=int(effect_options.get('destinationoffset', 0)))
            elif effectname == "analyze":
                assert len(curinputs) == 1
                from svidreader.analyze_image import AnalyzeImage
                last = AnalyzeImage(curinputs[0], effect_options)
            elif effectname == "majority":
                assert len(curinputs) == 1
                from svidreader.majorityvote import MajorityVote
                last = MajorityVote(curinputs[0],  window=int(effect_options.get('window', 10)), scale=float(effect_options.get('scale', 1)), foreground='foreground' in effect_options)
            elif effectname == "change_framerate":
                assert len(curinputs) == 1
                last = ChangeFramerate(curinputs[0], factor=float(effect_options.get('factor')))
            elif effectname == "light_detector":
                assert len(curinputs) == 1
                from svidreader.light_detector import LightDetector
                last = LightDetector(curinputs[0], mode=effect_options.get('mode','blinking'))
            elif effectname == "normalized_contrast":
                from svidreader.mask_unfocused import NormalizedContrast
                last = NormalizedContrast(curinputs[0])
            elif effectname == "radial_contrast":
                from svidreader.local_radial import RadialContrast
                last = RadialContrast(curinputs[0])
            elif effectname == "const":
                assert len(curinputs) == 1
                last = ConstFrame(curinputs[0], frame=int(effect_options.get('frame')))
            elif effectname == "math":
                last = Math(curinputs, expression=effect_options.get('exp'), library=effect_options.get('library','numpy'))
            elif effectname == "crop":
                assert len(curinputs) == 1
                w = -1
                h = -1
                x = 0
                y = 0
                if "size" in effect_options:
                    sp = effect_options['size'].split('x')
                    w = int(sp[0])
                    h = int(sp[1])
                if "rect" in effect_options:
                    rect = effect_options['rect']
                    sp = rect.split('x')
                    x = int(sp[0])
                    y = int(sp[1])
                    w = int(sp[2])
                    h = int(sp[3])
                last = Crop(curinputs[0], x = x, y = y, width = w, height=h)
            elif effectname == "perprojection":
                assert len(curinputs) == 1
                from svidreader.cameraprojection import PerspectiveCameraProjection
                last = PerspectiveCameraProjection(curinputs[0], config_file=effect_options.get('calibration', None))
            elif effectname == "scraper":
                assert len(curinputs) == 1
                from svidreader.videoscraper import VideoScraper
                last = VideoScraper(curinputs[0], tokens=effect_options['tokens'])
            elif effectname == "argmax":
                assert len(curinputs) == 1
                last = MaxIndex(curinputs[0], count=effect_options.get('count',1), radius=effect_options.get('radius',1))
            elif effectname == "viewer":
                assert len(curinputs) == 1
                from svidreader.viewer import MatplotlibViewer
                last = MatplotlibViewer(curinputs[0], backend=effect_options.get('backend','matplotlib'), gui_callback=gui_callback)
            elif effectname == "dump":
                assert len(curinputs) == 1
                last = DumpToFile(reader=curinputs[0], outputfile=effect_options['output'], writer=effect_options.get('writer', None), opts=effect_options, makedir='mkdir' in effect_options, comment=effect_options.get('comment',None))
            elif effectname == "arange":
                last = Arange(inputs=curinputs, ncols=int(effect_options.get('ncols','-1')))
            elif effectname == "concatenate":
                last = Concatenate(inputs=curinputs)
            elif effectname == "scale":
                assert len(curinputs) == 1
                last = Scale(reader=curinputs[0], scale=float(effect_options['scale']))
            elif effectname == "overlay":
                assert len(curinputs) == 2
                from svidreader.effects import Overlay
                last = Overlay(reader=curinputs[0], overlay=curinputs[1], x=effect_options.get('x', 0),
                               y=effect_options.get('y', 0))
            else:
                raise Exception("Effectname " + effectname + " not known")
            for out in curoutputs:
                filtergraph[out] = last
        except Exception as e:
            raise e
    filtergraph['out'] = last
    return filtergraph
