import svidreader.filtergraph as filtergraph
from svidreader.imagecache import ImageCache
import numpy as np
import argparse
import queue
import os
import threading
import time
import logging


def main():
    parser = argparse.ArgumentParser(description='Process program arguments.')
    parser.add_argument('-i', '--input', nargs='*')
    parser.add_argument('-f', '--frames', nargs='*')
    parser.add_argument('-o', '--output')
    parser.add_argument('-g', '--filtergraph')
    parser.add_argument('-r', '--recursive')
    parser.add_argument('-vr','--videoreader', default='iio', choices=('iio', 'decord'))
    parser.add_argument('-ac', '--autocache', default='True', choices=('True','False'))
    parser.add_argument('-mp', '--matplotlib', action='store_true', default=False, help='Activate Matplotlib')
    parser.add_argument('-d', '--debug',help="Print lots of debugging statements",action="store_const", dest="loglevel", const=logging.DEBUG,default=logging.WARNING)
    parser.add_argument('-v', '--verbose',help="Be verbose",action="store_const", dest="loglevel", const=logging.INFO,)
    args = parser.parse_args()

    logging.basicConfig(level=args.loglevel)

    files = []
    if args.input is not None:
        for f in args.input:
            if os.path.isdir(f) and args.recursive:
                get_files_recursive(f, files)
            elif os.path.exists(f):
                files.append(f)
            else:
                raise Exception(f"File {f} not found")

    if args.frames is not None:
        for f in args.frames:
            files.append(f)

    for i in range(len(files)):
        files[i] = filtergraph.get_reader(files[i], backend=args.videoreader, cache=args.autocache=="True")

    q = queue.Queue()

    def gui_worker():
        global app
        app = None
        while True:
            tmp = q.get(block=True)
            if tmp is None:
                break
            if tmp == "runqt":
                if app is None:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication([])
            else:
                tmp()


    gui_thread = threading.Thread(target=gui_worker, daemon=True)
    gui_thread.start()

    def gui_callback(function):
        q.put(function)

    fg = filtergraph.create_filtergraph_from_string(files, args.filtergraph, gui_callback=gui_callback)
    out = fg['out']

    def start_qt():
        if app is not None:
            app.exec()
            sys.exit()
    q.put(start_qt)

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        app.quit()
        out.close()
        exit()

    outputfile = None
    if args.output is not None:
        outputfile = open(args.output, 'w')

    import sys
    class process_printer:
        def __init__(self):
            self.lastval = -1

        def print(self, finished, fps="", remaining=0):
            val = int(finished * 1000)
            if self.lastval == val:
                return
            self.lastval = val
            sys.stdout.write('\r')
            str = "[%-20s] %.1f%% %.1f fps %0.1f minutes" % ('=' * (val // 50), finished * 100, fps, remaining)
            sys.stdout.write(str)
            sys.stdout.flush()

    if args.matplotlib:
        import matplotlib.pyplot as plt
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0)
    else:
        pp = process_printer()
        curtime = time.time()
        lasttime = curtime
        try:
            for i in range(0, out.n_frames):
                if i % 100 == 0:
                    lasttime = curtime
                    curtime = time.time()
                data = out.read(index=i)
                if outputfile is not None:
                    outputfile.write(str(i) + ' ' + ' '.join(map(str, np.asarray([data]).flatten())) + '\n')
                pp.print(i / out.n_frames, fps = 100 / (curtime - lasttime), remaining=(out.n_frames - i) * ((curtime - lasttime) / 100)/60)
        except Exception:
            out.close()
            raise
        if outputfile is not None:
            outputfile.close()
    if app is not None:
        app.quit()
    out.close()


if __name__ == '__main__':
    main()
