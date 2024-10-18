import svidreader.filtergraph as filtergraph
import numpy as np
import argparse
import queue
import os
import threading
import logging
import sys
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


class GuiApplication:
    def __init__(self):
        self.q = queue.Queue()
        self.gui_thread = None
        self.app = None

    def gui_worker(self):
        while True:
            tmp = self.q.get(block=True)
            if tmp is None:
                break
            if tmp == "runqt":
                if self.app is None:
                    from PyQt5.QtWidgets import QApplication
                    self.app = QApplication([])
                    print("app_set")
            else:
                tmp()

    def run(self):
        def exec_qt():
            if self.app is not None:
                self.app.exec()
                sys.exit()
            else:
                print("warning, app not set")

        self.q.put(exec_qt)

    def start(self):
        self.gui_thread = threading.Thread(target=self.gui_worker, daemon=True)
        self.gui_thread.start()

    def gui_callback(self, function):
        self.q.put(function)


def main():
    parser = argparse.ArgumentParser(description='Process program arguments.')
    parser.add_argument('-i', '--input', nargs='*')
    parser.add_argument('-f', '--frames', nargs='*', type=int, default=None)
    parser.add_argument('-o', '--output')
    parser.add_argument('-g', '--filtergraph')
    parser.add_argument('-r', '--recursive')
    parser.add_argument('-j', '--jobs', default=1, type=int)
    parser.add_argument('-vr', '--videoreader', default='iio', choices=('iio', 'decord'))
    parser.add_argument('-ac', '--autocache', default='True', choices=('True', 'False'))
    parser.add_argument('-mp', '--matplotlib', action='store_true', default=False, help='Activate Matplotlib')
    parser.add_argument('-d', '--debug', help="Print lots of debugging statements", action="store_const",
                        dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel",
                        const=logging.INFO, )
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

    for i in range(len(files)):
        files[i] = filtergraph.get_reader(files[i], backend=args.videoreader, cache=args.autocache == "True")

    ga = GuiApplication()
    ga.start()

    fg = filtergraph.create_filtergraph_from_string(files, args.filtergraph, gui_callback=ga.gui_callback)
    out = fg['out']
    ga.run()

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        ga.app.quit()
        out.close()
        exit()

    outputfile = None
    if args.output is not None:
        outputfile = open(args.output, 'w')

    if args.matplotlib:
        import matplotlib.pyplot as plt
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(0)
    else:
        frames = range(out.n_frames) if args.frames is None else args.frames
        try:
            def process_frame(frame_idx):
                data = out.read(index=frame_idx)
                if outputfile is not None:
                    outputfile.write(f"{frame_idx} {' '.join(map(str, np.asarray([data]).flatten()))} \n")

            if args.jobs != 1:
                thread_map(process_frame, frames, max_workers=args.jobs, chunksize=1)
            else:
                for frame_idx in tqdm(frames):
                    process_frame(frame_idx)
        except Exception:
            out.close()
            raise
        if outputfile is not None:
            outputfile.close()
    if ga.app is not None:
        ga.app.quit()
    out.close(recursive=True)
    out = None


if __name__ == '__main__':
    main()
