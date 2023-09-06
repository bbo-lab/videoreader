from svidreader.video_supplier import VideoSupplier
import threading

import numpy as np

class MatplotlibViewer(VideoSupplier):
    def __init__(self, reader, cmap=None, backend="matplotlib"):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.backend = backend
        self.exit_event = None
        self.trigger_worker = None
        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider
            from matplotlib.widgets import Button
            from matplotlib.widgets import TextBox
            self.ax = plt.axes([0.0, 0.05, 1, 0.95])
            self.im = self.ax.imshow(np.random.randn(10, 10), vmin=0, vmax=255, cmap=cmap)
            self.ax.axis('off')
            self.ax_button_previous_frame = plt.axes([0.0, 0.0, 0.1, 0.05])
            self.ax_slider_frame = plt.axes([0.15, 0.0, 0.2, 0.05])
            self.ax_button_next_frame = plt.axes([0.4, 0.0, 0.1, 0.05])
            self.ax_textbox = plt.axes([0.5, 0.0, 0.1, 0.05])

            self.button_previous_frame =  Button(ax=self.ax_button_previous_frame, label="<", color='pink', hovercolor='tomato')
            self.slider_frame = Slider(ax=self.ax_slider_frame,label='',valmin=0,valmax=reader.n_frames,valinit=0)
            self.button_next_frame =  Button(ax=self.ax_button_next_frame, label=">", color='pink', hovercolor='tomato')
            self.textbox_frame = TextBox(ax=self.ax_textbox, label='', initial='0')

            self.slider_frame.on_changed(self.submit_slider)
            self.button_previous_frame.on_clicked(self.previous_frame)
            self.button_next_frame.on_clicked(self.next_frame)
            self.textbox_frame.on_submit(self.submit_textbox)
            self.frame = 0
            self.updating = False
            self.th = None
            plt.tight_layout()
            plt.show(block=False)
        self.pipe = None

    def read(self, index,source=None):
        self.frame = index
        img = self.inputs[0].read(index)
        if self.backend == "opencv":
            import cv2
            try:
                cv2.imshow("CV-Preview", img)
            except:
                pass
        elif self.backend == "ffplay":
            import os
            import subprocess as sp
            if self.pipe == None:
                command = ["ffplay",
                           '-f', 'rawvideo',
                           '-vcodec', 'rawvideo',
                           '-video_size', str(img.shape[1]) + 'x' + str(img.shape[0]),  # size of one frame
                           '-pixel_format', 'rgb24' if len(img.shape) == 3 and img.shape[2] == 3 else 'gray8',
                           '-framerate', '200',
                           '-i','-']
                self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1000, preexec_fn=os.setpgrp)
            self.pipe.stdin.write(img.astype(np.uint8).tobytes())
        elif self.backend == "skimage":
            from skimage import io
            io.imshow(img)
        elif self.backend == "matplotlib":
            if not self.updating:
                self.updating = True
                if source != self.slider_frame:
                    self.slider_frame.set_val(self.frame)
                if source != self.textbox_frame:
                    self.textbox_frame.set_val(self.frame)
                self.im.set_array(img)
                self.ax.figure.canvas .draw_idle()
                self.ax.figure.canvas.flush_events()
                self.updating = False
        else:
            raise Exception("Unknown backend")
        return img


    def submit_slider(self,val):
        self.read(int(val), source=self.slider_frame)


    def submit_textbox(self,val):
        self.read(int(val), source=self.textbox_frame)

    def close(self):
        super().close()
        if self.pipe is not None:
            self.pipe.stdin.close()
        if self.exit_event is not None:
            self.exit_event.set()
            self.trigger_worker.set()


    def worker(self):
        while True:
            if self.trigger_worker.wait():
                self.trigger_worker.clear()
                if self.update_frame_event.is_set():
                    self.update_frame_event.clear()
                    tmp = self.toread
                    self.toread = None
                    self.read(index=tmp)
                if self.exit_event.is_set():
                    th = None
                    return


    def read_frame_internal(self, index):
        self.toread = index
        if self.th is None:
            self.th = threading.Thread(target=self.worker, daemon=True)
            self.update_frame_event = threading.Event()
            self.exit_event = threading.Event()
            self.trigger_worker = threading.Event()
            self.th.start()
        self.update_frame_event.set()
        self.trigger_worker.set()

    def previous_frame(self,event):
        self.read_frame_internal(index=self.frame - 1)


    def next_frame(self,event):
        self.read_frame_internal(index=self.frame + 1)


