from svidreader.video_supplier import VideoSupplier


import numpy as np

class MatplotlibViewer(VideoSupplier):
    def __init__(self, reader, cmap=None, backend="matplotlib"):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.backend = backend
        if backend == "matplotlib":
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Slider
            from matplotlib.widgets import Button
            self.ax = plt.axes([0.0, 0.05, 1, 0.95])
            self.im = self.ax.imshow(np.random.randn(10, 10), vmin=0, vmax=255, cmap=cmap)
            self.ax.axis('off')
            self.ax_button_previous_frame = plt.axes([0.0, 0.0, 0.1, 0.05])
            self.ax_slider_frame = plt.axes([0.2, 0.0, 0.1, 0.05])
            self.ax_button_next_frame = plt.axes([0.3, 0.0, 0.1, 0.05])

            self.button_previous_frame =  Button(ax=self.ax_button_previous_frame, label="<", color='pink', hovercolor='tomato')
            self.slider_frame = Slider(ax=self.ax_slider_frame,label='Frame',valmin=0,valmax=reader.n_frames,valinit=0)
            self.button_next_frame =  Button(ax=self.ax_button_next_frame, label=">", color='pink', hovercolor='tomato')

            self.slider_frame.on_changed(self.submit)
            self.button_previous_frame.on_clicked(self.previous_frame)
            self.button_next_frame.on_clicked(self.next_frame)
            self.frame = 0
            self.updating = False
            plt.tight_layout()
            plt.show(block=False)
        self.pipe = None

    def read(self, index,source=None):
        print(self.backend)
        self.frame = index
        img = self.inputs[0].read(index)
        if self.backend == "opencv":
            import cv2
            print("opencv-show")
            try:
                cv2.imshow("CV-Preview", img)
            except:
                pass
            print("showing")
        elif self.backend == "ffplay":
            import os
            import subprocess as sp
            if self.pipe == None:
                command = ["ffplay",
                           '-f', 'rawvideo',
                           '-vcodec', 'rawvideo',
                           '-video_size', str(img.shape[1]) + 'x' + str(img.shape[0]),  # size of one frame
                           '-pixel_format', 'rgb24' if img.shape[2] == 3 else 'gray8',
                           '-framerate', '200',
                           '-i','-']
                print(command)
                self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1000, preexec_fn=os.setpgrp)
            self.pipe.stdin.write(img.tobytes())
        elif self.backend == "skimage":
            from skimage import io
            io.imshow(img)
        elif self.backend == "matplotlib":
            import matplotlib.pyplot as plt
            if not self.updating:
                self.updating = True
                if source != self.slider_frame:
                    self.slider_frame.set_val(self.frame)
                self.im.set_array(img)
                plt.gcf().canvas.draw_idle()
                if source != self.slider_frame:
                    plt.gcf().canvas.start_event_loop(0.01)
                self.updating = False
        else:
            raise Exception("Unknown backend")
        return img


    def submit(self,val):
        self.read(int(val), source=self.slider_frame)


    def close(self):
        super().close()
        if self.pipe is not None:
            self.pipe.stdin.close()

    def previous_frame(self,event):
        self.read(index=self.frame - 1)


    def next_frame(self,event):
        self.read(index=self.frame + 1)


