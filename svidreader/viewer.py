from svidreader.video_supplier import VideoSupplier
import threading

import numpy as np

class MatplotlibViewer(VideoSupplier):
    def __init__(self, reader, cmap=None, backend="qt", gui_callback = None):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        self.backend = backend
        self.exit_event = None
        self.trigger_worker = None
        self.updating = False
        self.gui_callback = gui_callback
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
            self.slider_frame = Slider(ax=self.ax_slider_frame,label='',valmin=0,valmax=reader.n_frames - 1,valinit=0)
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
        elif backend == "qt":
            self.gui_loaded = False
            def redraw(source=None, current_frame=None):
                if not self.gui_loaded:
                    return
                self.updating = True
                self.textbox_frame.setText(str(self.frame))
                if current_frame is None:
                    current_frame = self.read(self.frame)
                if source != self.slider_frame:
                    self.slider_frame.setValue(self.frame)
                if source != self.textbox_frame:
                    self.textbox_frame.setText(str(self.slider_frame.value()))
                if len(current_frame) == 3 and current_frame.shape[2] == 2:
                    current_frame = np.dstack((current_frame[:,:,0],current_frame[:,:,1],current_frame[:,:,1]))
                self.img.setImage(np.swapaxes(current_frame, 0, 1),autoLevels=current_frame.dtype!=np.uint8)
                self.updating = False

            self.redraw = redraw

            def submit_frame():
                self.frame = int(self.textbox_frame.text())
                self.redraw(source=self.textbox_frame)

            def submit_slider_frame():
                self.frame = self.slider_frame.value()
                self.redraw(source=self.slider_frame)

            def run_qt():
                from PyQt5.QtCore import Qt
                from PyQt5 import QtWidgets
                from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QLineEdit, QWidget, QSlider, QComboBox, QPushButton, QButtonGroup
                from pyqtgraph import PlotWidget, plot
                import pyqtgraph as pg
                import os
                from PyQt5.QtCore import QTimer

                self.main_window = QWidget()
                buttomWidget = QWidget()
                globalLayout = QVBoxLayout()
                buttomLayout = QHBoxLayout()
                buttomWidget.setLayout(buttomLayout)
                self.main_window.setLayout(globalLayout)
                self.graphWidget = pg.PlotWidget()
                globalLayout.addWidget(self.graphWidget)

                self.updating = True
                current_frame = self.read(0)
                if len(current_frame) == 3 and current_frame.shape[2] == 2:
                    current_frame = np.dstack((current_frame[:,:,0],current_frame[:,:,1],current_frame[:,:,1]))
                self.img = pg.ImageItem(np.swapaxes(current_frame, 0, 1), autoLevels=current_frame.dtype!=np.uint8)
                self.updating = False
                self.graphWidget.addItem(self.img)

                self.slider_frame = QSlider(Qt.Horizontal)
                self.slider_frame.setMinimum(0)
                self.slider_frame.setMaximum(self.n_frames)
                self.slider_frame.setValue(0)
                self.textbox_frame = QLineEdit()
                self.buttonPlay = (QPushButton("<"), QPushButton("■"), QPushButton(">"))
                buttonGroupPlay = QButtonGroup()
                buttonGroupPlay.setExclusive(True)

                self.comboBoxCopyToClipboard = QComboBox()
                self.comboBoxCopyToClipboard.addItem('None')
                self.comboBoxCopyToClipboard.addItem('Coordinate')
                self.comboBoxCopyToClipboard.addItem('Value')

                def mouse_clicked(evt):
                    vb = self.graphWidget.plotItem.vb
                    scene_coords = evt.scenePos()
                    if self.graphWidget.sceneBoundingRect().contains(scene_coords):
                        mouse_point = vb.mapSceneToView(scene_coords)
                        print(f'clicked plot X: {mouse_point.x()}, Y: {mouse_point.y()}, event: {evt}')
                        match str(self.comboBoxCopyToClipboard.currentText()):
                            case 'Coordinate':
                                import pyperclip
                                pyperclip.copy(",".join([str(p) for p in (mouse_point.x(), mouse_point.y())]))
                            case 'Value':
                                pass

                self.graphWidget.scene().sigMouseClicked.connect(mouse_clicked)
                self.slider_frame.valueChanged.connect(submit_slider_frame)
                self.textbox_frame.setText('0')
                self.textbox_frame.setMaximumWidth(100)
                globalLayout.addWidget(buttomWidget)
                buttomLayout.addWidget(self.slider_frame)
                for button in self.buttonPlay:
                    button.setCheckable(True)
                    button.setMaximumWidth(30)
                    buttonGroupPlay.addButton(button)
                    buttomLayout.addWidget(button)

                self.play = 0
                def run_through_frames():
                    self.frame += self.play
                    self.redraw(source=buttonGroupPlay)

                self.timer = QTimer(self.main_window)
                self.timer.setSingleShot(False)
                self.timer.setInterval(20)  # in milliseconds, so 5000 = 5 seconds
                self.timer.timeout.connect(run_through_frames)

                def buttonPlayClicked(object):
                    buttonGroupPlay.id(object)
                    if object.text() == '<' or object.text() == '>':
                        self.timer.start()
                        if object.text() == '<':
                            self.play = -1
                        if object.text() == '>':
                            self.play = 1
                    if object.text() == '■':
                        self.timer.stop()
                        self.play = 0

                buttonGroupPlay.buttonClicked.connect(buttonPlayClicked)
                buttomLayout.addWidget(self.textbox_frame)
                buttomLayout.addWidget(self.comboBoxCopyToClipboard)
                self.textbox_frame.returnPressed.connect(submit_frame)
                self.main_window.show()
                self.gui_loaded = True
            if gui_callback == None:
                run_qt()
            else:
                gui_callback("runqt")
                gui_callback(run_qt)
        elif backend == 'ffplay' or backend == 'skimage':
            pass
        else:
            raise Exception(f'Backend {backend} not known')
        self.pipe = None

    def read(self, index,source=None, force_type=np):
        self.frame = index
        img = VideoSupplier.convert(self.inputs[0].read(index, force_type=force_type), np)
        if self.backend == "opencv":
            import cv2
            try:
                cv2.imshow("CV-Preview", img)
            except:
                pass
        elif self.backend == "ffplay":
            import os
            import subprocess as sp
            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
            if self.pipe == None:
                command = ["ffmpeg.ffplay",
                           '-f', 'rawvideo',
                           '-vcodec', 'rawvideo',
                           '-video_size', str(img.shape[1]) + 'x' + str(img.shape[0]),  # size of one frame
                           '-pixel_format', 'rgb24' if img.shape[2] >= 2 else 'gray8',
                           '-framerate', '200',
                           '-i','-']
                print(command)
                self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1000, preexec_fn=os.setpgrp)
            if img.shape[2] == 2:
                img = np.dstack((img[:,:,0],img[:,:,1],img[:,:,1]))
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
        elif self.backend == "qt":
            if not self.updating:
                self.redraw(source = None, current_frame=img)
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


