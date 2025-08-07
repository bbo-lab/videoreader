from svidreader.video_supplier import VideoSupplier
import multiprocessing
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DumpToFile(VideoSupplier):
    def __init__(self, reader, outputfile, writer=None, opts=None, makedir=False, comment=None, fps=None):
        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        if opts is None:
            opts = {}
        self.outputfile = outputfile
        self.output = None
        self.l = multiprocessing.Lock()
        self.pipe = None
        self.shape = None
        self.fps = fps
        self.opts = opts
        if makedir:
            from pathlib import Path
            Path(outputfile).parent.mkdir(parents=True, exist_ok=True)
        if writer is not None and writer == "ffmpeg":
            self.type = "ffmpeg_movie"
        elif outputfile.endswith('.mp4'):
            self.type = "movie"
            self.outputfile = outputfile
        elif outputfile.endswith('.zip'):
            self.type = "zip"
        elif outputfile.endswith('.png'):
            self.type = "png"
        elif outputfile.endswith(".svg"):
            self.type = "svg"
        elif outputfile.endswith(".tif"):
            self.type = "tif"
            import imageio.v2 as imageio
            self.output = imageio.get_writer(outputfile, format='tiff', mode='I')
        else:
            self.type = "csv"
            self.mapkeys = None
            self.output = open(outputfile, 'w')
            if comment is not None:
                self.output.write(comment + '\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close(recursive=True)
        if exc_type is not None:
            logger.log(logging.ERROR, f"Exception in DumpToFile: {exc_value}")
        return False

    def close(self, recursive=False):
        logger.log(logging.DEBUG, f"Closing filewrite {self.outputfile}")
        super().close(recursive=recursive)
        if self.output is not None:
            self.output.close()
            self.output = None
        if self.pipe is not None:
            self.pipe.stdin.close()
            self.pipe.wait()
            self.pipe = None

    def read(self, index, force_type=np):
        data = self.inputs[0].read(index=index, force_type=force_type)
        if self.type == "movie":
            import imageio
            if self.output is None:
                self.output = imageio.get_writer(self.outputfile, fps=self.fps, quality=8)
            if data is not None:
                with self.l:
                    self.output.append_data(data)
        elif self.type == "csv":
            if self.mapkeys is None and isinstance(data, dict):
                self.mapkeys = data.keys()
                self.output.write(f"index {' '.join(self.mapkeys)} \n")
            self.output.write(f"{index} {' '.join([str(data[k]) for k in self.mapkeys])} \n")
        elif self.type == "zip":
            import os
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            import cv2
            for k, v in os.environ.items():
                if k.startswith("QT_") and "cv2" in v:
                    del os.environ[k]

            import zipfile
            import yaml
            if self.output is None:
                with self.l:  #Double check to make sure file was not created in the meantime
                    if self.output is None:
                        self.output = zipfile.ZipFile(self.outputfile, mode="w", compression=zipfile.ZIP_STORED)
                        self.keyframes = self.opts.get('keyframes', 1)
                        info = {'keyframes': self.keyframes}
                        self.output.writestr("info.yaml", yaml.dump(info))
            out_data = VideoSupplier.convert(data, module=np)
            if isinstance(out_data, str):
                filetype = "svg"
            elif out_data.dtype == np.uint8 or out_data.dtype == np.uint16:
                filetype = "png"
            elif out_data.dtype == np.float32 or out_data.dtype == np.float64:
                filetype = "exr"
                if out_data.dtype == np.float64:
                    out_data = out_data.astype(np.float32)
            else:
                raise Exception(f"Datatype not understood {type(out_data)}")
            img_name = f"{index:06d}.{filetype}"
            if filetype == "png" or filetype == "exr":
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                if index % self.keyframes != 0:
                    out_data = np.copy(out_data)
                    out_data -= self.inputs[0].read(index=(index // self.keyframes) * self.keyframes)
                    out_data += 127
                image_encoded = \
                    cv2.imencode(f'.{filetype}', cv2.cvtColor(out_data, cv2.COLOR_RGB2BGR) if out_data.shape[2] == 3 else out_data,
                                 encode_param)[1].tobytes()
                with self.l:
                    self.output.writestr(img_name, image_encoded)
            elif filetype == "svg":
                self.output.writestr(img_name, out_data)
        elif self.type == "png":
            import imageio
            out_data = VideoSupplier.convert(data, module=np)
            imageio.v3.imwrite(self.output.format(index), out_data)
        elif self.type == "svg":
            with open(self.output.format(index)) as outfile:
                outfile.write(data)
        elif self.type == "tif":
            if np.any(np.isnan(data)):
                logger.log(logging.WARNING, f"NaN values in frame {index} of {self.outputfile}, replacing with 0")
                data = np.nan_to_num(data, nan=0)
            with self.l:
                self.output.append_data(data)
        elif self.type == "ffmpeg_movie":
            import subprocess as sp
            import os
            with self.l:
                if self.pipe is None:
                    encoder = self.opts.get('encoder', 'libx264')
                    if encoder is None:
                        encoder = 'hevc_nvenc'
                    if encoder == 'hevc_nvenc':
                        codec = ['-i', '-', '-an', '-vcodec', 'hevc_nvenc']
                    elif encoder == 'h264_nvenc':
                        codec = ['-i', '-', '-an', '-vcodec', 'h264_nvenc']
                    elif encoder == '264_vaapi':
                        codec = ['-hwaccel', 'vaapi' '-hwaccel_output_format', 'hevc_vaapi', '-vaapi_device',
                                 '/dev/dri/renderD128', '-i',
                                 '-', '-an', '-c:v', 'hevc_vaapi']
                    elif encoder == 'uncompressed':
                        codec = ['-f', 'rawvideo']
                    elif encoder == 'libx264':
                        codec = ['-i', '-', '-vcodec', 'libx264']
                    elif encoder == 'h264_v4l2m2m':
                        codec = ['-i', '-', '-c:v', 'h264_v4l2m2m']
                    elif encoder == 'dummy':
                        codec = ['null']
                    else:
                        raise Exception(f"Encoder {encoder} not known")
                    pix_fmt = 'rgb24'
                    if data.shape[2] == 1:
                        pix_fmt = 'gray8'
                    quality = ['-b:v', self.opts.get('bitrate')] if 'bitrate' in self.opts else ['-crf', str(self.opts.get('crf', 15))]
                    self.shape = data.shape
                    command = ["ffmpeg",
                               '-y',  # (optional) overwrite output file if it exists
                               '-f', 'rawvideo',
                               '-vcodec', 'rawvideo',
                               '-s', f'{self.shape[1]}x{self.shape[0]}',  # size of one frame
                               '-pix_fmt', pix_fmt,
                               '-r', str(self.opts.get('fps','60')),  # frames per second
                               '-rtbufsize', '2G',
                               *codec,
                               '-preset', self.opts.get('preset', 'slow'),
                               '-qmin', '10',
                               '-qmax', '26',
                               *quality,
                               self.outputfile]
                    print(command)
                    logger.log(logging.INFO, f"{' '.join(command)}")
                    self.pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.STDOUT, bufsize=1000, preexec_fn=os.setpgrp)
                assert self.shape == data.shape
                assert data.dtype == np.uint8
                self.pipe.stdin.write(data.tobytes())
        return data
