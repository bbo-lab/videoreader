from svidreader.video_supplier import VideoSupplier
import os


class VideoScraper(VideoSupplier):
    def __init__(self, reader, tokens="dog, cat", subdivision_shape=(5,3), overlay=True):
        import clip
        import torch
        from torchvision.datasets import CIFAR100
        from PIL import Image

        super().__init__(n_frames=reader.n_frames, inputs=(reader,))
        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text = clip.tokenize(tokens.split(",")).to(self.device)
        self.subdivision_shape = subdivision_shape
        self.overlay = overlay

    def read(self, index):
        import clip
        import torch
        from torchvision.datasets import CIFAR100
        from PIL import Image

        img = self.inputs[0].read(index=index)
        image = self.preprocess(Image.fromarray(img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.subdivision_shape is not None:
                import numpy as np
                if self.overlay:
                    result = np.zeros_like(img,dtype=float)
                else:
                    result = np.zeros(dtype=np.uint8,shape=(*self.subdivision_shape, len(self.text)))
                for x in range(self.subdivision_shape[0]):
                    for y in range(self.subdivision_shape[1]):
                        cropped = img[
                            img.shape[0] * x       // (self.subdivision_shape[0] + 1):
                            img.shape[0] * (x + 2) // (self.subdivision_shape[0] + 1),
                            img.shape[1] * y       // (self.subdivision_shape[1] + 1):
                            img.shape[1] * (y + 2) // (self.subdivision_shape[1] + 1)]
                        image = self.preprocess(Image.fromarray(cropped)).unsqueeze(0).to(self.device)
                        image_features = self.model.encode_image(image)
                        text_features = self.model.encode_text(self.text)

                        logits_per_image, logits_per_text = self.model(image, self.text)
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                        print(probs)
                        probs *= 0.25
                        if self.overlay:
                            result[
                                img.shape[0] * x // (self.subdivision_shape[0] + 1):
                                img.shape[0] * (x + 2) // (self.subdivision_shape[0] + 1),
                                img.shape[1] * y // (self.subdivision_shape[1] + 1):
                                img.shape[1] * (y + 2) // (self.subdivision_shape[1] + 1)] += cropped * probs.flatten()[np.newaxis, np.newaxis,:]
                        else:
                            result[x,y] = (probs * 255).astype(np.uint8)
                return result.astype(np.uint8)
            else:
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(self.text)

                logits_per_image, logits_per_text = self.model(image, self.text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)
        return img