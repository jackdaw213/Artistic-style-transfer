import os
import numpy as np

class ExternalInputCallable:
    def __init__(self, content_dir, style_dir, batch_size):
        self.content = os.listdir(content_dir)
        self.style = os.listdir(style_dir)

        self.pair = list(zip(self.content, self.style))

        self.content_dir = content_dir
        self.style_dir = style_dir

        self.full_iterations = len(self.pair) // batch_size

        self.perm = None  
        self.last_seen_epoch = (None)

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration()
        
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=21 + sample_info.epoch_idx).permutation(
                len(self.pair)
            )
        sample_idx = self.perm[sample_info.idx_in_epoch]

        content, style = self.pair[sample_idx]

        content = np.fromfile(os.path.join(self.content_dir, content), dtype=np.uint8)
        style = np.fromfile(os.path.join(self.style_dir, style), dtype=np.uint8)

        return content, style