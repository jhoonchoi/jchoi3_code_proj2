import imageio
import numpy as np
import torch


def to_uint8(image):
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def save_gif(frames, path, fps=15, loop=0):
    duration = 1000 // fps  # ms per frame
    imageio.mimsave(path, frames, duration=duration, loop=loop)
    return path


def hstack_frames(*frame_lists, pad=8, pad_value=255):
    n = len(frame_lists[0])
    assert all(len(f) == n for f in frame_lists), "frame lists must match"
    out = []
    for i in range(n):
        parts = []
        for j, frames in enumerate(frame_lists):
            if j > 0:
                h = frames[i].shape[0]
                parts.append(np.full((h, pad, 3), pad_value, dtype=np.uint8))
            parts.append(frames[i])
        out.append(np.concatenate(parts, axis=1))
    return out
