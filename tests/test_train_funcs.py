import numpy as np
from connectome.core.train_funcs import preprocess_images, process_images, get_voronoi_averages


def test_preprocess_images_resizes_and_colors():
    # create single-channel 256x256 image
    img = np.random.randint(0, 256, size=(256, 256), dtype=np.uint8)
    imgs = np.stack([img, img], axis=0)  # batch of 2

    processed = preprocess_images(imgs)

    # should be resized to 512x512 and have 3 channels
    assert processed.shape == (2, 512, 512, 3)


def test_process_images_and_voronoi():
    batch_size = 3
    imgs = np.random.randint(0, 256, size=(batch_size, 512, 512, 3), dtype=np.uint8)
    # create dummy voronoi indices (one per pixel)
    voronoi_indices = np.arange(512 * 512, dtype=np.int32)

    processed = process_images(imgs, voronoi_indices)
    # processed shape should be (batch, pixels, 5) => r,g,b,mean,cellIdx
    assert processed.shape == (batch_size, 512 * 512, 5)

    # get averages (should return list size == batch_size)
    avgs = get_voronoi_averages(processed)
    assert isinstance(avgs, list) and len(avgs) == batch_size 