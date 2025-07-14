from torchvision.datasets import utils

from_path = "/mnt/Datasets/stl10_binary.tar.gz"
to_path = "/mnt/Datasets/stl10_binary"

utils.extract_archive(from_path, to_path)
