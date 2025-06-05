#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision.models as models
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower


def main() -> None:
    # pth 파일도 괜찮고, 학습된 모델을 불러와서 executorch에 사용할수 있는 pte파일로 추출하는 코드입니다.
    # model = models.mobilenet_v3_small(weights='DEFAULT').eval()
    model = models.mobilenet_v3_large(weights='DEFAULT').eval()

    sample_inputs = (torch.randn(1, 3, 224, 224), )

    et_program = to_edge_transform_and_lower(
        torch.export.export(model, sample_inputs),
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with open("model.pte", "wb") as file:
        et_program.write_to_file(file)


if __name__ == "__main__":
    main()
