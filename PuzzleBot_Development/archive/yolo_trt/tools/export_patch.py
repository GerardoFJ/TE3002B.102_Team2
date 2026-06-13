#!/usr/bin/env python3
"""Export best.pt -> ONNX at a given imgsz and patch out the unsupported Mod op.

YOLO26's end-to-end head uses a `Mod` op (topk_index % num_classes) that
TensorRT 8.2's ONNX parser cannot import. Since the indices are non-negative,
`a % b` is exactly `a - (a / b) * b`, which uses only TRT-supported ops.

Usage:  python3 export_patch.py <imgsz>
Writes: /home/puzzlebot/best_<imgsz>_nomod.onnx
"""
import os
import sys

from ultralytics import YOLO
import onnx
from onnx import helper

PT = "/home/puzzlebot/best.pt"


def main():
    sz = int(sys.argv[1])
    YOLO(PT).export(format="onnx", opset=12, imgsz=sz, simplify=False,
                    dynamic=False)
    raw = f"/home/puzzlebot/best_{sz}.onnx"
    os.replace("/home/puzzlebot/best.onnx", raw)

    m = onnx.load(raw)
    g = m.graph
    new, n_rep = [], 0
    for n in g.node:
        if n.op_type == "Mod":
            a, b = n.input
            o = n.output[0]
            bs = n.name
            new += [
                helper.make_node("Div", [a, b], [bs + "_d"], name=bs + "_dN"),
                helper.make_node("Mul", [bs + "_d", b], [bs + "_m"],
                                 name=bs + "_mN"),
                helper.make_node("Sub", [a, bs + "_m"], [o], name=bs + "_sN"),
            ]
            n_rep += 1
        else:
            new.append(n)
    del g.node[:]
    g.node.extend(new)
    onnx.checker.check_model(m)
    out = f"/home/puzzlebot/best_{sz}_nomod.onnx"
    onnx.save(m, out)
    print(f"OK imgsz={sz} replaced {n_rep} Mod -> {out}")


if __name__ == "__main__":
    main()
