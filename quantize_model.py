# -*- coding: utf-8 -*-
"""
quantize_model.py
-----------------
One-time script to quantize MobileFaceNet from FP32 to INT8.
Run this once before starting main.py.

Usage
-----
    python quantize_model.py

Input  : models/mobilefacenet.onnx      (~16 MB, FP32)
Output : models/mobilefacenet_int8.onnx (~4 MB,  INT8)

Expected speedup: ~2-4x faster inference under the same CPU constraint.
The INT8 model produces the same 512-d embeddings; face recognition
accuracy is not meaningfully affected.
"""

import os

INPUT_MODEL  = "models/mobilefacenet.onnx"
OUTPUT_MODEL = "models/mobilefacenet_int8.onnx"


def main():
    if not os.path.isfile(INPUT_MODEL):
        print("[ERROR] Input model not found: %s" % INPUT_MODEL)
        print("  Download buffalo_sc from https://github.com/deepinsight/insightface/releases")
        print("  Extract w600k_mbf.onnx and save as models/mobilefacenet.onnx")
        return

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("[ERROR] onnxruntime.quantization not available.")
        print("  Install with: pip install onnxruntime>=1.8")
        return

    print("[QUANTIZE] Input  : %s (%.1f MB)" % (INPUT_MODEL, os.path.getsize(INPUT_MODEL) / 1e6))
    print("[QUANTIZE] Output : %s" % OUTPUT_MODEL)
    print("[QUANTIZE] Quantizing FP32 -> INT8 ...")

    os.makedirs(os.path.dirname(OUTPUT_MODEL) if os.path.dirname(OUTPUT_MODEL) else ".", exist_ok=True)

    # QUInt8 produces QLinearConv ops (supported in onnxruntime 1.8 CPU provider).
    # QInt8 produces ConvInteger ops which are NOT supported in onnxruntime 1.8.
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QUInt8,
    )

    out_mb = os.path.getsize(OUTPUT_MODEL) / 1e6
    print("[QUANTIZE] Done. Saved: %s (%.1f MB)" % (OUTPUT_MODEL, out_mb))
    print("[QUANTIZE] Update drone_detection_worker.py:")
    print('           model_path="models/mobilefacenet_int8.onnx"')


if __name__ == "__main__":
    main()
