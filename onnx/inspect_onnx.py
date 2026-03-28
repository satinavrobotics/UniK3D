#!/usr/bin/env python3
"""
Debug script to inspect ONNX model and find the problematic node_stack_6
"""
import onnx
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_onnx.py <path_to_onnx_model>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"Loading ONNX model: {model_path}")
model = onnx.load(model_path)

# Find node_stack_6 or any Concat nodes
print("\n=== Searching for Concat/Stack nodes ===")
concat_nodes = []
for i, node in enumerate(model.graph.node):
    if 'Concat' in node.op_type or 'Stack' in node.op_type or node.name == 'node_stack_6':
        concat_nodes.append((i, node))
        print(f"\nNode {i}: {node.name}")
        print(f"  Op: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        if node.attribute:
            for attr in node.attribute:
                print(f"  Attr {attr.name}: {attr}")

# Print input/output info
print("\n=== Model Inputs ===")
for inp in model.graph.input:
    print(f"  {inp.name}: {inp.type}")

print("\n=== Model Outputs ===")
for out in model.graph.output:
    print(f"  {out.name}: {out.type}")

print(f"\nFound {len(concat_nodes)} Concat/Stack nodes")
