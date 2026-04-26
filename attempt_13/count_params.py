import sys
sys.path.append('/Users/vivekpandey/chess/attempt_11')
sys.path.append('/Users/vivekpandey/chess/attempt_13')

import torch
import torch.nn as nn

# Import ChessNet from attempt_11
import importlib.util
spec11 = importlib.util.spec_from_file_location("chess_engine_11", "/Users/vivekpandey/chess/attempt_11/chess_engine.py")
chess_engine_11 = importlib.util.module_from_spec(spec11)
spec11.loader.exec_module(chess_engine_11)

# Import ChessNet from attempt_13
spec13 = importlib.util.spec_from_file_location("chess_engine_13", "/Users/vivekpandey/chess/attempt_13/chess_engine.py")
chess_engine_13 = importlib.util.module_from_spec(spec13)
spec13.loader.exec_module(chess_engine_13)

# Create models
print("Creating models...")
model_11 = chess_engine_11.ChessNet(
    initial_channels=512,
    res_channels=256,
    num_res_blocks=8
)

model_13 = chess_engine_13.ChessNet(
    initial_channels=512,
    res_channels=256,
    num_res_blocks=16
)

# Count parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

total_11, trainable_11 = count_parameters(model_11)
total_13, trainable_13 = count_parameters(model_13)

print("\n" + "="*80)
print("MODEL PARAMETER COMPARISON")
print("="*80)
print(f"\nAttempt 11 (8 residual blocks):")
print(f"  Total parameters:     {total_11:,}")
print(f"  Trainable parameters: {trainable_11:,}")

print(f"\nAttempt 13 (16 residual blocks):")
print(f"  Total parameters:     {total_13:,}")
print(f"  Trainable parameters: {trainable_13:,}")

print(f"\nDifference:")
print(f"  Additional parameters: {total_13 - total_11:,}")
print(f"  Percentage increase:   {((total_13 - total_11) / total_11) * 100:.2f}%")
print(f"  Ratio (13/11):         {total_13 / total_11:.2f}x")

print("\n" + "="*80)

# Break down by component
print("\nPARAMETER BREAKDOWN:")
print("="*80)

def get_component_params(model):
    initial_conv = sum(p.numel() for p in model.initial_conv.parameters())
    initial_bn = sum(p.numel() for p in model.initial_bn.parameters())
    transition_conv = sum(p.numel() for p in model.transition_conv.parameters())
    transition_bn = sum(p.numel() for p in model.transition_bn.parameters())
    res_blocks = sum(p.numel() for block in model.res_blocks for p in block.parameters())
    value_head = (sum(p.numel() for p in model.value_conv.parameters()) +
                  sum(p.numel() for p in model.value_bn.parameters()) +
                  sum(p.numel() for p in model.fc1.parameters()) +
                  sum(p.numel() for p in model.fc2.parameters()))
    policy_head = (sum(p.numel() for p in model.policy_conv1.parameters()) +
                   sum(p.numel() for p in model.policy_bn1.parameters()) +
                   sum(p.numel() for p in model.policy_conv2.parameters()))

    return {
        'initial_conv': initial_conv,
        'initial_bn': initial_bn,
        'transition_conv': transition_conv,
        'transition_bn': transition_bn,
        'res_blocks': res_blocks,
        'value_head': value_head,
        'policy_head': policy_head
    }

components_11 = get_component_params(model_11)
components_13 = get_component_params(model_13)

print("\nAttempt 11 (8 blocks):")
for name, count in components_11.items():
    print(f"  {name:20s}: {count:12,}")

print("\nAttempt 13 (16 blocks):")
for name, count in components_13.items():
    print(f"  {name:20s}: {count:12,}")

print("\nDifference:")
for name in components_11.keys():
    diff = components_13[name] - components_11[name]
    if diff != 0:
        print(f"  {name:20s}: +{diff:11,}")

print("="*80)
