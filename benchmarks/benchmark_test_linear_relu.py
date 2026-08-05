import torch
import torch.nn as nn
import torch.onnx

# 1. Define the combinatorial model (Linear + ReLU)
class FusedModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(FusedModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

# 2. Initialise the model and a dummy tensor
input_size = 100
output_size = 100
model = FusedModel(input_size, output_size)
model.eval() 

dummy_input = torch.randn(1, input_size)

# 3. Export to ONNX 
onnx_path = "fused_linear_relu.onnx"
torch.onnx.export(
    model,            
    dummy_input,
    onnx_path,
    export_params=True,         
    opset_version=18,    
    do_constant_folding=True,   
    input_names=['input'],
    output_names=['output']
)

print(f"Success! Combinatorial model saved to {onnx_path}")
