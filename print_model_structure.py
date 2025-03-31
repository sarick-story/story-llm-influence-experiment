import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model
model_path = "./tinyllama_1b_model"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

# Print model class and structure
print(f"Model class: {type(model).__name__}")
print(f"Model module class: {type(model.model).__name__}")

# Try to access parameters directly to confirm paths
try:
    print("\n=== Testing Direct Parameter Access ===")
    # Try to access a layer 21 parameter
    param = model.model.layers[21].self_attn.q_proj.weight
    print(f"Direct access to layer 21 q_proj weight: Shape {param.shape}")
    
    # Test access via getattr
    param_name = "model.layers.21.self_attn.q_proj.weight"
    parts = param_name.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    print(f"Access via getattr: {obj.shape}")
except Exception as e:
    print(f"Error accessing parameter: {e}")

# List all parameters to find the correct path
print("\n=== All Model Parameter Names ===")
for name, param in model.named_parameters():
    print(name)

# Print model state dict keys
print("\n=== State Dict Keys ===")
for key in model.state_dict().keys():
    print(key)

# Print the model structure
print("=== Model Module Names ===")
for name, _ in model.named_modules():
    if "layer" in name and ("self_attn" in name or "mlp" in name) and name.count(".") <= 3:
        print(name)

# Print first 20 parameter names without filtering
print("\n=== First 20 Model Parameter Names ===")
for i, (name, _) in enumerate(model.named_parameters()):
    if i < 20:
        print(name) 