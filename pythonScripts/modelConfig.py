import onnxruntime as ort
print(f"Runtime Version: {ort.__version__}")

# Model loading
session = ort.InferenceSession("model/gansynth.onnx")

# Input information
print("=== Inputs ===")
for inp in session.get_inputs():
    print("Name:", inp.name)
    print("Shape:", inp.shape)
    print("Type:", inp.type)
    print()

# Output Information
print("=== Outputs ===")
for out in session.get_outputs():
    print("Name:", out.name)
    print("Shape:", out.shape)
    print("Type:", out.type)
    print()