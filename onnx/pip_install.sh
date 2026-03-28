# if ubuntu 22.04 no break-system-packages
if [ -f /etc/os-release ] && grep -q "22.04" /etc/os-release; then
    pip install onnxruntime-gpu onnx huggingface_hub
else
    pip install onnxruntime-gpu onnx huggingface_hub --break-system-packages
fi