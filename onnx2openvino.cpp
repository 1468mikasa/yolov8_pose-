#include <openvino/openvino.hpp>

ov::CompiledModel compiled_model = core.compile_model("<INPUT_MODEL>.onnx", "AUTO");