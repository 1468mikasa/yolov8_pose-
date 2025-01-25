#include <openvino/openvino.hpp>
ov::Core core;
ov::CompiledModel compiled_model = core.compile_model("model.xml", "AUTO");
ov::InferRequest infer_request = compiled_model.create_infer_request();

// Get input port for model with one input
auto input_port = compiled_model.input();
// Create tensor from external memory
ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), memory_ptr);
// Set input tensor for model with one input
infer_request.set_input_tensor(input_tensor);

infer_request.start_async();
infer_request.wait();

// Get output tensor by tensor name
auto output = infer_request.get_tensor("tensor_name");
const float *output_buffer = output.data<const float>();
// output_buffer[] - accessing output tensor data

ov_shape_free(&input_shape);
ov_tensor_free(output_tensor);
ov_output_const_port_free(input_port);
ov_tensor_free(tensor);
ov_infer_request_free(infer_request);
ov_compiled_model_free(compiled_model);
ov_model_free(model);
ov_core_free(core);