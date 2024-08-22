Regarding tensor parallel implementation of Llama31_8b model
- Implemented the tensor parallel MLP submodule, current PCC = 0.58
- Working on ttnn.all_gather API on w2 tensor
- On running the module by adding ttnn.all_gather API, facing the following error:
`E       RuntimeError: TT_FATAL @ ../ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.cpp:135: input_tensor.device()->get_ethernet_sockets(this->receiver_device_id.value()).size() >= 2 * this->num_links
E       info:
E       2 Device all gather requires at least 2 eth connections per link`
- To run the submodule, run the command: `pytest models/demos/wormhole/llama31_8b/tests/test_llama_mlp.py`

- On running ttnn.all_gather API in unit test, facing the following error:
`>       return self.function(*function_args, **function_kwargs)
E       IndexError: vector
ttnn/ttnn/decorators.py:328: IndexError`
- To run the unit test, run the command: `pytest tests/ttnn/unit_tests/test_multi_device.py::test_ttnn_multi_device_all_gather_device_mesh`
