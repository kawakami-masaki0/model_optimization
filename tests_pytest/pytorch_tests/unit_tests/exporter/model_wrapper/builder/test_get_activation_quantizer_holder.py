# Copyright 2025 Sony Semiconductor Israel, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest

from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_activation_quantizer_holder
from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
from mct_quantizers import PytorchActivationQuantizationHolder, PytorchPreservingActivationQuantizationHolder, PytorchFLNActivationQuantizationHolder

from tests_pytest._test_util.graph_builder_utils import build_node, build_nbits_qc

fw_impl = PytorchImplementation()

# test case for get_activation_quantizer_holder
test_input_0 = (build_node('Node0', qcs=[build_nbits_qc(a_enable=True, q_preserving=True)]), False)
test_input_1 = (build_node('Node1', qcs=[build_nbits_qc(a_enable=True, q_preserving=True)]), True)
test_input_2 = (build_node('Node2', qcs=[build_nbits_qc(a_enable=True, q_preserving=False)]), False)
test_input_3 = (build_node('Node3', qcs=[build_nbits_qc(a_enable=True, q_preserving=False)]), True)

test_expected_0 = (PytorchPreservingActivationQuantizationHolder, False)
test_expected_1 = (PytorchPreservingActivationQuantizationHolder, True)
test_expected_2 = (PytorchActivationQuantizationHolder,)
test_expected_3 = (PytorchActivationQuantizationHolder,)

@pytest.mark.parametrize(("inputs", "expected"), [
    (test_input_0, test_expected_0),
    (test_input_1, test_expected_1),
    (test_input_2, test_expected_2),
    (test_input_3, test_expected_3),
])
def test_get_activation_quantizer_holder(inputs, expected):
    
    node = inputs[0]
    node.candidates_quantization_cfg[0].activation_quantization_cfg.set_activation_quantization_param({'threshold': 8.0, 'is_signed': False})
    node.final_activation_quantization_cfg = node.candidates_quantization_cfg[0].activation_quantization_cfg
    
    node.quantization_bypass = inputs[1] # set bypass
    result = get_activation_quantizer_holder(node, fw_impl=fw_impl)
    assert isinstance(result, expected[0])
    if isinstance(result, PytorchPreservingActivationQuantizationHolder) or isinstance(result, PytorchFLNActivationQuantizationHolder):
        assert result.quantization_bypass == expected[1]
