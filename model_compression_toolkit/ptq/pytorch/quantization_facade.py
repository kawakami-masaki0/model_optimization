# Copyright 2022 Sony Semiconductor Israel, Inc. All rights reserved.
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
import copy

from typing import Callable, Union, Tuple, Optional

from model_compression_toolkit.core.common.user_info import UserInformation
from model_compression_toolkit.core.common.visualization.tensorboard_writer import init_tensorboard_writer
from model_compression_toolkit.logger import Logger
from model_compression_toolkit.constants import PYTORCH
from model_compression_toolkit.target_platform_capabilities.schema.mct_current_schema import TargetPlatformCapabilities
from model_compression_toolkit.target_platform_capabilities.tpc_io_handler import load_target_platform_capabilities
from model_compression_toolkit.verify_packages import FOUND_TORCH
from model_compression_toolkit.core.common.mixed_precision.resource_utilization_tools.resource_utilization import ResourceUtilization
from model_compression_toolkit.core import CoreConfig
from model_compression_toolkit.core.common.mixed_precision.mixed_precision_quantization_config import \
    MixedPrecisionQuantizationConfig
from model_compression_toolkit.core.runner import core_runner
from model_compression_toolkit.ptq.runner import ptq_runner
from model_compression_toolkit.core.analyzer import analyzer_model_quantization
from model_compression_toolkit.core.common.quantization.quantize_graph_weights import quantize_graph_weights
from model_compression_toolkit.metadata import create_model_metadata

if FOUND_TORCH:
    from model_compression_toolkit.core.pytorch.default_framework_info import set_pytorch_info
    from model_compression_toolkit.core.pytorch.pytorch_implementation import PytorchImplementation
    from model_compression_toolkit.target_platform_capabilities.constants import DEFAULT_TP_MODEL
    from torch.nn import Module
    from model_compression_toolkit.exporter.model_wrapper.pytorch.builder.fully_quantized_model_builder import get_exportable_pytorch_model
    from model_compression_toolkit import get_target_platform_capabilities
    from mct_quantizers.pytorch.metadata import add_metadata
    from model_compression_toolkit.target_platform_capabilities.targetplatform2framework.attach2pytorch import \
        AttachTpcToPytorch

    DEFAULT_PYTORCH_TPC = get_target_platform_capabilities(PYTORCH, DEFAULT_TP_MODEL)

    @set_pytorch_info
    def pytorch_post_training_quantization(in_module: Module,
                                           representative_data_gen: Callable,
                                           target_resource_utilization: ResourceUtilization = None,
                                           core_config: CoreConfig = CoreConfig(),
                                           target_platform_capabilities: Union[TargetPlatformCapabilities, str] = DEFAULT_PYTORCH_TPC
                                           ) -> Tuple[Module, Optional[UserInformation]]:
        """
        Quantize a trained Pytorch module using post-training quantization.
        By default, the module is quantized using a symmetric constraint quantization thresholds
        (power of two) as defined in the default FrameworkQuantizationCapabilities.
        The module is first optimized using several transformations (e.g. BatchNormalization folding to
        preceding layers). Then, using a given dataset, statistics (e.g. min/max, histogram, etc.) are
        being collected for each layer's output (and input, depends on the quantization configuration).
        Thresholds are then being calculated using the collected statistics and the module is quantized
        (both coefficients and activations by default).
        If gptq_config is passed, the quantized weights are optimized using gradient based post
        training quantization by comparing points between the float and quantized modules, and minimizing the
        observed loss.

        Args:
            in_module (Module): Pytorch module to quantize.
            representative_data_gen (Callable): Dataset used for calibration.
            target_resource_utilization (ResourceUtilization): ResourceUtilization object to limit the search of the mixed-precision configuration as desired.
            core_config (CoreConfig): Configuration object containing parameters of how the model should be quantized, including mixed precision parameters.
            target_platform_capabilities (Union[TargetPlatformCapabilities, str]): TargetPlatformCapabilities to optimize the PyTorch model according to.

        Returns:
            A quantized module and information the user may need to handle the quantized module.

        Examples:

            Import a Pytorch module:

            >>> from torchvision import models
            >>> module = models.mobilenet_v2()

            Create a random dataset generator, for required number of calibration iterations (num_calibration_batches):
            In this example a random dataset of 10 batches each containing 4 images is used.

            >>> import numpy as np
            >>> num_calibration_batches = 10
            >>> def repr_datagen():
            >>>     for _ in range(num_calibration_batches):
            >>>         yield [np.random.random((4, 3, 224, 224))]

            Import MCT and pass the module with the representative dataset generator to get a quantized module
            Set number of clibration iterations to 1:

            >>> import model_compression_toolkit as mct
            >>> quantized_module, quantization_info = mct.ptq.pytorch_post_training_quantization(module, repr_datagen)

        """

        if core_config.debug_config.bypass:
            return in_module, None

        if core_config.is_mixed_precision_enabled:
            if not isinstance(core_config.mixed_precision_config, MixedPrecisionQuantizationConfig):
                Logger.critical("Given quantization config to mixed-precision facade is not of type "
                                "MixedPrecisionQuantizationConfig. Please use "
                                "pytorch_post_training_quantization API, or pass a valid mixed precision "
                                "configuration.")  # pragma: no cover

        tb_w = init_tensorboard_writer()

        fw_impl = PytorchImplementation()

        target_platform_capabilities = load_target_platform_capabilities(target_platform_capabilities)
        # Attach tpc model to framework
        attach2pytorch = AttachTpcToPytorch()
        framework_platform_capabilities = attach2pytorch.attach(target_platform_capabilities,
                                                             core_config.quantization_config.custom_tpc_opset_to_layer)

        # Ignore hessian info service as it is not used here yet.
        tg, bit_widths_config, _, scheduling_info = core_runner(in_model=in_module,
                                                                representative_data_gen=representative_data_gen,
                                                                core_config=core_config,
                                                                fw_impl=fw_impl,
                                                                fqc=framework_platform_capabilities,
                                                                target_resource_utilization=target_resource_utilization,
                                                                tb_w=tb_w)

        # At this point, tg is a graph that went through substitutions (such as BN folding) and is
        # ready for quantization (namely, it holds quantization params, etc.) but the weights are
        # not quantized yet. For this reason, we use it to create a graph that acts as a "float" graph
        # for things like similarity analyzer (because the quantized and float graph should have the same
        # architecture to find the appropriate compare points for similarity computation).
        similarity_baseline_graph = copy.deepcopy(tg)

        graph_with_stats_correction = ptq_runner(tg,
                                                 representative_data_gen,
                                                 core_config,
                                                 fw_impl,
                                                 tb_w)

        if core_config.debug_config.analyze_similarity:
            quantized_graph = quantize_graph_weights(graph_with_stats_correction)
            analyzer_model_quantization(representative_data_gen,
                                        tb_w,
                                        similarity_baseline_graph,
                                        quantized_graph,
                                        fw_impl)

        exportable_model, user_info = get_exportable_pytorch_model(graph_with_stats_correction)
        if framework_platform_capabilities.tpc.add_metadata:
            exportable_model = add_metadata(exportable_model,
                                            create_model_metadata(fqc=framework_platform_capabilities,
                                                                  scheduling_info=scheduling_info))
        return exportable_model, user_info


else:
    # If torch is not installed,
    # we raise an exception when trying to use these functions.
    def pytorch_post_training_quantization(*args, **kwargs):
        Logger.critical("PyTorch must be installed to use 'pytorch_post_training_quantization_experimental'. "
                        "The 'torch' package is missing.")  # pragma: no cover
