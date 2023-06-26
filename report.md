# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | hubconf.py:41 | add "import torch_mlu" |
| 2 | src/transformers/tokenization_utils_base.py:66 | add "import torch_mlu" |
| 3 | src/transformers/trainer_utils.py:37 | change "is_torch_cuda_available," to "is_torch_mlu_available, " |
| 4 | src/transformers/trainer_utils.py:44 | add "import torch_mlu" |
| 5 | src/transformers/trainer_utils.py:94 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 6 | src/transformers/trainer_utils.py:95 | change "# ^^ safe to call this function even if cuda is not available" to "# ^^ safe to call this function even if mlu is not available " |
| 7 | src/transformers/trainer_utils.py:416 | change "if is_torch_cuda_available():" to "if is_torch_mlu_available(): " |
| 8 | src/transformers/trainer_utils.py:471 | change "self.torch.cuda.reset_peak_memory_stats()" to "self.torch.mlu.reset_peak_memory_stats() " |
| 9 | src/transformers/trainer_utils.py:472 | change "self.torch.cuda.empty_cache()" to "self.torch.mlu.empty_cache() " |
| 10 | src/transformers/trainer_utils.py:476 | change "self.gpu_mem_used_at_start = self.torch.cuda.memory_allocated()" to "self.gpu_mem_used_at_start = self.torch.mlu.memory_allocated() " |
| 11 | src/transformers/trainer_utils.py:500 | change "self.torch.cuda.empty_cache()" to "self.torch.mlu.empty_cache() " |
| 12 | src/transformers/trainer_utils.py:509 | change "self.gpu_mem_used_now = self.torch.cuda.memory_allocated()" to "self.gpu_mem_used_now = self.torch.mlu.memory_allocated() " |
| 13 | src/transformers/trainer_utils.py:510 | change "self.gpu_mem_used_peak = self.torch.cuda.max_memory_allocated()" to "self.gpu_mem_used_peak = self.torch.mlu.max_memory_allocated() " |
| 14 | src/transformers/convert_graph_to_onnx.py:273 | add "import torch_mlu" |
| 15 | src/transformers/pytorch_utils.py:17 | add "import torch_mlu" |
| 16 | src/transformers/convert_pytorch_checkpoint_to_tf2.py:102 | add "import torch_mlu" |
| 17 | src/transformers/debug_utils.py:21 | add "import torch_mlu" |
| 18 | src/transformers/optimization.py:22 | add "import torch_mlu" |
| 19 | src/transformers/modeling_utils.py:30 | add "import torch_mlu" |
| 20 | src/transformers/modeling_utils.py:1991 | change "https://test.pypi.org/simple/ bitsandbytes-cudaXXX` where XXX is your CUDA version (e.g. 11.6 = 116)." to "https://test.pypi.org/simple/ bitsandbytes-mluXXX` where XXX is your CUDA version (e.g. 11.6 = 116). " |
| 21 | src/transformers/modeling_utils.py:2106 | change "# The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info." to "# The max memory utils require PyTorch >= 1.10 to have torch.mlu.mem_get_info. " |
| 22 | src/transformers/testing_utils.py:498 | add "import torch_mlu" |
| 23 | src/transformers/testing_utils.py:500 | change "return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple GPUs")(test_case)" to "return unittest.skipUnless(torch.mlu.device_count() > 1, "test requires multiple GPUs")(test_case) " |
| 24 | src/transformers/testing_utils.py:512 | change "return unittest.skipUnless(torch.cuda.device_count() < 2, "test requires 0 or 1 GPU")(test_case)" to "return unittest.skipUnless(torch.mlu.device_count() < 2, "test requires 0 or 1 GPU")(test_case) " |
| 25 | src/transformers/testing_utils.py:524 | change "return unittest.skipUnless(torch.cuda.device_count() < 3, "test requires 0 or 1 or 2 GPUs")(test_case)" to "return unittest.skipUnless(torch.mlu.device_count() < 3, "test requires 0 or 1 or 2 GPUs")(test_case) " |
| 26 | src/transformers/testing_utils.py:547 | change "torch_device = "cuda" if torch.cuda.is_available() else "cpu"" to "torch_device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 27 | src/transformers/testing_utils.py:574 | change "return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)" to "return unittest.skipUnless(torch_device == "mlu", "test requires CUDA")(test_case) " |
| 28 | src/transformers/testing_utils.py:578 | change """"Decorator marking a test that requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0"""" to """"Decorator marking a test that requires torch>=1.10, using Ampere GPU or newer arch with mlu>=11.0""" " |
| 29 | src/transformers/testing_utils.py:581 | change ""test requires torch>=1.10, using Ampere GPU or newer arch with cuda>=11.0"," to ""test requires torch>=1.10, using Ampere GPU or newer arch with mlu>=11.0", " |
| 30 | src/transformers/testing_utils.py:594 | change """"Decorator marking a test that requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7."""" to """"Decorator marking a test that requires Ampere or a newer GPU arch, mlu>=11 and torch>=1.7.""" " |
| 31 | src/transformers/testing_utils.py:596 | change "is_torch_tf32_available(), "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7"" to "is_torch_tf32_available(), "test requires Ampere or a newer GPU arch, mlu>=11 and torch>=1.7" " |
| 32 | src/transformers/testing_utils.py:758 | change "return torch.cuda.device_count()" to "return torch.mlu.device_count() " |
| 33 | src/transformers/modeling_tf_pytorch_utils.py:451 | add "import torch_mlu" |
| 34 | src/transformers/training_args.py:59 | add "import torch_mlu" |
| 35 | src/transformers/training_args.py:270 | change "no_cuda (`bool`, *optional*, defaults to `False`):" to "no_mlu (`bool`, *optional*, defaults to `False`): " |
| 36 | src/transformers/training_args.py:286 | change "NVIDIA architecture or using CPU (no_cuda). This is an experimental API and it may change." to "NVIDIA architecture or using CPU (no_mlu). This is an experimental API and it may change. " |
| 37 | src/transformers/training_args.py:295 | change "The backend to use for mixed precision training. Must be one of `"auto", "cuda_amp", "apex", "cpu_amp"`." to "The backend to use for mixed precision training. Must be one of `"auto", "mlu_amp", "apex", "cpu_amp"`. " |
| 38 | src/transformers/training_args.py:306 | change "on PyTorch's version default of `torch.backends.cuda.matmul.allow_tf32`. For more details please refer to" to "on PyTorch's version default of `torch.backends.mlu.matmul.allow_tf32`. For more details please refer to " |
| 39 | src/transformers/training_args.py:544 | change "`"nvfuser"`, `"aot_nvfuser"`, `"aot_cudagraphs"`, `"ofi"`, `"fx2trt"`, `"onnxrt"` and `"ipex"`." to "`"nvfuser"`, `"aot_nvfuser"`, `"aot_mlugraphs"`, `"ofi"`, `"fx2trt"`, `"onnxrt"` and `"ipex"`. " |
| 40 | src/transformers/training_args.py:732 | change "no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})" to "no_mlu: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"}) " |
| 41 | src/transformers/training_args.py:755 | change "" architecture or using CPU (no_cuda). This is an experimental API and it may change."" to "" architecture or using CPU (no_mlu). This is an experimental API and it may change." " |
| 42 | src/transformers/training_args.py:776 | change ""choices": ["auto", "cuda_amp", "apex", "cpu_amp"]," to ""choices": ["auto", "mlu_amp", "apex", "cpu_amp"], " |
| 43 | src/transformers/training_args.py:1026 | change ""choices": ["auto", "cuda_amp", "apex", "cpu_amp"]," to ""choices": ["auto", "mlu_amp", "apex", "cpu_amp"], " |
| 44 | src/transformers/training_args.py:1185 | change "if self.no_cuda and not is_torch_bf16_cpu_available() and not is_torch_tpu_available():" to "if self.no_mlu and not is_torch_bf16_cpu_available() and not is_torch_tpu_available(): " |
| 45 | src/transformers/training_args.py:1188 | change "elif not self.no_cuda and torch.cuda.is_available() and not is_torch_bf16_gpu_available():" to "elif not self.no_mlu and torch.mlu.is_available() and not is_torch_bf16_gpu_available(): " |
| 46 | src/transformers/training_args.py:1191 | change ""Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU with cuda>=11.0"" to ""Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU with mlu>=11.0" " |
| 47 | src/transformers/training_args.py:1204 | change "" `--half_precision_backend cuda_amp` instead"" to "" `--half_precision_backend mlu_amp` instead" " |
| 48 | src/transformers/training_args.py:1227 | change "and (self.device.type != "cuda")" to "and (self.device.type != "mlu") " |
| 49 | src/transformers/training_args.py:1239 | change "and (self.device.type != "cuda")" to "and (self.device.type != "mlu") " |
| 50 | src/transformers/training_args.py:1268 | change "torch.backends.cuda.matmul.allow_tf32 = True" to "torch.backends.mlu.matmul.allow_tf32 = True " |
| 51 | src/transformers/training_args.py:1276 | change "torch.backends.cuda.matmul.allow_tf32 = True" to "torch.backends.mlu.matmul.allow_tf32 = True " |
| 52 | src/transformers/training_args.py:1278 | change "raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")" to "raise ValueError("--tf32 requires Ampere or a newer GPU arch, mlu>=11 and torch>=1.7") " |
| 53 | src/transformers/training_args.py:1281 | change "torch.backends.cuda.matmul.allow_tf32 = False" to "torch.backends.mlu.matmul.allow_tf32 = False " |
| 54 | src/transformers/training_args.py:1511 | change "if self.no_cuda:" to "if self.no_mlu: " |
| 55 | src/transformers/training_args.py:1577 | change "device = torch.device("cuda", local_rank)" to "device = torch.device("mlu", local_rank) " |
| 56 | src/transformers/training_args.py:1584 | change "device = torch.device("cuda", self.local_rank)" to "device = torch.device("mlu", self.local_rank) " |
| 57 | src/transformers/training_args.py:1601 | change "device = torch.device("cuda", self.local_rank)" to "device = torch.device("mlu", self.local_rank) " |
| 58 | src/transformers/training_args.py:1634 | change "# GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`" to "# GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `mlu:0` " |
| 59 | src/transformers/training_args.py:1636 | change "device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 60 | src/transformers/training_args.py:1639 | change "self._n_gpu = torch.cuda.device_count()" to "self._n_gpu = torch.mlu.device_count() " |
| 61 | src/transformers/training_args.py:1644 | change "torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)" to "torch.distributed.init_process_group(backend="cncl", timeout=self.ddp_timeout_delta) " |
| 62 | src/transformers/training_args.py:1645 | change "device = torch.device("cuda", self.local_rank)" to "device = torch.device("mlu", self.local_rank) " |
| 63 | src/transformers/training_args.py:1648 | change "if device.type == "cuda":" to "if device.type == "mlu": " |
| 64 | src/transformers/training_args.py:1649 | change "torch.cuda.set_device(device)" to "torch.mlu.set_device(device) " |
| 65 | src/transformers/integrations.py:39 | add "import torch_mlu" |
| 66 | src/transformers/modeling_outputs.py:18 | add "import torch_mlu" |
| 67 | src/transformers/configuration_utils.py:336 | add "import torch_mlu" |
| 68 | src/transformers/image_utils.py:54 | add "import torch_mlu" |
| 69 | src/transformers/modelcard.py:530 | add "import torch_mlu" |
| 70 | src/transformers/modelcard.py:895 | change "if trainer.use_cuda_amp:" to "if trainer.use_mlu_amp: " |
| 71 | src/transformers/trainer_pt_utils.py:32 | add "import torch_mlu" |
| 72 | src/transformers/trainer_pt_utils.py:210 | change "device: Optional[torch.device] = torch.device("cuda")," to "device: Optional[torch.device] = torch.device("mlu"), " |
| 73 | src/transformers/trainer_pt_utils.py:952 | change "The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and" to "The GPU allocated and peak memory reporting is done with `torch.mlu.memory_allocated()` and " |
| 74 | src/transformers/trainer_pt_utils.py:953 | change "`torch.cuda.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as" to "`torch.mlu.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as " |
| 75 | src/transformers/trainer_pt_utils.py:954 | change "`torch.cuda` memory management system doesn't track any memory allocated outside of pytorch. For example, the very" to "`torch.mlu` memory management system doesn't track any memory allocated outside of pytorch. For example, the very " |
| 76 | src/transformers/trainer_pt_utils.py:955 | change "first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory." to "first mlu call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory. " |
| 77 | src/transformers/trainer_pt_utils.py:961 | change "`torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`'s tracker" to "`torch.mlu.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`'s tracker " |
| 78 | src/transformers/trainer_pt_utils.py:968 | change "`torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [`Trainer`] will disrupt" to "`torch.mlu.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [`Trainer`] will disrupt " |
| 79 | src/transformers/trainer_pt_utils.py:969 | change "the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves." to "the normal behavior of any such tools that rely on calling `torch.mlu.reset_peak_memory_stats` themselves. " |
| 80 | src/transformers/trainer.py:59 | add "import torch_mlu" |
| 81 | src/transformers/trainer.py:453 | change "# postpone switching model to cuda when:" to "# postpone switching model to mlu when: " |
| 82 | src/transformers/trainer.py:559 | change "self.use_cuda_amp = False" to "self.use_mlu_amp = False " |
| 83 | src/transformers/trainer.py:595 | change "args.half_precision_backend = "cuda_amp"" to "args.half_precision_backend = "mlu_amp" " |
| 84 | src/transformers/trainer.py:602 | change "if args.half_precision_backend == "cuda_amp":" to "if args.half_precision_backend == "mlu_amp": " |
| 85 | src/transformers/trainer.py:603 | change "self.use_cuda_amp = True" to "self.use_mlu_amp = True " |
| 86 | src/transformers/trainer.py:621 | change "self.scaler = torch.cuda.amp.GradScaler()" to "self.scaler = torch.mlu.amp.GradScaler() " |
| 87 | src/transformers/trainer.py:636 | change "and self.use_cuda_amp" to "and self.use_mlu_amp " |
| 88 | src/transformers/trainer.py:1333 | change "self.use_cuda_amp = False" to "self.use_mlu_amp = False " |
| 89 | src/transformers/trainer.py:2270 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 90 | src/transformers/trainer.py:2272 | change "torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])" to "torch.mlu.random.set_rng_state(checkpoint_rng_state["mlu"]) " |
| 91 | src/transformers/trainer.py:2275 | change "torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])" to "torch.mlu.random.set_rng_state_all(checkpoint_rng_state["mlu"]) " |
| 92 | src/transformers/trainer.py:2364 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 93 | src/transformers/trainer.py:2367 | change "rng_states["cuda"] = torch.cuda.random.get_rng_state_all()" to "rng_states["mlu"] = torch.mlu.random.get_rng_state_all() " |
| 94 | src/transformers/trainer.py:2369 | change "rng_states["cuda"] = torch.cuda.random.get_rng_state()" to "rng_states["mlu"] = torch.mlu.random.get_rng_state() " |
| 95 | src/transformers/trainer.py:2605 | change "if self.use_cuda_amp or self.use_cpu_amp:" to "if self.use_mlu_amp or self.use_cpu_amp: " |
| 96 | src/transformers/trainer.py:2610 | change "else torch.cuda.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)" to "else torch.mlu.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype) " |
| 97 | src/transformers/trainer.py:2613 | change "ctx_manager = torch.cuda.amp.autocast()" to "ctx_manager = torch.mlu.amp.autocast() " |
| 98 | src/transformers/activations.py:18 | add "import torch_mlu" |
| 99 | src/transformers/trainer_seq2seq.py:17 | add "import torch_mlu" |
| 100 | src/transformers/file_utils.py:115 | change "is_torch_cuda_available," to "is_torch_mlu_available, " |
| 101 | src/transformers/modeling_flax_pytorch_utils.py:205 | add "import torch_mlu" |
| 102 | src/transformers/deepspeed.py:28 | add "import torch_mlu" |
| 103 | src/transformers/image_transforms.py:45 | add "import torch_mlu" |
| 104 | src/transformers/time_series_utils.py:21 | add "import torch_mlu" |
| 105 | src/transformers/training_args_tf.py:125 | change "no_cuda (`bool`, *optional*, defaults to `False`):" to "no_mlu (`bool`, *optional*, defaults to `False`): " |
| 106 | src/transformers/training_args_tf.py:198 | change "if self.no_cuda:" to "if self.no_mlu: " |
| 107 | src/transformers/benchmark/benchmark_utils.py:41 | change "from torch.cuda import empty_cache as torch_empty_cache" to "from torch.mlu import empty_cache as torch_empty_cache " |
| 108 | src/transformers/benchmark/benchmark_args_utils.py:66 | change "cuda: bool = field(" to "mlu: bool = field( " |
| 109 | src/transformers/benchmark/benchmark_args_utils.py:68 | change "metadata={"help": "Whether to run on available cuda devices. Cuda can be disabled via --no-cuda."}," to "metadata={"help": "Whether to run on available mlu devices. Cuda can be disabled via --no-mlu."}, " |
| 110 | src/transformers/benchmark/benchmark_args_tf.py:35 | change ""no_cuda"," to ""no_mlu", " |
| 111 | src/transformers/benchmark/benchmark_args_tf.py:130 | change "if self.cuda:" to "if self.mlu: " |
| 112 | src/transformers/benchmark/benchmark_args.py:25 | add "import torch_mlu" |
| 113 | src/transformers/benchmark/benchmark_args.py:38 | change ""no_cuda"," to ""no_mlu", " |
| 114 | src/transformers/benchmark/benchmark_args.py:81 | change "if not self.cuda:" to "if not self.mlu: " |
| 115 | src/transformers/benchmark/benchmark_args.py:88 | change "device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 116 | src/transformers/benchmark/benchmark_args.py:89 | change "n_gpu = torch.cuda.device_count()" to "n_gpu = torch.mlu.device_count() " |
| 117 | src/transformers/benchmark/benchmark_args.py:100 | change "return torch.cuda.current_device()" to "return torch.mlu.current_device() " |
| 118 | src/transformers/benchmark/benchmark.py:38 | add "import torch_mlu" |
| 119 | src/transformers/sagemaker/training_args_sm.py:21 | add "import torch_mlu" |
| 120 | src/transformers/sagemaker/training_args_sm.py:85 | change "if self.no_cuda:" to "if self.no_mlu: " |
| 121 | src/transformers/sagemaker/training_args_sm.py:90 | change "device = torch.device("cuda", local_rank)" to "device = torch.device("mlu", local_rank) " |
| 122 | src/transformers/sagemaker/training_args_sm.py:97 | change "device = torch.device("cuda", self.local_rank)" to "device = torch.device("mlu", self.local_rank) " |
| 123 | src/transformers/sagemaker/training_args_sm.py:104 | change "# GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`" to "# GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `mlu:0` " |
| 124 | src/transformers/sagemaker/training_args_sm.py:106 | change "device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 125 | src/transformers/sagemaker/training_args_sm.py:109 | change "self._n_gpu = torch.cuda.device_count()" to "self._n_gpu = torch.mlu.device_count() " |
| 126 | src/transformers/sagemaker/training_args_sm.py:114 | change "torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)" to "torch.distributed.init_process_group(backend="cncl", timeout=self.ddp_timeout_delta) " |
| 127 | src/transformers/sagemaker/training_args_sm.py:115 | change "device = torch.device("cuda", self.local_rank)" to "device = torch.device("mlu", self.local_rank) " |
| 128 | src/transformers/sagemaker/training_args_sm.py:118 | change "if device.type == "cuda":" to "if device.type == "mlu": " |
| 129 | src/transformers/sagemaker/training_args_sm.py:119 | change "torch.cuda.set_device(device)" to "torch.mlu.set_device(device) " |
| 130 | src/transformers/generation/stopping_criteria.py:7 | add "import torch_mlu" |
| 131 | src/transformers/generation/logits_process.py:21 | add "import torch_mlu" |
| 132 | src/transformers/generation/beam_search.py:21 | add "import torch_mlu" |
| 133 | src/transformers/generation/beam_search.py:135 | change "Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be" to "Defines the device type (*e.g.*, `"cpu"` or `"mlu"`) on which this instance of `BeamSearchScorer` will be " |
| 134 | src/transformers/generation/beam_search.py:409 | change "Defines the device type (*e.g.*, `"cpu"` or `"cuda"`) on which this instance of `BeamSearchScorer` will be" to "Defines the device type (*e.g.*, `"cpu"` or `"mlu"`) on which this instance of `BeamSearchScorer` will be " |
| 135 | src/transformers/generation/utils.py:23 | add "import torch_mlu" |
| 136 | src/transformers/onnx/convert.py:108 | change "The device on which the ONNX model will be exported. Either `cpu` or `cuda`." to "The device on which the ONNX model will be exported. Either `cpu` or `mlu`. " |
| 137 | src/transformers/onnx/convert.py:127 | add "import torch_mlu" |
| 138 | src/transformers/onnx/convert.py:146 | change "if device.type == "cuda" and torch.cuda.is_available():" to "if device.type == "mlu" and torch.mlu.is_available(): " |
| 139 | src/transformers/onnx/convert.py:312 | change "The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for" to "The device on which the ONNX model will be exported. Either `cpu` or `mlu`. Only PyTorch is supported for " |
| 140 | src/transformers/onnx/convert.py:325 | change "if is_tf_available() and isinstance(model, TFPreTrainedModel) and device == "cuda":" to "if is_tf_available() and isinstance(model, TFPreTrainedModel) and device == "mlu": " |
| 141 | src/transformers/onnx/config.py:525 | add "import torch_mlu" |
| 142 | src/transformers/commands/pt_to_tf.py:48 | add "import torch_mlu" |
| 143 | src/transformers/commands/env.py:37 | change "pt_cuda_available = "NA"" to "pt_mlu_available = "NA" " |
| 144 | src/transformers/commands/env.py:39 | add "import torch_mlu" |
| 145 | src/transformers/commands/env.py:42 | change "pt_cuda_available = torch.cuda.is_available()" to "pt_mlu_available = torch.mlu.is_available() " |
| 146 | src/transformers/commands/env.py:45 | change "tf_cuda_available = "NA"" to "tf_mlu_available = "NA" " |
| 147 | src/transformers/commands/env.py:52 | change "tf_cuda_available = tf.test.is_gpu_available()" to "tf_mlu_available = tf.test.is_gpu_available() " |
| 148 | src/transformers/commands/env.py:55 | change "tf_cuda_available = bool(tf.config.list_physical_devices("GPU"))" to "tf_mlu_available = bool(tf.config.list_physical_devices("GPU")) " |
| 149 | src/transformers/commands/env.py:76 | change ""PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})"," to ""PyTorch version (GPU?)": f"{pt_version} ({pt_mlu_available})", " |
| 150 | src/transformers/commands/env.py:77 | change ""Tensorflow version (GPU?)": f"{tf_version} ({tf_cuda_available})"," to ""Tensorflow version (GPU?)": f"{tf_version} ({tf_mlu_available})", " |
| 151 | src/transformers/models/flava/modeling_flava.py:23 | add "import torch_mlu" |
| 152 | src/transformers/models/flava/convert_dalle_to_flava_codebook.py:19 | add "import torch_mlu" |
| 153 | src/transformers/models/flava/convert_flava_original_pytorch_to_hf.py:19 | add "import torch_mlu" |
| 154 | src/transformers/models/albert/modeling_albert.py:22 | add "import torch_mlu" |
| 155 | src/transformers/models/albert/convert_albert_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 156 | src/transformers/models/esm/modeling_esmfold.py:22 | add "import torch_mlu" |
| 157 | src/transformers/models/esm/modeling_esmfold.py:326 | change "with torch.cuda.amp.autocast(enabled=False):" to "with torch.mlu.amp.autocast(enabled=False): " |
| 158 | src/transformers/models/esm/modeling_esmfold.py:341 | change "with torch.cuda.amp.autocast(enabled=False):" to "with torch.mlu.amp.autocast(enabled=False): " |
| 159 | src/transformers/models/esm/modeling_esmfold.py:926 | change "with torch.cuda.amp.autocast(enabled=False):" to "with torch.mlu.amp.autocast(enabled=False): " |
| 160 | src/transformers/models/esm/modeling_esmfold.py:1540 | change "with torch.cuda.amp.autocast(enabled=False):" to "with torch.mlu.amp.autocast(enabled=False): " |
| 161 | src/transformers/models/esm/modeling_esm.py:20 | add "import torch_mlu" |
| 162 | src/transformers/models/esm/convert_esm.py:24 | add "import torch_mlu" |
| 163 | src/transformers/models/esm/convert_esm.py:331 | change "their_output = esm.cuda().infer([row[1] for row in sample_data])" to "their_output = esm.mlu().infer([row[1] for row in sample_data]) " |
| 164 | src/transformers/models/esm/convert_esm.py:332 | change "our_output = model.cuda()(" to "our_output = model.mlu()( " |
| 165 | src/transformers/models/esm/convert_esm.py:333 | change "input_ids=hf_tokens["input_ids"].cuda(), attention_mask=hf_tokens["attention_mask"].cuda()" to "input_ids=hf_tokens["input_ids"].mlu(), attention_mask=hf_tokens["attention_mask"].mlu() " |
| 166 | src/transformers/models/esm/openfold_utils/tensor_utils.py:19 | add "import torch_mlu" |
| 167 | src/transformers/models/esm/openfold_utils/chunk_utils.py:19 | add "import torch_mlu" |
| 168 | src/transformers/models/esm/openfold_utils/feats.py:18 | add "import torch_mlu" |
| 169 | src/transformers/models/esm/openfold_utils/data_transforms.py:19 | add "import torch_mlu" |
| 170 | src/transformers/models/esm/openfold_utils/loss.py:18 | add "import torch_mlu" |
| 171 | src/transformers/models/esm/openfold_utils/rigid_utils.py:22 | add "import torch_mlu" |
| 172 | src/transformers/models/esm/openfold_utils/rigid_utils.py:670 | change "def cuda(self) -> Rotation:" to "def mlu(self) -> Rotation: " |
| 173 | src/transformers/models/esm/openfold_utils/rigid_utils.py:672 | change "Analogous to the cuda() method of torch Tensors" to "Analogous to the mlu() method of torch Tensors " |
| 174 | src/transformers/models/esm/openfold_utils/rigid_utils.py:678 | change "return Rotation(rot_mats=self._rot_mats.cuda(), quats=None)" to "return Rotation(rot_mats=self._rot_mats.mlu(), quats=None) " |
| 175 | src/transformers/models/esm/openfold_utils/rigid_utils.py:680 | change "return Rotation(rot_mats=None, quats=self._quats.cuda(), normalize_quats=False)" to "return Rotation(rot_mats=None, quats=self._quats.mlu(), normalize_quats=False) " |
| 176 | src/transformers/models/esm/openfold_utils/rigid_utils.py:1235 | change "def cuda(self) -> Rigid:" to "def mlu(self) -> Rigid: " |
| 177 | src/transformers/models/esm/openfold_utils/rigid_utils.py:1242 | change "return Rigid(self._rots.cuda(), self._trans.cuda())" to "return Rigid(self._rots.mlu(), self._trans.mlu()) " |
| 178 | src/transformers/models/deberta/modeling_deberta.py:20 | add "import torch_mlu" |
| 179 | src/transformers/models/pegasus_x/modeling_pegasus_x.py:23 | add "import torch_mlu" |
| 180 | src/transformers/models/led/modeling_led.py:24 | add "import torch_mlu" |
| 181 | src/transformers/models/hubert/convert_hubert_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 182 | src/transformers/models/hubert/modeling_hubert.py:21 | add "import torch_mlu" |
| 183 | src/transformers/models/hubert/modeling_hubert.py:1200 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 184 | src/transformers/models/hubert/convert_distilhubert_original_s3prl_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 185 | src/transformers/models/hubert/convert_hubert_original_s3prl_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 186 | src/transformers/models/decision_transformer/modeling_decision_transformer.py:22 | add "import torch_mlu" |
| 187 | src/transformers/models/decision_transformer/modeling_decision_transformer.py:25 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 188 | src/transformers/models/decision_transformer/modeling_decision_transformer.py:624 | change "torch.cuda.set_device(hidden_states.device)" to "torch.mlu.set_device(hidden_states.device) " |
| 189 | src/transformers/models/decision_transformer/modeling_decision_transformer.py:678 | change "if i == v[-1] and "cuda:" + str(k) != self.last_device:" to "if i == v[-1] and "mlu:" + str(k) != self.last_device: " |
| 190 | src/transformers/models/decision_transformer/modeling_decision_transformer.py:679 | change "hidden_states = hidden_states.to("cuda:" + str(k + 1))" to "hidden_states = hidden_states.to("mlu:" + str(k + 1)) " |
| 191 | src/transformers/models/electra/modeling_electra.py:22 | add "import torch_mlu" |
| 192 | src/transformers/models/electra/convert_electra_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 193 | src/transformers/models/rag/modeling_rag.py:21 | add "import torch_mlu" |
| 194 | src/transformers/models/gpt_sw3/tokenization_gpt_sw3.py:9 | add "import torch_mlu" |
| 195 | src/transformers/models/gpt_sw3/convert_megatron_to_pytorch.py:20 | add "import torch_mlu" |
| 196 | src/transformers/models/dialogpt/convert_dialogpt_original_pytorch_checkpoint_to_pytorch.py:18 | add "import torch_mlu" |
| 197 | src/transformers/models/speech_encoder_decoder/convert_mbart_wav2vec2_seq2seq_original_to_pytorch.py:21 | add "import torch_mlu" |
| 198 | src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py:20 | add "import torch_mlu" |
| 199 | src/transformers/models/speech_encoder_decoder/convert_speech_to_text_wav2vec2_seq2seq_original_to_pytorch.py:23 | add "import torch_mlu" |
| 200 | src/transformers/models/mask2former/modeling_mask2former.py:24 | add "import torch_mlu" |
| 201 | src/transformers/models/mask2former/image_processing_mask2former.py:57 | add "import torch_mlu" |
| 202 | src/transformers/models/mask2former/convert_mask2former_original_pytorch_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 203 | src/transformers/models/convbert/modeling_convbert.py:23 | add "import torch_mlu" |
| 204 | src/transformers/models/mctct/feature_extraction_mctct.py:22 | add "import torch_mlu" |
| 205 | src/transformers/models/mctct/modeling_mctct.py:22 | add "import torch_mlu" |
| 206 | src/transformers/models/mctct/modeling_mctct.py:809 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 207 | src/transformers/models/videomae/convert_videomae_to_pytorch.py:22 | add "import torch_mlu" |
| 208 | src/transformers/models/videomae/modeling_videomae.py:25 | add "import torch_mlu" |
| 209 | src/transformers/models/dit/convert_dit_unilm_to_pytorch.py:23 | add "import torch_mlu" |
| 210 | src/transformers/models/x_clip/modeling_x_clip.py:22 | add "import torch_mlu" |
| 211 | src/transformers/models/x_clip/convert_x_clip_original_pytorch_to_hf.py:20 | add "import torch_mlu" |
| 212 | src/transformers/models/dinat/modeling_dinat.py:22 | add "import torch_mlu" |
| 213 | src/transformers/models/mluke/convert_mluke_original_pytorch_checkpoint_to_pytorch.py:22 | add "import torch_mlu" |
| 214 | src/transformers/models/mgp_str/processing_mgp_str.py:27 | add "import torch_mlu" |
| 215 | src/transformers/models/mgp_str/modeling_mgp_str.py:21 | add "import torch_mlu" |
| 216 | src/transformers/models/oneformer/image_processing_oneformer.py:58 | add "import torch_mlu" |
| 217 | src/transformers/models/oneformer/modeling_oneformer.py:23 | add "import torch_mlu" |
| 218 | src/transformers/models/oneformer/modeling_oneformer.py:25 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 219 | src/transformers/models/oneformer/convert_to_hf_oneformer.py:27 | add "import torch_mlu" |
| 220 | src/transformers/models/oneformer/processing_oneformer.py:26 | add "import torch_mlu" |
| 221 | src/transformers/models/mpnet/modeling_mpnet.py:22 | add "import torch_mlu" |
| 222 | src/transformers/models/deit/convert_deit_timm_to_pytorch.py:24 | add "import torch_mlu" |
| 223 | src/transformers/models/deit/modeling_deit.py:23 | add "import torch_mlu" |
| 224 | src/transformers/models/table_transformer/modeling_table_transformer.py:23 | add "import torch_mlu" |
| 225 | src/transformers/models/table_transformer/convert_table_transformer_original_pytorch_checkpoint_to_pytorch.py:25 | add "import torch_mlu" |
| 226 | src/transformers/models/xlm/modeling_xlm.py:25 | add "import torch_mlu" |
| 227 | src/transformers/models/xlm/convert_xlm_original_pytorch_checkpoint_to_pytorch.py:22 | add "import torch_mlu" |
| 228 | src/transformers/models/xlnet/convert_xlnet_original_tf_checkpoint_to_pytorch.py:21 | add "import torch_mlu" |
| 229 | src/transformers/models/xlnet/modeling_xlnet.py:23 | add "import torch_mlu" |
| 230 | src/transformers/models/squeezebert/modeling_squeezebert.py:21 | add "import torch_mlu" |
| 231 | src/transformers/models/detr/image_processing_detr.py:67 | add "import torch_mlu" |
| 232 | src/transformers/models/detr/convert_detr_to_pytorch.py:23 | add "import torch_mlu" |
| 233 | src/transformers/models/detr/convert_detr_original_pytorch_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 234 | src/transformers/models/detr/modeling_detr.py:23 | add "import torch_mlu" |
| 235 | src/transformers/models/opt/modeling_opt.py:19 | add "import torch_mlu" |
| 236 | src/transformers/models/opt/convert_opt_original_pytorch_checkpoint_to_pytorch.py:21 | add "import torch_mlu" |
| 237 | src/transformers/models/reformer/convert_reformer_trax_checkpoint_to_pytorch.py:22 | add "import torch_mlu" |
| 238 | src/transformers/models/reformer/modeling_reformer.py:26 | add "import torch_mlu" |
| 239 | src/transformers/models/reformer/modeling_reformer.py:1433 | change "# use cuda generator if available" to "# use mlu generator if available " |
| 240 | src/transformers/models/reformer/modeling_reformer.py:1434 | change "if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:" to "if hasattr(torch.mlu, "default_generators") and len(torch.mlu.default_generators) > 0: " |
| 241 | src/transformers/models/reformer/modeling_reformer.py:1436 | change "device_idx = torch.cuda.current_device()" to "device_idx = torch.mlu.current_device() " |
| 242 | src/transformers/models/reformer/modeling_reformer.py:1437 | change "self.attention_seed = torch.cuda.default_generators[device_idx].seed()" to "self.attention_seed = torch.mlu.default_generators[device_idx].seed() " |
| 243 | src/transformers/models/reformer/modeling_reformer.py:1450 | change "# use cuda generator if available" to "# use mlu generator if available " |
| 244 | src/transformers/models/reformer/modeling_reformer.py:1451 | change "if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:" to "if hasattr(torch.mlu, "default_generators") and len(torch.mlu.default_generators) > 0: " |
| 245 | src/transformers/models/reformer/modeling_reformer.py:1453 | change "device_idx = torch.cuda.current_device()" to "device_idx = torch.mlu.current_device() " |
| 246 | src/transformers/models/reformer/modeling_reformer.py:1454 | change "self.feed_forward_seed = torch.cuda.default_generators[device_idx].seed()" to "self.feed_forward_seed = torch.mlu.default_generators[device_idx].seed() " |
| 247 | src/transformers/models/gpt_neox/modeling_gpt_neox.py:19 | add "import torch_mlu" |
| 248 | src/transformers/models/conditional_detr/image_processing_conditional_detr.py:68 | add "import torch_mlu" |
| 249 | src/transformers/models/conditional_detr/convert_conditional_detr_original_pytorch_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 250 | src/transformers/models/conditional_detr/modeling_conditional_detr.py:23 | add "import torch_mlu" |
| 251 | src/transformers/models/layoutlmv3/modeling_layoutlmv3.py:21 | add "import torch_mlu" |
| 252 | src/transformers/models/mbart/convert_mbart_original_checkpoint_to_pytorch.py:17 | add "import torch_mlu" |
| 253 | src/transformers/models/mbart/modeling_mbart.py:21 | add "import torch_mlu" |
| 254 | src/transformers/models/mbart/configuration_mbart.py:252 | add "import torch_mlu" |
| 255 | src/transformers/models/mt5/modeling_mt5.py:23 | add "import torch_mlu" |
| 256 | src/transformers/models/mt5/modeling_mt5.py:104 | change "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()" to "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.mlu.empty_cache() " |
| 257 | src/transformers/models/mt5/modeling_mt5.py:863 | change "get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map" to "get_device_map(len(self.block), range(torch.mlu.device_count())) if device_map is None else device_map " |
| 258 | src/transformers/models/mt5/modeling_mt5.py:867 | change "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))" to "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "mlu:" + str(min(self.device_map.keys())) " |
| 259 | src/transformers/models/mt5/modeling_mt5.py:868 | change "self.last_device = "cuda:" + str(max(self.device_map.keys()))" to "self.last_device = "mlu:" + str(max(self.device_map.keys())) " |
| 260 | src/transformers/models/mt5/modeling_mt5.py:872 | change "cuda_device = "cuda:" + str(k)" to "mlu_device = "mlu:" + str(k) " |
| 261 | src/transformers/models/mt5/modeling_mt5.py:873 | change "self.block[layer] = self.block[layer].to(cuda_device)" to "self.block[layer] = self.block[layer].to(mlu_device) " |
| 262 | src/transformers/models/mt5/modeling_mt5.py:894 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 263 | src/transformers/models/mt5/modeling_mt5.py:919 | change "torch.cuda.set_device(self.first_device)" to "torch.mlu.set_device(self.first_device) " |
| 264 | src/transformers/models/mt5/modeling_mt5.py:1005 | change "torch.cuda.set_device(hidden_states.device)" to "torch.mlu.set_device(hidden_states.device) " |
| 265 | src/transformers/models/mt5/modeling_mt5.py:1084 | change "if i == v[-1] and "cuda:" + str(k) != self.last_device:" to "if i == v[-1] and "mlu:" + str(k) != self.last_device: " |
| 266 | src/transformers/models/mt5/modeling_mt5.py:1085 | change "hidden_states = hidden_states.to("cuda:" + str(k + 1))" to "hidden_states = hidden_states.to("mlu:" + str(k + 1)) " |
| 267 | src/transformers/models/mt5/modeling_mt5.py:1346 | change "get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))" to "get_device_map(len(self.encoder.block), range(torch.mlu.device_count())) " |
| 268 | src/transformers/models/mt5/modeling_mt5.py:1368 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 269 | src/transformers/models/mt5/modeling_mt5.py:1473 | change "torch.cuda.set_device(self.decoder.first_device)" to "torch.mlu.set_device(self.decoder.first_device) " |
| 270 | src/transformers/models/mt5/modeling_mt5.py:1582 | change "get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))" to "get_device_map(len(self.encoder.block), range(torch.mlu.device_count())) " |
| 271 | src/transformers/models/mt5/modeling_mt5.py:1606 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 272 | src/transformers/models/mt5/modeling_mt5.py:1718 | change "torch.cuda.set_device(self.decoder.first_device)" to "torch.mlu.set_device(self.decoder.first_device) " |
| 273 | src/transformers/models/mt5/modeling_mt5.py:1726 | change "torch.cuda.set_device(self.decoder.first_device)" to "torch.mlu.set_device(self.decoder.first_device) " |
| 274 | src/transformers/models/mt5/modeling_mt5.py:1755 | change "torch.cuda.set_device(self.encoder.first_device)" to "torch.mlu.set_device(self.encoder.first_device) " |
| 275 | src/transformers/models/mt5/modeling_mt5.py:1903 | change "get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))" to "get_device_map(len(self.encoder.block), range(torch.mlu.device_count())) " |
| 276 | src/transformers/models/mt5/modeling_mt5.py:1922 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 277 | src/transformers/models/retribert/modeling_retribert.py:23 | add "import torch_mlu" |
| 278 | src/transformers/models/tvlt/modeling_tvlt.py:24 | add "import torch_mlu" |
| 279 | src/transformers/models/sew/convert_sew_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 280 | src/transformers/models/sew/modeling_sew.py:22 | add "import torch_mlu" |
| 281 | src/transformers/models/sew/modeling_sew.py:1073 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 282 | src/transformers/models/xglm/modeling_xglm.py:22 | add "import torch_mlu" |
| 283 | src/transformers/models/xglm/convert_xglm_original_ckpt_to_trfms.py:4 | add "import torch_mlu" |
| 284 | src/transformers/models/trajectory_transformer/convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py:17 | add "import torch_mlu" |
| 285 | src/transformers/models/trajectory_transformer/modeling_trajectory_transformer.py:23 | add "import torch_mlu" |
| 286 | src/transformers/models/xlm_roberta_xl/convert_xlm_roberta_xl_original_pytorch_checkpoint_to_pytorch.py:21 | add "import torch_mlu" |
| 287 | src/transformers/models/xlm_roberta_xl/modeling_xlm_roberta_xl.py:20 | add "import torch_mlu" |
| 288 | src/transformers/models/beit/modeling_beit.py:23 | add "import torch_mlu" |
| 289 | src/transformers/models/beit/image_processing_beit.py:41 | add "import torch_mlu" |
| 290 | src/transformers/models/beit/convert_beit_unilm_to_pytorch.py:23 | add "import torch_mlu" |
| 291 | src/transformers/models/dpt/convert_dpt_hybrid_to_pytorch.py:23 | add "import torch_mlu" |
| 292 | src/transformers/models/dpt/convert_dpt_to_pytorch.py:23 | add "import torch_mlu" |
| 293 | src/transformers/models/dpt/image_processing_dpt.py:41 | add "import torch_mlu" |
| 294 | src/transformers/models/dpt/modeling_dpt.py:28 | add "import torch_mlu" |
| 295 | src/transformers/models/transfo_xl/modeling_transfo_xl_utilities.py:21 | add "import torch_mlu" |
| 296 | src/transformers/models/transfo_xl/modeling_transfo_xl_utilities.py:25 | change "# CUDA_MAJOR = int(torch.version.cuda.split('.')[0])" to "# CUDA_MAJOR = int(torch.version.mlu.split('.')[0]) " |
| 297 | src/transformers/models/transfo_xl/modeling_transfo_xl_utilities.py:26 | change "# CUDA_MINOR = int(torch.version.cuda.split('.')[1])" to "# CUDA_MINOR = int(torch.version.mlu.split('.')[1]) " |
| 298 | src/transformers/models/transfo_xl/modeling_transfo_xl.py:24 | add "import torch_mlu" |
| 299 | src/transformers/models/transfo_xl/tokenization_transfo_xl.py:46 | add "import torch_mlu" |
| 300 | src/transformers/models/transfo_xl/convert_transfo_xl_original_tf_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 301 | src/transformers/models/resnet/convert_resnet_to_pytorch.py:26 | add "import torch_mlu" |
| 302 | src/transformers/models/resnet/modeling_resnet.py:19 | add "import torch_mlu" |
| 303 | src/transformers/models/qdqbert/modeling_qdqbert.py:24 | add "import torch_mlu" |
| 304 | src/transformers/models/clap/feature_extraction_clap.py:22 | add "import torch_mlu" |
| 305 | src/transformers/models/clap/convert_clap_original_pytorch_to_hf.py:19 | add "import torch_mlu" |
| 306 | src/transformers/models/clap/convert_clap_original_pytorch_to_hf.py:47 | change "device="cuda:0" if torch.cuda.is_available() else "cpu"," to "device="mlu:0" if torch.mlu.is_available() else "cpu", " |
| 307 | src/transformers/models/clap/modeling_clap.py:22 | add "import torch_mlu" |
| 308 | src/transformers/models/funnel/modeling_funnel.py:22 | add "import torch_mlu" |
| 309 | src/transformers/models/funnel/convert_funnel_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 310 | src/transformers/models/owlvit/image_processing_owlvit.py:45 | add "import torch_mlu" |
| 311 | src/transformers/models/owlvit/modeling_owlvit.py:23 | add "import torch_mlu" |
| 312 | src/transformers/models/owlvit/convert_owlvit_original_flax_to_hf.py:23 | add "import torch_mlu" |
| 313 | src/transformers/models/owlvit/processing_owlvit.py:135 | add "import torch_mlu" |
| 314 | src/transformers/models/speech_to_text/modeling_speech_to_text.py:22 | add "import torch_mlu" |
| 315 | src/transformers/models/speech_to_text/convert_s2t_fairseq_to_tfms.py:17 | add "import torch_mlu" |
| 316 | src/transformers/models/speech_to_text/feature_extraction_speech_to_text.py:22 | add "import torch_mlu" |
| 317 | src/transformers/models/sew_d/convert_sew_d_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 318 | src/transformers/models/sew_d/modeling_sew_d.py:23 | add "import torch_mlu" |
| 319 | src/transformers/models/sew_d/modeling_sew_d.py:1613 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 320 | src/transformers/models/roberta_prelayernorm/convert_roberta_prelayernorm_original_pytorch_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 321 | src/transformers/models/roberta_prelayernorm/modeling_roberta_prelayernorm.py:21 | add "import torch_mlu" |
| 322 | src/transformers/models/deberta_v2/modeling_deberta_v2.py:20 | add "import torch_mlu" |
| 323 | src/transformers/models/plbart/convert_plbart_original_checkpoint_to_torch.py:17 | add "import torch_mlu" |
| 324 | src/transformers/models/plbart/modeling_plbart.py:21 | add "import torch_mlu" |
| 325 | src/transformers/models/bart/modeling_bart.py:22 | add "import torch_mlu" |
| 326 | src/transformers/models/bart/configuration_bart.py:267 | add "import torch_mlu" |
| 327 | src/transformers/models/bart/convert_bart_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 328 | src/transformers/models/lilt/modeling_lilt.py:20 | add "import torch_mlu" |
| 329 | src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py:20 | add "import torch_mlu" |
| 330 | src/transformers/models/ernie_m/modeling_ernie_m.py:21 | add "import torch_mlu" |
| 331 | src/transformers/models/convnextv2/convert_convnextv2_to_pytorch.py:24 | add "import torch_mlu" |
| 332 | src/transformers/models/convnextv2/modeling_convnextv2.py:20 | add "import torch_mlu" |
| 333 | src/transformers/models/bigbird_pegasus/convert_bigbird_pegasus_tf_to_pytorch.py:20 | add "import torch_mlu" |
| 334 | src/transformers/models/bigbird_pegasus/configuration_bigbird_pegasus.py:283 | add "import torch_mlu" |
| 335 | src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py:24 | add "import torch_mlu" |
| 336 | src/transformers/models/realm/modeling_realm.py:22 | add "import torch_mlu" |
| 337 | src/transformers/models/switch_transformers/convert_big_switch.py:6 | add "import torch_mlu" |
| 338 | src/transformers/models/switch_transformers/modeling_switch_transformers.py:23 | add "import torch_mlu" |
| 339 | src/transformers/models/xlm_prophetnet/modeling_xlm_prophetnet.py:24 | add "import torch_mlu" |
| 340 | src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py:40 | add "import torch_mlu" |
| 341 | src/transformers/models/megatron_bert/modeling_megatron_bert.py:25 | add "import torch_mlu" |
| 342 | src/transformers/models/longt5/modeling_longt5.py:23 | add "import torch_mlu" |
| 343 | src/transformers/models/gpt2/convert_gpt2_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 344 | src/transformers/models/gpt2/modeling_gpt2.py:24 | add "import torch_mlu" |
| 345 | src/transformers/models/gpt2/modeling_gpt2.py:27 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 346 | src/transformers/models/gpt2/modeling_gpt2.py:658 | change "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()" to "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.mlu.empty_cache() " |
| 347 | src/transformers/models/gpt2/modeling_gpt2.py:701 | change "get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map" to "get_device_map(len(self.h), range(torch.mlu.device_count())) if device_map is None else device_map " |
| 348 | src/transformers/models/gpt2/modeling_gpt2.py:705 | change "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))" to "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "mlu:" + str(min(self.device_map.keys())) " |
| 349 | src/transformers/models/gpt2/modeling_gpt2.py:706 | change "self.last_device = "cuda:" + str(max(self.device_map.keys()))" to "self.last_device = "mlu:" + str(max(self.device_map.keys())) " |
| 350 | src/transformers/models/gpt2/modeling_gpt2.py:712 | change "cuda_device = "cuda:" + str(k)" to "mlu_device = "mlu:" + str(k) " |
| 351 | src/transformers/models/gpt2/modeling_gpt2.py:713 | change "self.h[block] = self.h[block].to(cuda_device)" to "self.h[block] = self.h[block].to(mlu_device) " |
| 352 | src/transformers/models/gpt2/modeling_gpt2.py:732 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 353 | src/transformers/models/gpt2/modeling_gpt2.py:868 | change "torch.cuda.set_device(hidden_states.device)" to "torch.mlu.set_device(hidden_states.device) " |
| 354 | src/transformers/models/gpt2/modeling_gpt2.py:922 | change "if i == v[-1] and "cuda:" + str(k) != self.last_device:" to "if i == v[-1] and "mlu:" + str(k) != self.last_device: " |
| 355 | src/transformers/models/gpt2/modeling_gpt2.py:923 | change "hidden_states = hidden_states.to("cuda:" + str(k + 1))" to "hidden_states = hidden_states.to("mlu:" + str(k + 1)) " |
| 356 | src/transformers/models/gpt2/modeling_gpt2.py:980 | change "get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))" to "get_device_map(len(self.transformer.h), range(torch.mlu.device_count())) " |
| 357 | src/transformers/models/gpt2/modeling_gpt2.py:999 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 358 | src/transformers/models/gpt2/modeling_gpt2.py:1094 | change "torch.cuda.set_device(self.transformer.first_device)" to "torch.mlu.set_device(self.transformer.first_device) " |
| 359 | src/transformers/models/gpt2/modeling_gpt2.py:1172 | change "get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))" to "get_device_map(len(self.transformer.h), range(torch.mlu.device_count())) " |
| 360 | src/transformers/models/gpt2/modeling_gpt2.py:1193 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 361 | src/transformers/models/gpt2/modeling_gpt2.py:1309 | change "torch.cuda.set_device(self.transformer.first_device)" to "torch.mlu.set_device(self.transformer.first_device) " |
| 362 | src/transformers/models/gpt2/configuration_gpt2.py:247 | add "import torch_mlu" |
| 363 | src/transformers/models/openai/convert_openai_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 364 | src/transformers/models/openai/modeling_openai.py:25 | add "import torch_mlu" |
| 365 | src/transformers/models/jukebox/tokenization_jukebox.py:314 | add "import torch_mlu" |
| 366 | src/transformers/models/jukebox/convert_jukebox.py:23 | add "import torch_mlu" |
| 367 | src/transformers/models/jukebox/modeling_jukebox.py:22 | add "import torch_mlu" |
| 368 | src/transformers/models/vit_mae/convert_vit_mae_to_pytorch.py:20 | add "import torch_mlu" |
| 369 | src/transformers/models/vit_mae/modeling_vit_mae.py:25 | add "import torch_mlu" |
| 370 | src/transformers/models/nezha/modeling_nezha.py:24 | add "import torch_mlu" |
| 371 | src/transformers/models/gpt_neox_japanese/modeling_gpt_neox_japanese.py:19 | add "import torch_mlu" |
| 372 | src/transformers/models/vilt/modeling_vilt.py:22 | add "import torch_mlu" |
| 373 | src/transformers/models/vilt/convert_vilt_original_to_pytorch.py:23 | add "import torch_mlu" |
| 374 | src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py:23 | add "import torch_mlu" |
| 375 | src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py:169 | add "import torch_mlu" |
| 376 | src/transformers/models/convnext/convert_convnext_to_pytorch.py:25 | add "import torch_mlu" |
| 377 | src/transformers/models/convnext/modeling_convnext.py:20 | add "import torch_mlu" |
| 378 | src/transformers/models/blip_2/modeling_blip_2.py:21 | add "import torch_mlu" |
| 379 | src/transformers/models/blip_2/modeling_blip_2.py:1267 | change ">>> device = "cuda" if torch.cuda.is_available() else "cpu"" to ">>> device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 380 | src/transformers/models/blip_2/modeling_blip_2.py:1328 | change ">>> device = "cuda" if torch.cuda.is_available() else "cpu"" to ">>> device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 381 | src/transformers/models/blip_2/modeling_blip_2.py:1376 | change ">>> device = "cuda" if torch.cuda.is_available() else "cpu"" to ">>> device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 382 | src/transformers/models/blip_2/modeling_blip_2.py:1442 | change ">>> device = "cuda" if torch.cuda.is_available() else "cpu"" to ">>> device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 383 | src/transformers/models/blip_2/modeling_blip_2.py:1605 | change "if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.cuda.device_count() > 1:" to "if len(hf_device_map) > 1 and "language_model" not in hf_device_map and torch.mlu.device_count() > 1: " |
| 384 | src/transformers/models/blip_2/modeling_blip_2.py:1645 | change ">>> device = "cuda" if torch.cuda.is_available() else "cpu"" to ">>> device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 385 | src/transformers/models/blip_2/modeling_blip_2.py:1672 | change ">>> device = "cuda" if torch.cuda.is_available() else "cpu"" to ">>> device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 386 | src/transformers/models/blip_2/convert_blip_2_original_to_pytorch.py:24 | add "import torch_mlu" |
| 387 | src/transformers/models/blip_2/convert_blip_2_original_to_pytorch.py:150 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 388 | src/transformers/models/bloom/modeling_bloom.py:21 | add "import torch_mlu" |
| 389 | src/transformers/models/bloom/configuration_bloom.py:211 | add "import torch_mlu" |
| 390 | src/transformers/models/bloom/convert_bloom_original_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 391 | src/transformers/models/nystromformer/convert_nystromformer_original_pytorch_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 392 | src/transformers/models/nystromformer/modeling_nystromformer.py:21 | add "import torch_mlu" |
| 393 | src/transformers/models/deformable_detr/convert_deformable_detr_to_pytorch.py:23 | add "import torch_mlu" |
| 394 | src/transformers/models/deformable_detr/convert_deformable_detr_to_pytorch.py:147 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 395 | src/transformers/models/deformable_detr/modeling_deformable_detr.py:24 | add "import torch_mlu" |
| 396 | src/transformers/models/deformable_detr/modeling_deformable_detr.py:37 | change "is_torch_cuda_available," to "is_torch_mlu_available, " |
| 397 | src/transformers/models/deformable_detr/modeling_deformable_detr.py:48 | change "from .load_custom import load_cuda_kernels" to "from .load_custom import load_mlu_kernels " |
| 398 | src/transformers/models/deformable_detr/modeling_deformable_detr.py:54 | change "if is_torch_cuda_available() and is_ninja_available():" to "if is_torch_mlu_available() and is_ninja_available(): " |
| 399 | src/transformers/models/deformable_detr/modeling_deformable_detr.py:57 | change "MultiScaleDeformableAttention = load_cuda_kernels()" to "MultiScaleDeformableAttention = load_mlu_kernels() " |
| 400 | src/transformers/models/deformable_detr/image_processing_deformable_detr.py:68 | add "import torch_mlu" |
| 401 | src/transformers/models/deformable_detr/load_custom.py:20 | change "def load_cuda_kernels():" to "def load_mlu_kernels(): " |
| 402 | src/transformers/models/deformable_detr/load_custom.py:29 | change "os.path.join("cuda", "ms_deform_attn_cuda.cu")," to "os.path.join("mlu", "ms_deform_attn_mlu.cu"), " |
| 403 | src/transformers/models/deformable_detr/load_custom.py:37 | change "with_cuda=True," to "with_mlu=True, " |
| 404 | src/transformers/models/deformable_detr/load_custom.py:41 | change "extra_cuda_cflags=[" to "extra_mlu_cflags=[ " |
| 405 | src/transformers/models/codegen/modeling_codegen.py:19 | add "import torch_mlu" |
| 406 | src/transformers/models/codegen/tokenization_codegen.py:31 | add "import torch_mlu" |
| 407 | src/transformers/models/codegen/configuration_codegen.py:206 | add "import torch_mlu" |
| 408 | src/transformers/models/codegen/tokenization_codegen_fast.py:29 | add "import torch_mlu" |
| 409 | src/transformers/models/chinese_clip/convert_chinese_clip_original_pytorch_to_hf.py:18 | add "import torch_mlu" |
| 410 | src/transformers/models/chinese_clip/modeling_chinese_clip.py:22 | add "import torch_mlu" |
| 411 | src/transformers/models/roberta/convert_roberta_original_pytorch_checkpoint_to_pytorch.py:22 | add "import torch_mlu" |
| 412 | src/transformers/models/roberta/modeling_roberta.py:21 | add "import torch_mlu" |
| 413 | src/transformers/models/mobilevit/image_processing_mobilevit.py:39 | add "import torch_mlu" |
| 414 | src/transformers/models/mobilevit/modeling_mobilevit.py:23 | add "import torch_mlu" |
| 415 | src/transformers/models/mobilevit/convert_mlcvnets_to_pytorch.py:23 | add "import torch_mlu" |
| 416 | src/transformers/models/blip/convert_blip_original_pytorch_to_hf.py:20 | add "import torch_mlu" |
| 417 | src/transformers/models/blip/modeling_blip.py:20 | add "import torch_mlu" |
| 418 | src/transformers/models/blip/modeling_blip_text.py:20 | add "import torch_mlu" |
| 419 | src/transformers/models/luke/convert_luke_original_pytorch_checkpoint_to_pytorch.py:21 | add "import torch_mlu" |
| 420 | src/transformers/models/luke/modeling_luke.py:21 | add "import torch_mlu" |
| 421 | src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py:44 | add "import torch_mlu" |
| 422 | src/transformers/models/glpn/modeling_glpn.py:21 | add "import torch_mlu" |
| 423 | src/transformers/models/glpn/convert_glpn_to_pytorch.py:23 | add "import torch_mlu" |
| 424 | src/transformers/models/fnet/convert_fnet_original_flax_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 425 | src/transformers/models/fnet/modeling_fnet.py:22 | add "import torch_mlu" |
| 426 | src/transformers/models/unispeech_sat/modeling_unispeech_sat.py:23 | add "import torch_mlu" |
| 427 | src/transformers/models/unispeech_sat/modeling_unispeech_sat.py:1451 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 428 | src/transformers/models/unispeech_sat/convert_unispeech_original_s3prl_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 429 | src/transformers/models/unispeech_sat/convert_unispeech_sat_original_pytorch_checkpoint_to_pytorch.py:21 | add "import torch_mlu" |
| 430 | src/transformers/models/mvp/modeling_mvp.py:21 | add "import torch_mlu" |
| 431 | src/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py:23 | add "import torch_mlu" |
| 432 | src/transformers/models/swin2sr/convert_swin2sr_original_to_pytorch.py:20 | add "import torch_mlu" |
| 433 | src/transformers/models/swin2sr/modeling_swin2sr.py:23 | add "import torch_mlu" |
| 434 | src/transformers/models/graphormer/collating_graphormer.py:7 | add "import torch_mlu" |
| 435 | src/transformers/models/graphormer/modeling_graphormer.py:21 | add "import torch_mlu" |
| 436 | src/transformers/models/megatron_gpt2/checkpoint_reshaping_and_interoperability.py:22 | add "import torch_mlu" |
| 437 | src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py:40 | add "import torch_mlu" |
| 438 | src/transformers/models/whisper/convert_openai_to_hf.py:21 | add "import torch_mlu" |
| 439 | src/transformers/models/whisper/modeling_whisper.py:22 | add "import torch_mlu" |
| 440 | src/transformers/models/vit_msn/modeling_vit_msn.py:22 | add "import torch_mlu" |
| 441 | src/transformers/models/vit_msn/convert_msn_to_pytorch.py:21 | add "import torch_mlu" |
| 442 | src/transformers/models/levit/convert_levit_timm_to_pytorch.py:25 | add "import torch_mlu" |
| 443 | src/transformers/models/levit/modeling_levit.py:21 | add "import torch_mlu" |
| 444 | src/transformers/models/bert_generation/modeling_bert_generation.py:20 | add "import torch_mlu" |
| 445 | src/transformers/models/lxmert/convert_lxmert_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 446 | src/transformers/models/lxmert/modeling_lxmert.py:24 | add "import torch_mlu" |
| 447 | src/transformers/models/unispeech/convert_unispeech_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 448 | src/transformers/models/unispeech/modeling_unispeech.py:23 | add "import torch_mlu" |
| 449 | src/transformers/models/unispeech/modeling_unispeech.py:1444 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 450 | src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py:21 | add "import torch_mlu" |
| 451 | src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py:1154 | change ">>> device = "cuda"" to ">>> device = "mlu" " |
| 452 | src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py:1169 | change ">>> device = "cuda"" to ">>> device = "mlu" " |
| 453 | src/transformers/models/gptsan_japanese/modeling_gptsan_japanese.py:1185 | change ">>> device = "cuda"" to ">>> device = "mlu" " |
| 454 | src/transformers/models/gptsan_japanese/convert_gptsan_tf_checkpoint_to_pytorch.py:25 | add "import torch_mlu" |
| 455 | src/transformers/models/wavlm/convert_wavlm_original_s3prl_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 456 | src/transformers/models/wavlm/convert_wavlm_original_pytorch_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 457 | src/transformers/models/wavlm/modeling_wavlm.py:22 | add "import torch_mlu" |
| 458 | src/transformers/models/wavlm/modeling_wavlm.py:1372 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 459 | src/transformers/models/tapas/modeling_tapas.py:24 | add "import torch_mlu" |
| 460 | src/transformers/models/ibert/quant_modules.py:21 | add "import torch_mlu" |
| 461 | src/transformers/models/ibert/modeling_ibert.py:23 | add "import torch_mlu" |
| 462 | src/transformers/models/gpt_neo/configuration_gpt_neo.py:165 | add "import torch_mlu" |
| 463 | src/transformers/models/gpt_neo/modeling_gpt_neo.py:21 | add "import torch_mlu" |
| 464 | src/transformers/models/markuplm/modeling_markuplm.py:21 | add "import torch_mlu" |
| 465 | src/transformers/models/yolos/convert_yolos_to_pytorch.py:23 | add "import torch_mlu" |
| 466 | src/transformers/models/yolos/image_processing_yolos.py:67 | add "import torch_mlu" |
| 467 | src/transformers/models/yolos/modeling_yolos.py:23 | add "import torch_mlu" |
| 468 | src/transformers/models/big_bird/modeling_big_bird.py:24 | add "import torch_mlu" |
| 469 | src/transformers/models/splinter/modeling_splinter.py:22 | add "import torch_mlu" |
| 470 | src/transformers/models/maskformer/modeling_maskformer_swin.py:24 | add "import torch_mlu" |
| 471 | src/transformers/models/maskformer/modeling_maskformer.py:24 | add "import torch_mlu" |
| 472 | src/transformers/models/maskformer/convert_maskformer_swin_to_pytorch.py:25 | add "import torch_mlu" |
| 473 | src/transformers/models/maskformer/convert_maskformer_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 474 | src/transformers/models/maskformer/image_processing_maskformer.py:61 | add "import torch_mlu" |
| 475 | src/transformers/models/maskformer/convert_maskformer_resnet_to_pytorch.py:25 | add "import torch_mlu" |
| 476 | src/transformers/models/mobilebert/modeling_mobilebert.py:29 | add "import torch_mlu" |
| 477 | src/transformers/models/mobilebert/convert_mobilebert_original_tf_checkpoint_to_pytorch.py:17 | add "import torch_mlu" |
| 478 | src/transformers/models/git/modeling_git.py:23 | add "import torch_mlu" |
| 479 | src/transformers/models/git/convert_git_to_pytorch.py:25 | add "import torch_mlu" |
| 480 | src/transformers/models/altclip/modeling_altclip.py:20 | add "import torch_mlu" |
| 481 | src/transformers/models/vit/modeling_vit.py:22 | add "import torch_mlu" |
| 482 | src/transformers/models/vit/convert_dino_to_pytorch.py:23 | add "import torch_mlu" |
| 483 | src/transformers/models/vit/convert_vit_timm_to_pytorch.py:24 | add "import torch_mlu" |
| 484 | src/transformers/models/donut/modeling_donut_swin.py:25 | add "import torch_mlu" |
| 485 | src/transformers/models/donut/convert_donut_to_pytorch.py:19 | add "import torch_mlu" |
| 486 | src/transformers/models/dpr/convert_dpr_original_checkpoint_to_pytorch.py:19 | add "import torch_mlu" |
| 487 | src/transformers/models/dpr/modeling_dpr.py:21 | add "import torch_mlu" |
| 488 | src/transformers/models/roc_bert/modeling_roc_bert.py:21 | add "import torch_mlu" |
| 489 | src/transformers/models/van/modeling_van.py:21 | add "import torch_mlu" |
| 490 | src/transformers/models/van/convert_van_to_pytorch.py:28 | add "import torch_mlu" |
| 491 | src/transformers/models/imagegpt/convert_imagegpt_original_tf2_to_pytorch.py:20 | add "import torch_mlu" |
| 492 | src/transformers/models/imagegpt/modeling_imagegpt.py:22 | add "import torch_mlu" |
| 493 | src/transformers/models/imagegpt/modeling_imagegpt.py:25 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 494 | src/transformers/models/imagegpt/modeling_imagegpt.py:808 | change "torch.cuda.set_device(hidden_states.device)" to "torch.mlu.set_device(hidden_states.device) " |
| 495 | src/transformers/models/imagegpt/modeling_imagegpt.py:862 | change "if i == v[-1] and "cuda:" + str(k) != self.last_device:" to "if i == v[-1] and "mlu:" + str(k) != self.last_device: " |
| 496 | src/transformers/models/imagegpt/modeling_imagegpt.py:863 | change "hidden_states = hidden_states.to("cuda:" + str(k + 1))" to "hidden_states = hidden_states.to("mlu:" + str(k + 1)) " |
| 497 | src/transformers/models/imagegpt/modeling_imagegpt.py:981 | change ">>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to ">>> device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 498 | src/transformers/models/align/convert_align_tf_to_hf.py:24 | add "import torch_mlu" |
| 499 | src/transformers/models/align/modeling_align.py:21 | add "import torch_mlu" |
| 500 | src/transformers/models/visual_bert/modeling_visual_bert.py:22 | add "import torch_mlu" |
| 501 | src/transformers/models/visual_bert/convert_visual_bert_original_pytorch_checkpoint_to_pytorch.py:22 | add "import torch_mlu" |
| 502 | src/transformers/models/t5/modeling_t5.py:24 | add "import torch_mlu" |
| 503 | src/transformers/models/t5/modeling_t5.py:233 | change "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()" to "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.mlu.empty_cache() " |
| 504 | src/transformers/models/t5/modeling_t5.py:892 | change "get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map" to "get_device_map(len(self.block), range(torch.mlu.device_count())) if device_map is None else device_map " |
| 505 | src/transformers/models/t5/modeling_t5.py:896 | change "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))" to "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "mlu:" + str(min(self.device_map.keys())) " |
| 506 | src/transformers/models/t5/modeling_t5.py:897 | change "self.last_device = "cuda:" + str(max(self.device_map.keys()))" to "self.last_device = "mlu:" + str(max(self.device_map.keys())) " |
| 507 | src/transformers/models/t5/modeling_t5.py:901 | change "cuda_device = "cuda:" + str(k)" to "mlu_device = "mlu:" + str(k) " |
| 508 | src/transformers/models/t5/modeling_t5.py:902 | change "self.block[layer] = self.block[layer].to(cuda_device)" to "self.block[layer] = self.block[layer].to(mlu_device) " |
| 509 | src/transformers/models/t5/modeling_t5.py:923 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 510 | src/transformers/models/t5/modeling_t5.py:948 | change "torch.cuda.set_device(self.first_device)" to "torch.mlu.set_device(self.first_device) " |
| 511 | src/transformers/models/t5/modeling_t5.py:1034 | change "torch.cuda.set_device(hidden_states.device)" to "torch.mlu.set_device(hidden_states.device) " |
| 512 | src/transformers/models/t5/modeling_t5.py:1113 | change "if i == v[-1] and "cuda:" + str(k) != self.last_device:" to "if i == v[-1] and "mlu:" + str(k) != self.last_device: " |
| 513 | src/transformers/models/t5/modeling_t5.py:1114 | change "hidden_states = hidden_states.to("cuda:" + str(k + 1))" to "hidden_states = hidden_states.to("mlu:" + str(k + 1)) " |
| 514 | src/transformers/models/t5/modeling_t5.py:1350 | change "get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))" to "get_device_map(len(self.encoder.block), range(torch.mlu.device_count())) " |
| 515 | src/transformers/models/t5/modeling_t5.py:1371 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 516 | src/transformers/models/t5/modeling_t5.py:1470 | change "torch.cuda.set_device(self.decoder.first_device)" to "torch.mlu.set_device(self.decoder.first_device) " |
| 517 | src/transformers/models/t5/modeling_t5.py:1558 | change "get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))" to "get_device_map(len(self.encoder.block), range(torch.mlu.device_count())) " |
| 518 | src/transformers/models/t5/modeling_t5.py:1581 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 519 | src/transformers/models/t5/modeling_t5.py:1686 | change "torch.cuda.set_device(self.decoder.first_device)" to "torch.mlu.set_device(self.decoder.first_device) " |
| 520 | src/transformers/models/t5/modeling_t5.py:1694 | change "torch.cuda.set_device(self.decoder.first_device)" to "torch.mlu.set_device(self.decoder.first_device) " |
| 521 | src/transformers/models/t5/modeling_t5.py:1723 | change "torch.cuda.set_device(self.encoder.first_device)" to "torch.mlu.set_device(self.encoder.first_device) " |
| 522 | src/transformers/models/t5/modeling_t5.py:1844 | change "get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))" to "get_device_map(len(self.encoder.block), range(torch.mlu.device_count())) " |
| 523 | src/transformers/models/t5/modeling_t5.py:1862 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 524 | src/transformers/models/t5/convert_t5x_checkpoint_to_pytorch.py:34 | add "import torch_mlu" |
| 525 | src/transformers/models/bridgetower/modeling_bridgetower.py:22 | add "import torch_mlu" |
| 526 | src/transformers/models/nat/modeling_nat.py:22 | add "import torch_mlu" |
| 527 | src/transformers/models/encoder_decoder/modeling_encoder_decoder.py:24 | add "import torch_mlu" |
| 528 | src/transformers/models/bit/convert_bit_to_pytorch.py:23 | add "import torch_mlu" |
| 529 | src/transformers/models/bit/modeling_bit.py:22 | add "import torch_mlu" |
| 530 | src/transformers/models/ernie/modeling_ernie.py:23 | add "import torch_mlu" |
| 531 | src/transformers/models/mobilenet_v1/convert_original_tf_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 532 | src/transformers/models/mobilenet_v1/modeling_mobilenet_v1.py:20 | add "import torch_mlu" |
| 533 | src/transformers/models/vit_hybrid/convert_vit_hybrid_timm_to_pytorch.py:24 | add "import torch_mlu" |
| 534 | src/transformers/models/vit_hybrid/modeling_vit_hybrid.py:22 | add "import torch_mlu" |
| 535 | src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 536 | src/transformers/models/wav2vec2/convert_wav2vec2_original_s3prl_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 537 | src/transformers/models/wav2vec2/modeling_wav2vec2.py:23 | add "import torch_mlu" |
| 538 | src/transformers/models/wav2vec2/modeling_wav2vec2.py:1718 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 539 | src/transformers/models/wav2vec2/tokenization_wav2vec2.py:47 | add "import torch_mlu" |
| 540 | src/transformers/models/fsmt/modeling_fsmt.py:34 | add "import torch_mlu" |
| 541 | src/transformers/models/fsmt/convert_fsmt_original_pytorch_checkpoint_to_pytorch.py:30 | add "import torch_mlu" |
| 542 | src/transformers/models/biogpt/convert_biogpt_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 543 | src/transformers/models/biogpt/modeling_biogpt.py:22 | add "import torch_mlu" |
| 544 | src/transformers/models/bort/convert_bort_original_gluonnlp_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 545 | src/transformers/models/mobilenet_v2/image_processing_mobilenet_v2.py:44 | add "import torch_mlu" |
| 546 | src/transformers/models/mobilenet_v2/convert_original_tf_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 547 | src/transformers/models/mobilenet_v2/modeling_mobilenet_v2.py:20 | add "import torch_mlu" |
| 548 | src/transformers/models/blenderbot/convert_blenderbot_original_pytorch_checkpoint_to_pytorch.py:19 | add "import torch_mlu" |
| 549 | src/transformers/models/blenderbot/configuration_blenderbot.py:251 | add "import torch_mlu" |
| 550 | src/transformers/models/blenderbot/modeling_blenderbot.py:25 | add "import torch_mlu" |
| 551 | src/transformers/models/marian/modeling_marian.py:24 | add "import torch_mlu" |
| 552 | src/transformers/models/marian/configuration_marian.py:253 | add "import torch_mlu" |
| 553 | src/transformers/models/marian/convert_marian_to_pytorch.py:26 | add "import torch_mlu" |
| 554 | src/transformers/models/ctrl/modeling_ctrl.py:21 | add "import torch_mlu" |
| 555 | src/transformers/models/flaubert/modeling_flaubert.py:24 | add "import torch_mlu" |
| 556 | src/transformers/models/layoutlm/configuration_layoutlm.py:201 | add "import torch_mlu" |
| 557 | src/transformers/models/layoutlm/modeling_layoutlm.py:21 | add "import torch_mlu" |
| 558 | src/transformers/models/bert/modeling_bert.py:25 | add "import torch_mlu" |
| 559 | src/transformers/models/bert/convert_bert_original_tf2_checkpoint_to_pytorch.py:32 | add "import torch_mlu" |
| 560 | src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 561 | src/transformers/models/bert/convert_bert_pytorch_checkpoint_to_original_tf.py:23 | add "import torch_mlu" |
| 562 | src/transformers/models/bert/convert_bert_token_dropping_original_tf2_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 563 | src/transformers/models/xmod/convert_xmod_original_pytorch_checkpoint_to_pytorch.py:21 | add "import torch_mlu" |
| 564 | src/transformers/models/xmod/modeling_xmod.py:20 | add "import torch_mlu" |
| 565 | src/transformers/models/roformer/modeling_roformer.py:23 | add "import torch_mlu" |
| 566 | src/transformers/models/roformer/convert_roformer_original_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 567 | src/transformers/models/swin/convert_swin_simmim_to_pytorch.py:22 | add "import torch_mlu" |
| 568 | src/transformers/models/swin/modeling_swin.py:23 | add "import torch_mlu" |
| 569 | src/transformers/models/swin/convert_swin_timm_to_pytorch.py:6 | add "import torch_mlu" |
| 570 | src/transformers/models/time_series_transformer/modeling_time_series_transformer.py:22 | add "import torch_mlu" |
| 571 | src/transformers/models/efficientnet/convert_efficientnet_to_pytorch.py:27 | add "import torch_mlu" |
| 572 | src/transformers/models/efficientnet/modeling_efficientnet.py:21 | add "import torch_mlu" |
| 573 | src/transformers/models/regnet/convert_regnet_seer_10b_to_pytorch.py:30 | add "import torch_mlu" |
| 574 | src/transformers/models/regnet/convert_regnet_to_pytorch.py:26 | add "import torch_mlu" |
| 575 | src/transformers/models/regnet/modeling_regnet.py:19 | add "import torch_mlu" |
| 576 | src/transformers/models/data2vec/modeling_data2vec_audio.py:22 | add "import torch_mlu" |
| 577 | src/transformers/models/data2vec/modeling_data2vec_audio.py:1070 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 578 | src/transformers/models/data2vec/modeling_data2vec_text.py:20 | add "import torch_mlu" |
| 579 | src/transformers/models/data2vec/modeling_data2vec_vision.py:23 | add "import torch_mlu" |
| 580 | src/transformers/models/data2vec/convert_data2vec_audio_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 581 | src/transformers/models/data2vec/convert_data2vec_vision_original_pytorch_checkpoint_to_pytorch.py:5 | add "import torch_mlu" |
| 582 | src/transformers/models/data2vec/convert_data2vec_text_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 583 | src/transformers/models/distilbert/modeling_distilbert.py:26 | add "import torch_mlu" |
| 584 | src/transformers/models/informer/modeling_informer.py:21 | add "import torch_mlu" |
| 585 | src/transformers/models/m2m_100/configuration_m2m_100.py:239 | add "import torch_mlu" |
| 586 | src/transformers/models/m2m_100/modeling_m2m_100.py:22 | add "import torch_mlu" |
| 587 | src/transformers/models/m2m_100/convert_m2m100_original_checkpoint_to_pytorch.py:17 | add "import torch_mlu" |
| 588 | src/transformers/models/audio_spectrogram_transformer/modeling_audio_spectrogram_transformer.py:20 | add "import torch_mlu" |
| 589 | src/transformers/models/audio_spectrogram_transformer/feature_extraction_audio_spectrogram_transformer.py:22 | add "import torch_mlu" |
| 590 | src/transformers/models/audio_spectrogram_transformer/convert_audio_spectrogram_transformer_original_to_pytorch.py:22 | add "import torch_mlu" |
| 591 | src/transformers/models/clip/modeling_clip.py:21 | add "import torch_mlu" |
| 592 | src/transformers/models/clip/convert_clip_original_pytorch_to_hf.py:18 | add "import torch_mlu" |
| 593 | src/transformers/models/xlm_roberta/modeling_xlm_roberta.py:21 | add "import torch_mlu" |
| 594 | src/transformers/models/perceiver/convert_perceiver_haiku_to_pytorch.py:26 | add "import torch_mlu" |
| 595 | src/transformers/models/perceiver/modeling_perceiver.py:25 | add "import torch_mlu" |
| 596 | src/transformers/models/segformer/convert_segformer_original_to_pytorch.py:24 | add "import torch_mlu" |
| 597 | src/transformers/models/segformer/image_processing_segformer.py:41 | add "import torch_mlu" |
| 598 | src/transformers/models/segformer/modeling_segformer.py:21 | add "import torch_mlu" |
| 599 | src/transformers/models/longformer/configuration_longformer.py:213 | add "import torch_mlu" |
| 600 | src/transformers/models/longformer/modeling_longformer.py:21 | add "import torch_mlu" |
| 601 | src/transformers/models/longformer/convert_longformer_original_pytorch_lightning_to_pytorch.py:21 | add "import torch_mlu" |
| 602 | src/transformers/models/rembert/convert_rembert_tf_checkpoint_to_pytorch.py:20 | add "import torch_mlu" |
| 603 | src/transformers/models/rembert/modeling_rembert.py:22 | add "import torch_mlu" |
| 604 | src/transformers/models/upernet/convert_convnext_upernet_to_pytorch.py:21 | add "import torch_mlu" |
| 605 | src/transformers/models/upernet/modeling_upernet.py:19 | add "import torch_mlu" |
| 606 | src/transformers/models/upernet/convert_swin_upernet_to_pytorch.py:24 | add "import torch_mlu" |
| 607 | src/transformers/models/prophetnet/modeling_prophetnet.py:23 | add "import torch_mlu" |
| 608 | src/transformers/models/canine/modeling_canine.py:24 | add "import torch_mlu" |
| 609 | src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py:22 | add "import torch_mlu" |
| 610 | src/transformers/models/wav2vec2_conformer/modeling_wav2vec2_conformer.py:1694 | change "with torch.backends.cudnn.flags(enabled=False):" to "with torch.backends.mlufusion.flags(enabled=False): " |
| 611 | src/transformers/models/wav2vec2_conformer/convert_wav2vec2_conformer_original_pytorch_checkpoint_to_pytorch.py:23 | add "import torch_mlu" |
| 612 | src/transformers/models/trocr/convert_trocr_unilm_to_pytorch.py:22 | add "import torch_mlu" |
| 613 | src/transformers/models/trocr/modeling_trocr.py:23 | add "import torch_mlu" |
| 614 | src/transformers/models/cvt/convert_cvt_original_pytorch_checkpoint_to_pytorch.py:24 | add "import torch_mlu" |
| 615 | src/transformers/models/cvt/modeling_cvt.py:22 | add "import torch_mlu" |
| 616 | src/transformers/models/timesformer/convert_timesformer_to_pytorch.py:22 | add "import torch_mlu" |
| 617 | src/transformers/models/timesformer/modeling_timesformer.py:21 | add "import torch_mlu" |
| 618 | src/transformers/models/blenderbot_small/modeling_blenderbot_small.py:23 | add "import torch_mlu" |
| 619 | src/transformers/models/blenderbot_small/configuration_blenderbot_small.py:253 | add "import torch_mlu" |
| 620 | src/transformers/models/poolformer/convert_poolformer_original_to_pytorch.py:23 | add "import torch_mlu" |
| 621 | src/transformers/models/poolformer/modeling_poolformer.py:21 | add "import torch_mlu" |
| 622 | src/transformers/models/groupvit/convert_groupvit_nvlab_to_hf.py:25 | add "import torch_mlu" |
| 623 | src/transformers/models/groupvit/modeling_groupvit.py:24 | add "import torch_mlu" |
| 624 | src/transformers/models/speecht5/modeling_speecht5.py:22 | add "import torch_mlu" |
| 625 | src/transformers/models/speecht5/feature_extraction_speecht5.py:20 | add "import torch_mlu" |
| 626 | src/transformers/models/speecht5/convert_speecht5_original_pytorch_checkpoint_to_pytorch.py:19 | add "import torch_mlu" |
| 627 | src/transformers/models/speecht5/convert_hifigan.py:20 | add "import torch_mlu" |
| 628 | src/transformers/models/clipseg/modeling_clipseg.py:22 | add "import torch_mlu" |
| 629 | src/transformers/models/clipseg/convert_clipseg_original_pytorch_to_hf.py:21 | add "import torch_mlu" |
| 630 | src/transformers/models/mmbt/modeling_mmbt.py:19 | add "import torch_mlu" |
| 631 | src/transformers/models/layoutlmv2/modeling_layoutlmv2.py:20 | add "import torch_mlu" |
| 632 | src/transformers/models/layoutlmv2/modeling_layoutlmv2.py:605 | change "node_size = torch.cuda.device_count()" to "node_size = torch.mlu.device_count() " |
| 633 | src/transformers/models/swinv2/modeling_swinv2.py:23 | add "import torch_mlu" |
| 634 | src/transformers/models/swinv2/convert_swinv2_timm_to_pytorch.py:23 | add "import torch_mlu" |
| 635 | src/transformers/models/pegasus/convert_pegasus_tf_to_pytorch.py:22 | add "import torch_mlu" |
| 636 | src/transformers/models/pegasus/modeling_pegasus.py:23 | add "import torch_mlu" |
| 637 | src/transformers/models/deta/modeling_deta.py:24 | add "import torch_mlu" |
| 638 | src/transformers/models/deta/convert_deta_swin_to_pytorch.py:25 | add "import torch_mlu" |
| 639 | src/transformers/models/deta/convert_deta_swin_to_pytorch.py:264 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 640 | src/transformers/models/deta/image_processing_deta.py:64 | add "import torch_mlu" |
| 641 | src/transformers/models/deta/convert_deta_resnet_to_pytorch.py:25 | add "import torch_mlu" |
| 642 | src/transformers/models/deta/convert_deta_resnet_to_pytorch.py:258 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 643 | src/transformers/models/efficientformer/convert_efficientformer_original_pytorch_checkpoint_to_pytorch.py:26 | add "import torch_mlu" |
| 644 | src/transformers/models/efficientformer/modeling_efficientformer.py:21 | add "import torch_mlu" |
| 645 | src/transformers/models/gptj/modeling_gptj.py:20 | add "import torch_mlu" |
| 646 | src/transformers/models/gptj/modeling_gptj.py:458 | change "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()" to "model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.mlu.empty_cache() " |
| 647 | src/transformers/models/gptj/modeling_gptj.py:497 | change "get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map" to "get_device_map(len(self.h), range(torch.mlu.device_count())) if device_map is None else device_map " |
| 648 | src/transformers/models/gptj/modeling_gptj.py:501 | change "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))" to "self.first_device = "cpu" if "cpu" in self.device_map.keys() else "mlu:" + str(min(self.device_map.keys())) " |
| 649 | src/transformers/models/gptj/modeling_gptj.py:502 | change "self.last_device = "cuda:" + str(max(self.device_map.keys()))" to "self.last_device = "mlu:" + str(max(self.device_map.keys())) " |
| 650 | src/transformers/models/gptj/modeling_gptj.py:507 | change "cuda_device = "cuda:" + str(k)" to "mlu_device = "mlu:" + str(k) " |
| 651 | src/transformers/models/gptj/modeling_gptj.py:508 | change "self.h[block] = self.h[block].to(cuda_device)" to "self.h[block] = self.h[block].to(mlu_device) " |
| 652 | src/transformers/models/gptj/modeling_gptj.py:526 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 653 | src/transformers/models/gptj/modeling_gptj.py:642 | change "torch.cuda.set_device(hidden_states.device)" to "torch.mlu.set_device(hidden_states.device) " |
| 654 | src/transformers/models/gptj/modeling_gptj.py:690 | change "if i == v[-1] and "cuda:" + str(k) != self.last_device:" to "if i == v[-1] and "mlu:" + str(k) != self.last_device: " |
| 655 | src/transformers/models/gptj/modeling_gptj.py:691 | change "hidden_states = hidden_states.to("cuda:" + str(k + 1))" to "hidden_states = hidden_states.to("mlu:" + str(k + 1)) " |
| 656 | src/transformers/models/gptj/modeling_gptj.py:742 | change "get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))" to "get_device_map(len(self.transformer.h), range(torch.mlu.device_count())) " |
| 657 | src/transformers/models/gptj/modeling_gptj.py:761 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 658 | src/transformers/models/gptj/modeling_gptj.py:854 | change "torch.cuda.set_device(self.transformer.first_device)" to "torch.mlu.set_device(self.transformer.first_device) " |
| 659 | src/transformers/models/gptj/configuration_gptj.py:193 | add "import torch_mlu" |
| 660 | src/transformers/models/camembert/modeling_camembert.py:21 | add "import torch_mlu" |
| 661 | src/transformers/models/yoso/modeling_yoso.py:22 | add "import torch_mlu" |
| 662 | src/transformers/models/yoso/modeling_yoso.py:53 | change "def load_cuda_kernels():" to "def load_mlu_kernels(): " |
| 663 | src/transformers/models/yoso/modeling_yoso.py:63 | change "["fast_lsh_cumulation_torch.cpp", "fast_lsh_cumulation.cu", "fast_lsh_cumulation_cuda.cu"]" to "["fast_lsh_cumulation_torch.cpp", "fast_lsh_cumulation.cu", "fast_lsh_cumulation_mlu.cu"] " |
| 664 | src/transformers/models/yoso/modeling_yoso.py:168 | change "use_cuda = query_mask.is_cuda" to "use_mlu = query_mask.is_mlu " |
| 665 | src/transformers/models/yoso/modeling_yoso.py:175 | change "query_mask, query, key_mask, key, num_hash, hash_code_len, use_cuda, 1" to "query_mask, query, key_mask, key, num_hash, hash_code_len, use_mlu, 1 " |
| 666 | src/transformers/models/yoso/modeling_yoso.py:181 | change "query_mask, query_hash_code, key_mask, key_hash_code, value, hashtable_capacity, use_cuda, 1" to "query_mask, query_hash_code, key_mask, key_hash_code, value, hashtable_capacity, use_mlu, 1 " |
| 667 | src/transformers/models/yoso/modeling_yoso.py:196 | change "use_cuda = grad.is_cuda" to "use_mlu = grad.is_mlu " |
| 668 | src/transformers/models/yoso/modeling_yoso.py:202 | change "key_mask, key_hash_code, query_mask, query_hash_code, grad, hashtable_capacity, use_cuda, 1" to "key_mask, key_hash_code, query_mask, query_hash_code, grad, hashtable_capacity, use_mlu, 1 " |
| 669 | src/transformers/models/yoso/modeling_yoso.py:213 | change "use_cuda," to "use_mlu, " |
| 670 | src/transformers/models/yoso/modeling_yoso.py:225 | change "use_cuda," to "use_mlu, " |
| 671 | src/transformers/models/yoso/convert_yoso_pytorch_to_pytorch.py:19 | add "import torch_mlu" |
| 672 | src/transformers/models/yoso/configuration_yoso.py:79 | change "Whether or not to use custom cuda kernels which perform fast random projection via hadamard transform." to "Whether or not to use custom mlu kernels which perform fast random projection via hadamard transform. " |
| 673 | src/transformers/pipelines/__init__.py:110 | add "import torch_mlu" |
| 674 | src/transformers/pipelines/__init__.py:603 | change "Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this" to "Defines the device (*e.g.*, `"cpu"`, `"mlu:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this " |
| 675 | src/transformers/pipelines/depth_estimation.py:15 | add "import torch_mlu" |
| 676 | src/transformers/pipelines/pt_utils.py:2 | add "import torch_mlu" |
| 677 | src/transformers/pipelines/table_question_answering.py:17 | add "import torch_mlu" |
| 678 | src/transformers/pipelines/document_question_answering.py:38 | add "import torch_mlu" |
| 679 | src/transformers/pipelines/fill_mask.py:16 | add "import torch_mlu" |
| 680 | src/transformers/pipelines/base.py:50 | add "import torch_mlu" |
| 681 | src/transformers/pipelines/base.py:792 | change "self.device = torch.device(f"cuda:{device}")" to "self.device = torch.device(f"mlu:{device}") " |
| 682 | src/transformers/pipelines/base.py:893 | change "if self.device.type == "cuda":" to "if self.device.type == "mlu": " |
| 683 | src/transformers/pipelines/base.py:894 | change "torch.cuda.set_device(self.device)" to "torch.mlu.set_device(self.device) " |
| 684 | src/transformers/pipelines/base.py:1069 | change "if self.call_count > 10 and self.framework == "pt" and self.device.type == "cuda":" to "if self.call_count > 10 and self.framework == "pt" and self.device.type == "mlu": " |
| 685 | src/transformers/pipelines/object_detection.py:12 | add "import torch_mlu" |
| 686 | src/transformers/pipelines/automatic_speech_recognition.py:351 | add "import torch_mlu" |
| 687 | src/transformers/pipelines/question_answering.py:39 | add "import torch_mlu" |
| 688 | src/transformers/pipelines/zero_shot_object_detection.py:13 | add "import torch_mlu" |
| 689 | src/transformers/pipelines/conversational.py:12 | add "import torch_mlu" |
| 690 | src/transformers/data/data_collator.py:106 | add "import torch_mlu" |
| 691 | src/transformers/data/test_generation_utils.py:25 | add "import torch_mlu" |
| 692 | src/transformers/data/processors/utils.py:336 | add "import torch_mlu" |
| 693 | src/transformers/data/processors/squad.py:34 | add "import torch_mlu" |
| 694 | src/transformers/data/datasets/language_modeling.py:23 | add "import torch_mlu" |
| 695 | src/transformers/data/datasets/glue.py:22 | add "import torch_mlu" |
| 696 | src/transformers/data/datasets/squad.py:21 | add "import torch_mlu" |
| 697 | src/transformers/utils/import_utils.py:302 | change "def is_torch_cuda_available():" to "def is_torch_mlu_available(): " |
| 698 | src/transformers/utils/import_utils.py:304 | add "import torch_mlu" |
| 699 | src/transformers/utils/import_utils.py:306 | change "return torch.cuda.is_available()" to "return torch.mlu.is_available() " |
| 700 | src/transformers/utils/import_utils.py:318 | change "# some bits come from https://github.com/pytorch/pytorch/blob/2289a12f21c54da93bf5d696e3f9aea83dd9c10d/torch/testing/_internal/common_cuda.py#L51" to "# some bits come from https://github.com/pytorch/pytorch/blob/2289a12f21c54da93bf5d696e3f9aea83dd9c10d/torch/testing/_internal/common_mlu.py#L51 " |
| 701 | src/transformers/utils/import_utils.py:330 | change "if torch.cuda.is_available() and torch.version.cuda is not None:" to "if torch.mlu.is_available() and torch.version.mlu is not None: " |
| 702 | src/transformers/utils/import_utils.py:331 | change "if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:" to "if torch.mlu.get_device_properties(torch.mlu.current_device()).major < 8: " |
| 703 | src/transformers/utils/import_utils.py:333 | change "if int(torch.version.cuda.split(".")[0]) < 11:" to "if int(torch.version.mlu.split(".")[0]) < 11: " |
| 704 | src/transformers/utils/import_utils.py:335 | change "if not hasattr(torch.cuda.amp, "autocast"):" to "if not hasattr(torch.mlu.amp, "autocast"): " |
| 705 | src/transformers/utils/import_utils.py:378 | change "if not torch.cuda.is_available() or torch.version.cuda is None:" to "if not torch.mlu.is_available() or torch.version.mlu is None: " |
| 706 | src/transformers/utils/import_utils.py:380 | change "if torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8:" to "if torch.mlu.get_device_properties(torch.mlu.current_device()).major < 8: " |
| 707 | src/transformers/utils/import_utils.py:382 | change "if int(torch.version.cuda.split(".")[0]) < 11:" to "if int(torch.version.mlu.split(".")[0]) < 11: " |
| 708 | src/transformers/utils/generic.py:66 | add "import torch_mlu" |
| 709 | src/transformers/utils/__init__.py:150 | change "is_torch_cuda_available," to "is_torch_mlu_available, " |
| 710 | src/transformers/utils/bitsandbytes.py:8 | add "import torch_mlu" |
| 711 | src/transformers/utils/bitsandbytes.py:58 | change "if param.device.type != "cuda":" to "if param.device.type != "mlu": " |
| 712 | src/transformers/utils/fx.py:27 | add "import torch_mlu" |
| 713 | examples/legacy/run_openai_gpt.py:38 | add "import torch_mlu" |
| 714 | examples/legacy/run_openai_gpt.py:162 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 715 | examples/legacy/run_openai_gpt.py:164 | change "device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 716 | examples/legacy/run_openai_gpt.py:165 | change "n_gpu = torch.cuda.device_count()" to "n_gpu = torch.mlu.device_count() " |
| 717 | examples/legacy/run_camembert.py:2 | add "import torch_mlu" |
| 718 | examples/legacy/run_swag.py:30 | add "import torch_mlu" |
| 719 | examples/legacy/run_swag.py:230 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 720 | examples/legacy/run_swag.py:562 | change "parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Whether not to use CUDA when available") " |
| 721 | examples/legacy/run_swag.py:612 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 722 | examples/legacy/run_swag.py:613 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 723 | examples/legacy/run_swag.py:614 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 724 | examples/legacy/run_swag.py:616 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 725 | examples/legacy/run_swag.py:617 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 726 | examples/legacy/run_swag.py:618 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 727 | examples/legacy/run_transfo_xl.py:30 | add "import torch_mlu" |
| 728 | examples/legacy/run_transfo_xl.py:52 | change "parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even though CUA is available")" to "parser.add_argument("--no_mlu", action="store_true", help="Do not use CUDA even though CUA is available") " |
| 729 | examples/legacy/run_transfo_xl.py:69 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 730 | examples/legacy/seq2seq/run_eval.py:25 | add "import torch_mlu" |
| 731 | examples/legacy/seq2seq/run_eval.py:35 | change "DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"" to "DEFAULT_DEVICE = "mlu" if torch.mlu.is_available() else "cpu" " |
| 732 | examples/legacy/seq2seq/run_eval.py:108 | change "parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")" to "parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="mlu, mlu:1, cpu etc.") " |
| 733 | examples/legacy/seq2seq/utils.py:28 | add "import torch_mlu" |
| 734 | examples/legacy/seq2seq/old_test_fsmt_bleu_score.py:39 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 735 | examples/legacy/seq2seq/run_distributed_eval.py:24 | add "import torch_mlu" |
| 736 | examples/legacy/seq2seq/run_distributed_eval.py:65 | change "torch.distributed.init_process_group(backend="nccl", rank=local_rank)" to "torch.distributed.init_process_group(backend="cncl", rank=local_rank) " |
| 737 | examples/legacy/seq2seq/run_distributed_eval.py:69 | change "torch.cuda.set_device(local_rank)" to "torch.mlu.set_device(local_rank) " |
| 738 | examples/legacy/seq2seq/run_distributed_eval.py:70 | change "model = AutoModelForSeq2SeqLM.from_pretrained(model_name).cuda()" to "model = AutoModelForSeq2SeqLM.from_pretrained(model_name).mlu() " |
| 739 | examples/legacy/seq2seq/convert_model_to_fp16.py:19 | add "import torch_mlu" |
| 740 | examples/legacy/seq2seq/seq2seq_trainer.py:17 | add "import torch_mlu" |
| 741 | examples/legacy/token-classification/utils_ner.py:208 | add "import torch_mlu" |
| 742 | examples/legacy/pytorch-lightning/run_ner.py:9 | add "import torch_mlu" |
| 743 | examples/legacy/pytorch-lightning/run_glue.py:9 | add "import torch_mlu" |
| 744 | examples/legacy/question-answering/run_squad.py:27 | add "import torch_mlu" |
| 745 | examples/legacy/question-answering/run_squad.py:69 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 746 | examples/legacy/question-answering/run_squad.py:646 | change "parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Whether not to use CUDA when available") " |
| 747 | examples/legacy/question-answering/run_squad.py:705 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 748 | examples/legacy/question-answering/run_squad.py:706 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 749 | examples/legacy/question-answering/run_squad.py:707 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 750 | examples/legacy/question-answering/run_squad.py:709 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 751 | examples/legacy/question-answering/run_squad.py:710 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 752 | examples/legacy/question-answering/run_squad.py:711 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 753 | examples/legacy/multiple_choice/utils_multiple_choice.py:79 | add "import torch_mlu" |
| 754 | examples/research_projects/decision_transformer/run_decision_transformer.py:3 | add "import torch_mlu" |
| 755 | examples/research_projects/decision_transformer/run_decision_transformer.py:92 | change "device = "cuda"" to "device = "mlu" " |
| 756 | examples/research_projects/codeparrot/scripts/codeparrot_training.py:8 | add "import torch_mlu" |
| 757 | examples/research_projects/codeparrot/scripts/validation_loss.py:3 | add "import torch_mlu" |
| 758 | examples/research_projects/codeparrot/scripts/human_eval.py:7 | add "import torch_mlu" |
| 759 | examples/research_projects/rag/callbacks_rag.py:6 | add "import torch_mlu" |
| 760 | examples/research_projects/rag/use_own_knowledge_dataset.py:10 | add "import torch_mlu" |
| 761 | examples/research_projects/rag/use_own_knowledge_dataset.py:25 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 762 | examples/research_projects/rag/finetune_rag.py:14 | add "import torch_mlu" |
| 763 | examples/research_projects/rag/utils_rag.py:15 | add "import torch_mlu" |
| 764 | examples/research_projects/rag/eval_rag.py:10 | add "import torch_mlu" |
| 765 | examples/research_projects/rag/eval_rag.py:253 | change "args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 766 | examples/research_projects/rag/distributed_pytorch_retriever.py:7 | add "import torch_mlu" |
| 767 | examples/research_projects/rag/distributed_pytorch_retriever.py:58 | change "# nccl backend doesn't support gather/scatter operations while gloo" to "# cncl backend doesn't support gather/scatter operations while gloo " |
| 768 | examples/research_projects/rag/distributed_pytorch_retriever.py:59 | change "# is too slow to replace nccl for the core gpu communication" to "# is too slow to replace cncl for the core gpu communication " |
| 769 | examples/research_projects/movement-pruning/masked_run_glue.py:26 | add "import torch_mlu" |
| 770 | examples/research_projects/movement-pruning/masked_run_glue.py:66 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 771 | examples/research_projects/movement-pruning/masked_run_glue.py:779 | change "parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Avoid using CUDA when available") " |
| 772 | examples/research_projects/movement-pruning/masked_run_glue.py:826 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 773 | examples/research_projects/movement-pruning/masked_run_glue.py:827 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 774 | examples/research_projects/movement-pruning/masked_run_glue.py:828 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 775 | examples/research_projects/movement-pruning/masked_run_glue.py:830 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 776 | examples/research_projects/movement-pruning/masked_run_glue.py:831 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 777 | examples/research_projects/movement-pruning/masked_run_glue.py:832 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 778 | examples/research_projects/movement-pruning/masked_run_squad.py:27 | add "import torch_mlu" |
| 779 | examples/research_projects/movement-pruning/masked_run_squad.py:70 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 780 | examples/research_projects/movement-pruning/masked_run_squad.py:929 | change "parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Whether not to use CUDA when available") " |
| 781 | examples/research_projects/movement-pruning/masked_run_squad.py:992 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 782 | examples/research_projects/movement-pruning/masked_run_squad.py:993 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 783 | examples/research_projects/movement-pruning/masked_run_squad.py:994 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 784 | examples/research_projects/movement-pruning/masked_run_squad.py:996 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 785 | examples/research_projects/movement-pruning/masked_run_squad.py:997 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 786 | examples/research_projects/movement-pruning/masked_run_squad.py:998 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 787 | examples/research_projects/movement-pruning/counts_parameters.py:21 | add "import torch_mlu" |
| 788 | examples/research_projects/movement-pruning/bertarize.py:24 | add "import torch_mlu" |
| 789 | examples/research_projects/movement-pruning/emmental/modeling_bert_masked.py:25 | add "import torch_mlu" |
| 790 | examples/research_projects/movement-pruning/emmental/modules/binarizer.py:20 | add "import torch_mlu" |
| 791 | examples/research_projects/movement-pruning/emmental/modules/masked_nn.py:24 | add "import torch_mlu" |
| 792 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:27 | add "import torch_mlu" |
| 793 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:71 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 794 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:555 | change "parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Avoid using CUDA when available") " |
| 795 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:614 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 796 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:615 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 797 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:616 | change "args.n_gpu = torch.cuda.device_count()" to "args.n_gpu = torch.mlu.device_count() " |
| 798 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:618 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 799 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:619 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 800 | examples/research_projects/bert-loses-patience/run_glue_with_pabee.py:620 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 801 | examples/research_projects/bert-loses-patience/pabee/modeling_pabee_albert.py:19 | add "import torch_mlu" |
| 802 | examples/research_projects/bert-loses-patience/pabee/modeling_pabee_bert.py:21 | add "import torch_mlu" |
| 803 | examples/research_projects/bertology/run_prune_gpt.py:12 | add "import torch_mlu" |
| 804 | examples/research_projects/bertology/run_prune_gpt.py:326 | change "parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Whether not to use CUDA when available") " |
| 805 | examples/research_projects/bertology/run_prune_gpt.py:340 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 806 | examples/research_projects/bertology/run_prune_gpt.py:341 | change "args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 807 | examples/research_projects/bertology/run_prune_gpt.py:342 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 808 | examples/research_projects/bertology/run_prune_gpt.py:344 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 809 | examples/research_projects/bertology/run_prune_gpt.py:345 | change "args.device = torch.device("cuda", args.local_rank)" to "args.device = torch.device("mlu", args.local_rank) " |
| 810 | examples/research_projects/bertology/run_prune_gpt.py:347 | change "torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend" to "torch.distributed.init_process_group(backend="cncl")  # Initializes the distributed backend " |
| 811 | examples/research_projects/bertology/run_bertology.py:28 | add "import torch_mlu" |
| 812 | examples/research_projects/bertology/run_bertology.py:350 | change "parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Whether not to use CUDA when available") " |
| 813 | examples/research_projects/bertology/run_bertology.py:364 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 814 | examples/research_projects/bertology/run_bertology.py:365 | change "args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 815 | examples/research_projects/bertology/run_bertology.py:366 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 816 | examples/research_projects/bertology/run_bertology.py:368 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 817 | examples/research_projects/bertology/run_bertology.py:369 | change "args.device = torch.device("cuda", args.local_rank)" to "args.device = torch.device("mlu", args.local_rank) " |
| 818 | examples/research_projects/bertology/run_bertology.py:371 | change "torch.distributed.init_process_group(backend="nccl")  # Initializes the distributed backend" to "torch.distributed.init_process_group(backend="cncl")  # Initializes the distributed backend " |
| 819 | examples/research_projects/robust-speech-event/run_speech_recognition_ctc_bnb.py:31 | add "import torch_mlu" |
| 820 | examples/research_projects/robust-speech-event/eval.py:6 | add "import torch_mlu" |
| 821 | examples/research_projects/robust-speech-event/eval.py:82 | change "args.device = 0 if torch.cuda.is_available() else -1" to "args.device = 0 if torch.mlu.is_available() else -1 " |
| 822 | examples/research_projects/robust-speech-event/run_speech_recognition_ctc_streaming.py:28 | add "import torch_mlu" |
| 823 | examples/research_projects/self-training-text-classification/finetuning.py:30 | add "import torch_mlu" |
| 824 | examples/research_projects/fsner/src/fsner/tokenizer_utils.py:1 | add "import torch_mlu" |
| 825 | examples/research_projects/fsner/src/fsner/model.py:1 | add "import torch_mlu" |
| 826 | examples/research_projects/pplm/run_pplm_discrim_train.py:25 | add "import torch_mlu" |
| 827 | examples/research_projects/pplm/run_pplm_discrim_train.py:227 | change "no_cuda=False," to "no_mlu=False, " |
| 828 | examples/research_projects/pplm/run_pplm_discrim_train.py:229 | change "device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"" to "device = "mlu" if torch.mlu.is_available() and not no_mlu else "cpu" " |
| 829 | examples/research_projects/pplm/run_pplm_discrim_train.py:520 | change "parser.add_argument("--no_cuda", action="store_true", help="use to turn off cuda")" to "parser.add_argument("--no_mlu", action="store_true", help="use to turn off mlu") " |
| 830 | examples/research_projects/pplm/run_pplm.py:32 | add "import torch_mlu" |
| 831 | examples/research_projects/pplm/run_pplm.py:112 | change "device="cuda"," to "device="mlu", " |
| 832 | examples/research_projects/pplm/run_pplm.py:308 | change "def build_bows_one_hot_vectors(bow_indices, tokenizer, device="cuda"):" to "def build_bows_one_hot_vectors(bow_indices, tokenizer, device="mlu"): " |
| 833 | examples/research_projects/pplm/run_pplm.py:328 | change "device="cuda"," to "device="mlu", " |
| 834 | examples/research_projects/pplm/run_pplm.py:379 | change "if device == "cuda":" to "if device == "mlu": " |
| 835 | examples/research_projects/pplm/run_pplm.py:380 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 836 | examples/research_projects/pplm/run_pplm.py:417 | change "if device == "cuda":" to "if device == "mlu": " |
| 837 | examples/research_projects/pplm/run_pplm.py:418 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 838 | examples/research_projects/pplm/run_pplm.py:428 | change "device="cuda"," to "device="mlu", " |
| 839 | examples/research_projects/pplm/run_pplm.py:611 | change "no_cuda=False," to "no_mlu=False, " |
| 840 | examples/research_projects/pplm/run_pplm.py:620 | change "device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"" to "device = "mlu" if torch.mlu.is_available() and not no_mlu else "cpu" " |
| 841 | examples/research_projects/pplm/run_pplm.py:813 | change "parser.add_argument("--no_cuda", action="store_true", help="no cuda")" to "parser.add_argument("--no_mlu", action="store_true", help="no mlu") " |
| 842 | examples/research_projects/distillation/lm_seqs_dataset.py:19 | add "import torch_mlu" |
| 843 | examples/research_projects/distillation/train.py:26 | add "import torch_mlu" |
| 844 | examples/research_projects/distillation/train.py:292 | change "student.to(f"cuda:{args.local_rank}")" to "student.to(f"mlu:{args.local_rank}") " |
| 845 | examples/research_projects/distillation/train.py:298 | change "teacher.to(f"cuda:{args.local_rank}")" to "teacher.to(f"mlu:{args.local_rank}") " |
| 846 | examples/research_projects/distillation/train.py:315 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 847 | examples/research_projects/distillation/run_squad_w_distillation.py:26 | add "import torch_mlu" |
| 848 | examples/research_projects/distillation/run_squad_w_distillation.py:86 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 849 | examples/research_projects/distillation/run_squad_w_distillation.py:679 | change "parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Whether not to use CUDA when available") " |
| 850 | examples/research_projects/distillation/run_squad_w_distillation.py:731 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 851 | examples/research_projects/distillation/run_squad_w_distillation.py:732 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 852 | examples/research_projects/distillation/run_squad_w_distillation.py:733 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 853 | examples/research_projects/distillation/run_squad_w_distillation.py:735 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 854 | examples/research_projects/distillation/run_squad_w_distillation.py:736 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 855 | examples/research_projects/distillation/run_squad_w_distillation.py:737 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 856 | examples/research_projects/distillation/distiller.py:23 | add "import torch_mlu" |
| 857 | examples/research_projects/distillation/distiller.py:87 | change "self.pred_probs = self.pred_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs" to "self.pred_probs = self.pred_probs.to(f"mlu:{params.local_rank}") if params.n_gpu > 0 else self.pred_probs " |
| 858 | examples/research_projects/distillation/distiller.py:88 | change "self.token_probs = token_probs.to(f"cuda:{params.local_rank}") if params.n_gpu > 0 else token_probs" to "self.token_probs = token_probs.to(f"mlu:{params.local_rank}") if params.n_gpu > 0 else token_probs " |
| 859 | examples/research_projects/distillation/distiller.py:348 | change "batch = tuple(t.to(f"cuda:{self.params.local_rank}") for t in batch)" to "batch = tuple(t.to(f"mlu:{self.params.local_rank}") for t in batch) " |
| 860 | examples/research_projects/distillation/utils.py:25 | add "import torch_mlu" |
| 861 | examples/research_projects/distillation/utils.py:62 | change "assert torch.cuda.is_available()" to "assert torch.mlu.is_available() " |
| 862 | examples/research_projects/distillation/utils.py:115 | change "torch.cuda.set_device(params.local_rank)" to "torch.mlu.set_device(params.local_rank) " |
| 863 | examples/research_projects/distillation/utils.py:122 | change "backend="nccl"," to "backend="cncl", " |
| 864 | examples/research_projects/distillation/utils.py:133 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 865 | examples/research_projects/distillation/scripts/extract_distilbert.py:21 | add "import torch_mlu" |
| 866 | examples/research_projects/distillation/scripts/extract.py:21 | add "import torch_mlu" |
| 867 | examples/research_projects/adversarial/utils_hans.py:90 | add "import torch_mlu" |
| 868 | examples/research_projects/adversarial/run_hans.py:24 | add "import torch_mlu" |
| 869 | examples/research_projects/longform-qa/eli5_app.py:5 | add "import torch_mlu" |
| 870 | examples/research_projects/longform-qa/eli5_app.py:27 | change "qar_model = AutoModel.from_pretrained("yjernite/retribert-base-uncased").to("cuda:0")" to "qar_model = AutoModel.from_pretrained("yjernite/retribert-base-uncased").to("mlu:0") " |
| 871 | examples/research_projects/longform-qa/eli5_app.py:33 | change "s2s_model = AutoModelForSeq2SeqLM.from_pretrained("yjernite/bart_eli5").to("cuda:0")" to "s2s_model = AutoModelForSeq2SeqLM.from_pretrained("yjernite/bart_eli5").to("mlu:0") " |
| 872 | examples/research_projects/longform-qa/eli5_app.py:39 | change "model_name="t5-small", from_file="seq2seq_models/eli5_t5_model_1024_4.pth", device="cuda:0"" to "model_name="t5-small", from_file="seq2seq_models/eli5_t5_model_1024_4.pth", device="mlu:0" " |
| 873 | examples/research_projects/longform-qa/eli5_app.py:133 | change "device="cuda:0"," to "device="mlu:0", " |
| 874 | examples/research_projects/longform-qa/eli5_utils.py:11 | add "import torch_mlu" |
| 875 | examples/research_projects/longform-qa/eli5_utils.py:186 | change "def make_qa_retriever_model(model_name="google/bert_uncased_L-8_H-512_A-8", from_file=None, device="cuda:0"):" to "def make_qa_retriever_model(model_name="google/bert_uncased_L-8_H-512_A-8", from_file=None, device="mlu:0"): " |
| 876 | examples/research_projects/longform-qa/eli5_utils.py:202 | change "def make_qa_retriever_batch(qa_list, tokenizer, max_len=64, device="cuda:0"):" to "def make_qa_retriever_batch(qa_list, tokenizer, max_len=64, device="mlu:0"): " |
| 877 | examples/research_projects/longform-qa/eli5_utils.py:223 | change "make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"" to "make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device="mlu:0" " |
| 878 | examples/research_projects/longform-qa/eli5_utils.py:260 | change "make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"" to "make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device="mlu:0" " |
| 879 | examples/research_projects/longform-qa/eli5_utils.py:305 | change "make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"" to "make_qa_retriever_batch, tokenizer=tokenizer, max_len=args.max_length, device="mlu:0" " |
| 880 | examples/research_projects/longform-qa/eli5_utils.py:384 | change "def make_qa_s2s_model(model_name="facebook/bart-large", from_file=None, device="cuda:0"):" to "def make_qa_s2s_model(model_name="facebook/bart-large", from_file=None, device="mlu:0"): " |
| 881 | examples/research_projects/longform-qa/eli5_utils.py:393 | change "def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device="cuda:0"):" to "def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device="mlu:0"): " |
| 882 | examples/research_projects/longform-qa/eli5_utils.py:425 | change "make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"" to "make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="mlu:0" " |
| 883 | examples/research_projects/longform-qa/eli5_utils.py:464 | change "make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="cuda:0"" to "make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device="mlu:0" " |
| 884 | examples/research_projects/longform-qa/eli5_utils.py:537 | change "device="cuda:0"," to "device="mlu:0", " |
| 885 | examples/research_projects/longform-qa/eli5_utils.py:568 | change "def embed_passages_for_retrieval(passages, tokenizer, qa_embedder, max_length=128, device="cuda:0"):" to "def embed_passages_for_retrieval(passages, tokenizer, qa_embedder, max_length=128, device="mlu:0"): " |
| 886 | examples/research_projects/longform-qa/eli5_utils.py:579 | change "def embed_questions_for_retrieval(q_ls, tokenizer, qa_embedder, device="cuda:0"):" to "def embed_questions_for_retrieval(q_ls, tokenizer, qa_embedder, device="mlu:0"): " |
| 887 | examples/research_projects/longform-qa/eli5_utils.py:598 | change "device="cuda:0"," to "device="mlu:0", " |
| 888 | examples/research_projects/longform-qa/eli5_utils.py:631 | change "question, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10, min_length=20, device="cuda:0"" to "question, qa_embedder, tokenizer, wiki_passages, wiki_index, n_results=10, min_length=20, device="mlu:0" " |
| 889 | examples/research_projects/deebert/run_glue_deebert.py:11 | add "import torch_mlu" |
| 890 | examples/research_projects/deebert/run_glue_deebert.py:56 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 891 | examples/research_projects/deebert/run_glue_deebert.py:516 | change "parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Avoid using CUDA when available") " |
| 892 | examples/research_projects/deebert/run_glue_deebert.py:566 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 893 | examples/research_projects/deebert/run_glue_deebert.py:567 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 894 | examples/research_projects/deebert/run_glue_deebert.py:568 | change "args.n_gpu = torch.cuda.device_count()" to "args.n_gpu = torch.mlu.device_count() " |
| 895 | examples/research_projects/deebert/run_glue_deebert.py:570 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 896 | examples/research_projects/deebert/run_glue_deebert.py:571 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 897 | examples/research_projects/deebert/run_glue_deebert.py:572 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 898 | examples/research_projects/deebert/src/modeling_highway_bert.py:1 | add "import torch_mlu" |
| 899 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:24 | change "import pycuda.autoinit  # noqa: F401" to "import pymlu.autoinit  # noqa: F401 " |
| 900 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:25 | change "import pycuda.driver as cuda" to "import pymlu.driver as mlu " |
| 901 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:27 | add "import torch_mlu" |
| 902 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:218 | change "cuda.memcpy_htod_async(d_inputs[0], input_ids.ravel(), stream)" to "mlu.memcpy_htod_async(d_inputs[0], input_ids.ravel(), stream) " |
| 903 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:219 | change "cuda.memcpy_htod_async(d_inputs[1], attention_mask.ravel(), stream)" to "mlu.memcpy_htod_async(d_inputs[1], attention_mask.ravel(), stream) " |
| 904 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:220 | change "cuda.memcpy_htod_async(d_inputs[2], token_type_ids.ravel(), stream)" to "mlu.memcpy_htod_async(d_inputs[2], token_type_ids.ravel(), stream) " |
| 905 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:228 | change "cuda.memcpy_dtoh_async(h_output0, d_output0, stream)" to "mlu.memcpy_dtoh_async(h_output0, d_output0, stream) " |
| 906 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:229 | change "cuda.memcpy_dtoh_async(h_output1, d_output1, stream)" to "mlu.memcpy_dtoh_async(h_output1, d_output1, stream) " |
| 907 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:395 | change "with open(engine_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(" to "with open(engine_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_mlu_engine( " |
| 908 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:407 | change "d_inputs = [cuda.mem_alloc(binding_nbytes(binding)) for binding in engine if engine.binding_is_input(binding)]" to "d_inputs = [mlu.mem_alloc(binding_nbytes(binding)) for binding in engine if engine.binding_is_input(binding)] " |
| 909 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:410 | change "h_output0 = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)" to "h_output0 = mlu.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32) " |
| 910 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:411 | change "h_output1 = cuda.pagelocked_empty(tuple(context.get_binding_shape(4)), dtype=np.float32)" to "h_output1 = mlu.pagelocked_empty(tuple(context.get_binding_shape(4)), dtype=np.float32) " |
| 911 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:412 | change "d_output0 = cuda.mem_alloc(h_output0.nbytes)" to "d_output0 = mlu.mem_alloc(h_output0.nbytes) " |
| 912 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:413 | change "d_output1 = cuda.mem_alloc(h_output1.nbytes)" to "d_output1 = mlu.mem_alloc(h_output1.nbytes) " |
| 913 | examples/research_projects/quantization-qdqbert/evaluate-hf-trt-qa.py:416 | change "stream = cuda.Stream()" to "stream = mlu.Stream() " |
| 914 | examples/research_projects/quantization-qdqbert/trainer_quant_qa.py:24 | add "import torch_mlu" |
| 915 | examples/research_projects/quantization-qdqbert/trainer_quant_qa.py:171 | change "device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 916 | examples/research_projects/quantization-qdqbert/quant_trainer.py:21 | add "import torch_mlu" |
| 917 | examples/research_projects/quantization-qdqbert/quant_trainer.py:144 | change "model.cuda()" to "model.mlu() " |
| 918 | examples/research_projects/bertabs/modeling_bertabs.py:26 | add "import torch_mlu" |
| 919 | examples/research_projects/bertabs/modeling_bertabs.py:312 | change "# it gets TransformerDecoderLayer's cuda behavior automatically." to "# it gets TransformerDecoderLayer's mlu behavior automatically. " |
| 920 | examples/research_projects/bertabs/convert_bertabs_original_pytorch_checkpoint.py:26 | add "import torch_mlu" |
| 921 | examples/research_projects/bertabs/run_summarization.py:8 | add "import torch_mlu" |
| 922 | examples/research_projects/bertabs/run_summarization.py:274 | change ""--no_cuda"," to ""--no_mlu", " |
| 923 | examples/research_projects/bertabs/run_summarization.py:319 | change "args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 924 | examples/research_projects/bertabs/utils_summarization.py:4 | add "import torch_mlu" |
| 925 | examples/research_projects/bertabs/test_utils_summarization.py:18 | add "import torch_mlu" |
| 926 | examples/research_projects/mm-imdb/run_mmimdb.py:27 | add "import torch_mlu" |
| 927 | examples/research_projects/mm-imdb/run_mmimdb.py:63 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 928 | examples/research_projects/mm-imdb/run_mmimdb.py:409 | change "parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Avoid using CUDA when available") " |
| 929 | examples/research_projects/mm-imdb/run_mmimdb.py:460 | change "if args.local_rank == -1 or args.no_cuda:" to "if args.local_rank == -1 or args.no_mlu: " |
| 930 | examples/research_projects/mm-imdb/run_mmimdb.py:461 | change "device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 931 | examples/research_projects/mm-imdb/run_mmimdb.py:462 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 932 | examples/research_projects/mm-imdb/run_mmimdb.py:464 | change "torch.cuda.set_device(args.local_rank)" to "torch.mlu.set_device(args.local_rank) " |
| 933 | examples/research_projects/mm-imdb/run_mmimdb.py:465 | change "device = torch.device("cuda", args.local_rank)" to "device = torch.device("mlu", args.local_rank) " |
| 934 | examples/research_projects/mm-imdb/run_mmimdb.py:466 | change "torch.distributed.init_process_group(backend="nccl")" to "torch.distributed.init_process_group(backend="cncl") " |
| 935 | examples/research_projects/mm-imdb/utils_mmimdb.py:21 | add "import torch_mlu" |
| 936 | examples/research_projects/onnx/summarization/run_onnx_exporter.py:26 | add "import torch_mlu" |
| 937 | examples/research_projects/onnx/summarization/bart_onnx/generation_onnx.py:5 | add "import torch_mlu" |
| 938 | examples/research_projects/luke/run_luke_ner_no_trainer.py:29 | add "import torch_mlu" |
| 939 | examples/research_projects/luke/luke_utils.py:82 | add "import torch_mlu" |
| 940 | examples/research_projects/rag-end2end-retriever/callbacks_rag.py:6 | add "import torch_mlu" |
| 941 | examples/research_projects/rag-end2end-retriever/use_own_knowledge_dataset.py:10 | add "import torch_mlu" |
| 942 | examples/research_projects/rag-end2end-retriever/use_own_knowledge_dataset.py:18 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 943 | examples/research_projects/rag-end2end-retriever/finetune_rag.py:19 | add "import torch_mlu" |
| 944 | examples/research_projects/rag-end2end-retriever/finetune_rag.py:274 | change "free_gpu_list.append("cuda:" + str(position))" to "free_gpu_list.append("mlu:" + str(position)) " |
| 945 | examples/research_projects/rag-end2end-retriever/finetune_rag.py:291 | change "cuda_devices = random.sample(free_gpu_list, self.custom_config.index_gpus)" to "mlu_devices = random.sample(free_gpu_list, self.custom_config.index_gpus) " |
| 946 | examples/research_projects/rag-end2end-retriever/finetune_rag.py:293 | change "cuda_devices = free_gpu_list" to "mlu_devices = free_gpu_list " |
| 947 | examples/research_projects/rag-end2end-retriever/finetune_rag.py:295 | change "num_processes = len(cuda_devices)" to "num_processes = len(mlu_devices) " |
| 948 | examples/research_projects/rag-end2end-retriever/finetune_rag.py:299 | change "device = cuda_devices[rank]" to "device = mlu_devices[rank] " |
| 949 | examples/research_projects/rag-end2end-retriever/utils_rag.py:15 | add "import torch_mlu" |
| 950 | examples/research_projects/rag-end2end-retriever/eval_rag.py:10 | add "import torch_mlu" |
| 951 | examples/research_projects/rag-end2end-retriever/eval_rag.py:253 | change "args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() else "cpu") " |
| 952 | examples/research_projects/lxmert/extracting_data.py:11 | add "import torch_mlu" |
| 953 | examples/research_projects/lxmert/extracting_data.py:69 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 954 | examples/research_projects/lxmert/extracting_data.py:70 | change "self.config.model.device = "cuda"" to "self.config.model.device = "mlu" " |
| 955 | examples/research_projects/lxmert/visualizing_image.py:26 | add "import torch_mlu" |
| 956 | examples/research_projects/lxmert/modeling_frcnn.py:26 | add "import torch_mlu" |
| 957 | examples/research_projects/lxmert/modeling_frcnn.py:1844 | change ""max_detections"}, pad_value (int), location = {"cuda", "cpu"}" to ""max_detections"}, pad_value (int), location = {"mlu", "cpu"} " |
| 958 | examples/research_projects/lxmert/utils.py:48 | add "import torch_mlu" |
| 959 | examples/research_projects/lxmert/processing_image.py:22 | add "import torch_mlu" |
| 960 | examples/research_projects/jax-projects/hybrid_clip/run_hybrid_clip.py:38 | add "import torch_mlu" |
| 961 | examples/research_projects/visual_bert/extracting_data.py:11 | add "import torch_mlu" |
| 962 | examples/research_projects/visual_bert/extracting_data.py:69 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 963 | examples/research_projects/visual_bert/extracting_data.py:70 | change "self.config.model.device = "cuda"" to "self.config.model.device = "mlu" " |
| 964 | examples/research_projects/visual_bert/visualizing_image.py:26 | add "import torch_mlu" |
| 965 | examples/research_projects/visual_bert/modeling_frcnn.py:26 | add "import torch_mlu" |
| 966 | examples/research_projects/visual_bert/modeling_frcnn.py:1844 | change ""max_detections"}, pad_value (int), location = {"cuda", "cpu"}" to ""max_detections"}, pad_value (int), location = {"mlu", "cpu"} " |
| 967 | examples/research_projects/visual_bert/utils.py:48 | add "import torch_mlu" |
| 968 | examples/research_projects/visual_bert/processing_image.py:22 | add "import torch_mlu" |
| 969 | examples/research_projects/wav2vec2/run_pretrain.py:8 | add "import torch_mlu" |
| 970 | examples/research_projects/wav2vec2/run_pretrain.py:31 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 971 | examples/research_projects/wav2vec2/alignment.py:8 | add "import torch_mlu" |
| 972 | examples/research_projects/wav2vec2/alignment.py:16 | change "def __init__(self, model_name, input_wavs_sr, cuda):" to "def __init__(self, model_name, input_wavs_sr, mlu): " |
| 973 | examples/research_projects/wav2vec2/alignment.py:17 | change "self.cuda = cuda" to "self.mlu = mlu " |
| 974 | examples/research_projects/wav2vec2/alignment.py:21 | change "if self.cuda:" to "if self.mlu: " |
| 975 | examples/research_projects/wav2vec2/alignment.py:22 | change "self.model.to(device="cuda")" to "self.model.to(device="mlu") " |
| 976 | examples/research_projects/wav2vec2/alignment.py:46 | change "if self.cuda:" to "if self.mlu: " |
| 977 | examples/research_projects/wav2vec2/alignment.py:47 | change "inputs = inputs.to(device="cuda")" to "inputs = inputs.to(device="mlu") " |
| 978 | examples/research_projects/wav2vec2/alignment.py:214 | change "parser.add_argument("--cuda", action="store_true")" to "parser.add_argument("--mlu", action="store_true") " |
| 979 | examples/research_projects/wav2vec2/alignment.py:218 | change "aligner = Wav2Vec2Aligner(args.model_name, args.input_wavs_sr, args.cuda)" to "aligner = Wav2Vec2Aligner(args.model_name, args.input_wavs_sr, args.mlu) " |
| 980 | examples/research_projects/wav2vec2/run_common_voice.py:12 | add "import torch_mlu" |
| 981 | examples/research_projects/wav2vec2/run_common_voice.py:38 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 982 | examples/research_projects/wav2vec2/run_asr.py:12 | add "import torch_mlu" |
| 983 | examples/research_projects/wav2vec2/run_asr.py:35 | change "from torch.cuda.amp import autocast" to "from torch.mlu.amp import autocast " |
| 984 | examples/research_projects/information-gain-filtration/run_clm_igf.py:33 | add "import torch_mlu" |
| 985 | examples/research_projects/information-gain-filtration/run_clm_igf.py:84 | change "device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 986 | examples/research_projects/information-gain-filtration/run_clm_igf.py:97 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 987 | examples/research_projects/information-gain-filtration/run_clm_igf.py:140 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 988 | examples/research_projects/information-gain-filtration/run_clm_igf.py:182 | change "device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 989 | examples/research_projects/information-gain-filtration/run_clm_igf.py:207 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 990 | examples/research_projects/information-gain-filtration/run_clm_igf.py:240 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 991 | examples/research_projects/information-gain-filtration/run_clm_igf.py:261 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 992 | examples/research_projects/information-gain-filtration/igf/igf.py:10 | add "import torch_mlu" |
| 993 | examples/research_projects/information-gain-filtration/igf/igf.py:32 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 994 | examples/research_projects/information-gain-filtration/igf/igf.py:117 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 995 | examples/research_projects/information-gain-filtration/igf/igf.py:183 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 996 | examples/research_projects/information-gain-filtration/igf/igf.py:287 | change "device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")" to "device = torch.device("mlu:0" if torch.mlu.is_available() else "cpu") " |
| 997 | examples/research_projects/vqgan-clip/VQGAN_CLIP.py:5 | add "import torch_mlu" |
| 998 | examples/research_projects/vqgan-clip/loaders.py:3 | add "import torch_mlu" |
| 999 | examples/research_projects/vqgan-clip/loaders.py:58 | change "model.cuda()" to "model.mlu() " |
| 1000 | examples/research_projects/vqgan-clip/utils.py:4 | add "import torch_mlu" |
| 1001 | examples/research_projects/vqgan-clip/utils.py:13 | change "device = "cuda" if torch.cuda.is_available() else "cpu"" to "device = "mlu" if torch.mlu.is_available() else "cpu" " |
| 1002 | examples/research_projects/vqgan-clip/img_processing.py:3 | add "import torch_mlu" |
| 1003 | examples/research_projects/seq2seq-distillation/distillation.py:11 | add "import torch_mlu" |
| 1004 | examples/research_projects/seq2seq-distillation/distillation.py:112 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1005 | examples/research_projects/seq2seq-distillation/_test_bash_script.py:10 | add "import torch_mlu" |
| 1006 | examples/research_projects/seq2seq-distillation/convert_pl_checkpoint_to_hf.py:8 | add "import torch_mlu" |
| 1007 | examples/research_projects/seq2seq-distillation/run_eval.py:12 | add "import torch_mlu" |
| 1008 | examples/research_projects/seq2seq-distillation/run_eval.py:22 | change "DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"" to "DEFAULT_DEVICE = "mlu" if torch.mlu.is_available() else "cpu" " |
| 1009 | examples/research_projects/seq2seq-distillation/run_eval.py:95 | change "parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")" to "parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="mlu, mlu:1, cpu etc.") " |
| 1010 | examples/research_projects/seq2seq-distillation/callbacks.py:6 | add "import torch_mlu" |
| 1011 | examples/research_projects/seq2seq-distillation/finetune.py:15 | add "import torch_mlu" |
| 1012 | examples/research_projects/seq2seq-distillation/_test_seq2seq_examples_multi_gpu.py:7 | add "import torch_mlu" |
| 1013 | examples/research_projects/seq2seq-distillation/_test_seq2seq_examples_multi_gpu.py:13 | change "CUDA_AVAILABLE = torch.cuda.is_available()" to "CUDA_AVAILABLE = torch.mlu.is_available() " |
| 1014 | examples/research_projects/seq2seq-distillation/_test_seq2seq_examples.py:11 | add "import torch_mlu" |
| 1015 | examples/research_projects/seq2seq-distillation/_test_seq2seq_examples.py:28 | change "CUDA_AVAILABLE = torch.cuda.is_available()" to "CUDA_AVAILABLE = torch.mlu.is_available() " |
| 1016 | examples/research_projects/seq2seq-distillation/utils.py:14 | add "import torch_mlu" |
| 1017 | examples/research_projects/zero-shot-distillation/distill_classifier.py:7 | add "import torch_mlu" |
| 1018 | examples/research_projects/zero-shot-distillation/distill_classifier.py:168 | change "no_cuda: bool," to "no_mlu: bool, " |
| 1019 | examples/research_projects/zero-shot-distillation/distill_classifier.py:176 | change "if not no_cuda and torch.cuda.is_available():" to "if not no_mlu and torch.mlu.is_available(): " |
| 1020 | examples/research_projects/zero-shot-distillation/distill_classifier.py:177 | change "model = nn.DataParallel(model.cuda())" to "model = nn.DataParallel(model.mlu()) " |
| 1021 | examples/research_projects/zero-shot-distillation/distill_classifier.py:196 | change "with torch.cuda.amp.autocast(enabled=fp16):" to "with torch.mlu.amp.autocast(enabled=fp16): " |
| 1022 | examples/research_projects/zero-shot-distillation/distill_classifier.py:290 | change "training_args.no_cuda," to "training_args.no_mlu, " |
| 1023 | examples/research_projects/xtreme-s/run_xtreme_s.py:29 | add "import torch_mlu" |
| 1024 | examples/tensorflow/test_tensorflow_examples.py:82 | change "def is_cuda_available():" to "def is_mlu_available(): " |
| 1025 | examples/tensorflow/test_tensorflow_examples.py:112 | change "if is_cuda_available():" to "if is_mlu_available(): " |
| 1026 | examples/flax/vision/run_image_classification.py:37 | add "import torch_mlu" |
| 1027 | examples/pytorch/test_pytorch_examples.py:24 | add "import torch_mlu" |
| 1028 | examples/pytorch/test_pytorch_examples.py:96 | change "def is_cuda_and_apex_available():" to "def is_mlu_and_apex_available(): " |
| 1029 | examples/pytorch/test_pytorch_examples.py:97 | change "is_using_cuda = torch.cuda.is_available() and torch_device == "cuda"" to "is_using_mlu = torch.mlu.is_available() and torch_device == "mlu" " |
| 1030 | examples/pytorch/test_pytorch_examples.py:98 | change "return is_using_cuda and is_apex_available()" to "return is_using_mlu and is_apex_available() " |
| 1031 | examples/pytorch/test_pytorch_examples.py:126 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1032 | examples/pytorch/test_pytorch_examples.py:151 | change "if torch.cuda.device_count() > 1:" to "if torch.mlu.device_count() > 1: " |
| 1033 | examples/pytorch/test_pytorch_examples.py:155 | change "if torch_device != "cuda":" to "if torch_device != "mlu": " |
| 1034 | examples/pytorch/test_pytorch_examples.py:156 | change "testargs.append("--no_cuda")" to "testargs.append("--no_mlu") " |
| 1035 | examples/pytorch/test_pytorch_examples.py:177 | change "if torch_device != "cuda":" to "if torch_device != "mlu": " |
| 1036 | examples/pytorch/test_pytorch_examples.py:178 | change "testargs.append("--no_cuda")" to "testargs.append("--no_mlu") " |
| 1037 | examples/pytorch/test_pytorch_examples.py:203 | change "if torch_device != "cuda":" to "if torch_device != "mlu": " |
| 1038 | examples/pytorch/test_pytorch_examples.py:204 | change "testargs.append("--no_cuda")" to "testargs.append("--no_mlu") " |
| 1039 | examples/pytorch/test_pytorch_examples.py:233 | change "if torch_device != "cuda":" to "if torch_device != "mlu": " |
| 1040 | examples/pytorch/test_pytorch_examples.py:234 | change "testargs.append("--no_cuda")" to "testargs.append("--no_mlu") " |
| 1041 | examples/pytorch/test_pytorch_examples.py:322 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1042 | examples/pytorch/test_pytorch_examples.py:411 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1043 | examples/pytorch/test_pytorch_examples.py:441 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1044 | examples/pytorch/test_pytorch_examples.py:471 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1045 | examples/pytorch/test_pytorch_examples.py:503 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1046 | examples/pytorch/test_pytorch_examples.py:529 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1047 | examples/pytorch/test_pytorch_examples.py:557 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1048 | examples/pytorch/test_pytorch_examples.py:582 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1049 | examples/pytorch/test_accelerate_examples.py:26 | add "import torch_mlu" |
| 1050 | examples/pytorch/test_accelerate_examples.py:56 | change "def is_cuda_and_apex_available():" to "def is_mlu_and_apex_available(): " |
| 1051 | examples/pytorch/test_accelerate_examples.py:57 | change "is_using_cuda = torch.cuda.is_available() and torch_device == "cuda"" to "is_using_mlu = torch.mlu.is_available() and torch_device == "mlu" " |
| 1052 | examples/pytorch/test_accelerate_examples.py:58 | change "return is_using_cuda and is_apex_available()" to "return is_using_mlu and is_apex_available() " |
| 1053 | examples/pytorch/test_accelerate_examples.py:95 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1054 | examples/pytorch/test_accelerate_examples.py:121 | change "if torch.cuda.device_count() > 1:" to "if torch.mlu.device_count() > 1: " |
| 1055 | examples/pytorch/test_accelerate_examples.py:326 | change "if is_cuda_and_apex_available():" to "if is_mlu_and_apex_available(): " |
| 1056 | examples/pytorch/multiple-choice/run_swag.py:30 | add "import torch_mlu" |
| 1057 | examples/pytorch/multiple-choice/run_swag_no_trainer.py:34 | add "import torch_mlu" |
| 1058 | examples/pytorch/token-classification/run_ner_no_trainer.py:31 | add "import torch_mlu" |
| 1059 | examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py:30 | add "import torch_mlu" |
| 1060 | examples/pytorch/speech-recognition/run_speech_recognition_ctc.py:32 | add "import torch_mlu" |
| 1061 | examples/pytorch/image-classification/run_image_classification.py:24 | add "import torch_mlu" |
| 1062 | examples/pytorch/image-classification/run_image_classification_no_trainer.py:25 | add "import torch_mlu" |
| 1063 | examples/pytorch/language-modeling/run_clm.py:34 | add "import torch_mlu" |
| 1064 | examples/pytorch/language-modeling/run_clm_no_trainer.py:35 | add "import torch_mlu" |
| 1065 | examples/pytorch/language-modeling/run_mlm_no_trainer.py:35 | add "import torch_mlu" |
| 1066 | examples/pytorch/translation/run_translation_no_trainer.py:32 | add "import torch_mlu" |
| 1067 | examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py:32 | add "import torch_mlu" |
| 1068 | examples/pytorch/question-answering/run_qa_no_trainer.py:32 | add "import torch_mlu" |
| 1069 | examples/pytorch/speech-pretraining/run_wav2vec2_pretraining_no_trainer.py:26 | add "import torch_mlu" |
| 1070 | examples/pytorch/text-generation/run_generation.py:25 | add "import torch_mlu" |
| 1071 | examples/pytorch/text-generation/run_generation.py:80 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 1072 | examples/pytorch/text-generation/run_generation.py:192 | change "parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Avoid using CUDA when available") " |
| 1073 | examples/pytorch/text-generation/run_generation.py:201 | change "args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 1074 | examples/pytorch/text-generation/run_generation.py:202 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 1075 | examples/pytorch/text-generation/run_generation_contrastive_search.py:27 | add "import torch_mlu" |
| 1076 | examples/pytorch/text-generation/run_generation_contrastive_search.py:44 | change "torch.cuda.manual_seed_all(args.seed)" to "torch.mlu.manual_seed_all(args.seed) " |
| 1077 | examples/pytorch/text-generation/run_generation_contrastive_search.py:76 | change "parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")" to "parser.add_argument("--no_mlu", action="store_true", help="Avoid using CUDA when available") " |
| 1078 | examples/pytorch/text-generation/run_generation_contrastive_search.py:84 | change "args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")" to "args.device = torch.device("mlu" if torch.mlu.is_available() and not args.no_mlu else "cpu") " |
| 1079 | examples/pytorch/text-generation/run_generation_contrastive_search.py:85 | change "args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()" to "args.n_gpu = 0 if args.no_mlu else torch.mlu.device_count() " |
| 1080 | examples/pytorch/semantic-segmentation/run_semantic_segmentation_no_trainer.py:27 | add "import torch_mlu" |
| 1081 | examples/pytorch/semantic-segmentation/run_semantic_segmentation.py:26 | add "import torch_mlu" |
| 1082 | examples/pytorch/contrastive-image-text/run_clip.py:32 | add "import torch_mlu" |
| 1083 | examples/pytorch/text-classification/run_glue_no_trainer.py:26 | add "import torch_mlu" |
| 1084 | examples/pytorch/image-pretraining/run_mim.py:23 | add "import torch_mlu" |
| 1085 | examples/pytorch/image-pretraining/run_mae.py:22 | add "import torch_mlu" |
| 1086 | examples/pytorch/summarization/run_summarization_no_trainer.py:33 | add "import torch_mlu" |
| 1087 | scripts/benchmark/trainer-benchmark.py:116 | add "import torch_mlu" |
| 1088 | scripts/benchmark/trainer-benchmark.py:283 | change "properties = torch.cuda.get_device_properties(torch.device("cuda"))" to "properties = torch.mlu.get_device_properties(torch.device("mlu")) " |
| 1089 | scripts/benchmark/trainer-benchmark.py:290 | change "cuda        : {torch.version.cuda}" to "mlu        : {torch.version.mlu} " |
| 1090 | scripts/benchmark/trainer-benchmark.py:294 | change "{torch.cuda.device_count()} GPUs      : {properties.name}, {properties.total_memory/2**30:0.2f}GB" to "{torch.mlu.device_count()} GPUs      : {properties.name}, {properties.total_memory/2**30:0.2f}GB " |
| 1091 | scripts/distributed/torch-distributed-gpu-test.py:5 | change "# many nodes) can talk to each other via nccl and allocate gpu memory." to "# many nodes) can talk to each other via cncl and allocate gpu memory. " |
| 1092 | scripts/distributed/torch-distributed-gpu-test.py:49 | add "import torch_mlu" |
| 1093 | scripts/distributed/torch-distributed-gpu-test.py:64 | change "torch.cuda.set_device(local_rank)" to "torch.mlu.set_device(local_rank) " |
| 1094 | scripts/distributed/torch-distributed-gpu-test.py:65 | change "device = torch.device("cuda", local_rank)" to "device = torch.device("mlu", local_rank) " |
| 1095 | scripts/distributed/torch-distributed-gpu-test.py:72 | change "dist.init_process_group("nccl")" to "dist.init_process_group("cncl") " |
| 1096 | scripts/distributed/torch-distributed-gpu-test.py:76 | change "# test cuda is available and can allocate memory" to "# test mlu is available and can allocate memory " |
| 1097 | scripts/distributed/torch-distributed-gpu-test.py:77 | change "torch.cuda.is_available()" to "torch.mlu.is_available() " |
| 1098 | scripts/distributed/torch-distributed-gpu-test.py:78 | change "torch.ones(1).cuda(local_rank)" to "torch.ones(1).mlu(local_rank) " |
| 1099 | scripts/distributed/torch-distributed-gpu-test.py:88 | change "printflock(f"pt={torch.__version__}, cuda={torch.version.cuda}, nccl={torch.cuda.nccl.version()}")" to "printflock(f"pt={torch.__version__}, mlu={torch.version.mlu}, cncl={torch.mlu.cncl.version()}") " |
| 1100 | tests/test_configuration_common.py:198 | add "import torch_mlu" |
| 1101 | tests/test_modeling_flax_common.py:69 | add "import torch_mlu" |
| 1102 | tests/test_modeling_common.py:111 | add "import torch_mlu" |
| 1103 | tests/test_modeling_common.py:2315 | change "# move input tensors to cuda:O" to "# move input tensors to mlu:O " |
| 1104 | tests/test_modeling_common.py:2337 | change """"returns a list of cuda memory allocations per GPU in MBs"""" to """"returns a list of mlu memory allocations per GPU in MBs""" " |
| 1105 | tests/test_modeling_common.py:2340 | change "for id in range(torch.cuda.device_count()):" to "for id in range(torch.mlu.device_count()): " |
| 1106 | tests/test_modeling_common.py:2341 | change "with torch.cuda.device(id):" to "with torch.mlu.device(id): " |
| 1107 | tests/test_modeling_common.py:2342 | change "per_device_memory.append(torch.cuda.memory_allocated() >> 20)" to "per_device_memory.append(torch.mlu.memory_allocated() >> 20) " |
| 1108 | tests/test_modeling_common.py:2350 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1109 | tests/test_modeling_common.py:2353 | change "# Retrieve initial memory usage (can easily be ~0.6-1.5GB if cuda-kernels have been preloaded by previous tests)" to "# Retrieve initial memory usage (can easily be ~0.6-1.5GB if mlu-kernels have been preloaded by previous tests) " |
| 1110 | tests/test_modeling_common.py:2358 | change "model.to("cuda:0")" to "model.to("mlu:0") " |
| 1111 | tests/test_modeling_common.py:2366 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1112 | tests/test_modeling_common.py:2390 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1113 | tests/test_modeling_common.py:2417 | change "parallel_output = model(**cast_to_device(inputs_dict, "cuda:0"))" to "parallel_output = model(**cast_to_device(inputs_dict, "mlu:0")) " |
| 1114 | tests/test_modeling_common.py:2452 | change "model.generate(**cast_to_device(inputs_dict, "cuda:0"), num_beams=2)" to "model.generate(**cast_to_device(inputs_dict, "mlu:0"), num_beams=2) " |
| 1115 | tests/test_modeling_common.py:3232 | change "# cuda memory tracking and then we should be able to do a much more precise test." to "# mlu memory tracking and then we should be able to do a much more precise test. " |
| 1116 | tests/test_tokenization_common.py:2363 | add "import torch_mlu" |
| 1117 | tests/test_tokenization_common.py:3853 | change "training_args = TrainingArguments(output_dir=tmp_dir, do_train=True, no_cuda=True)" to "training_args = TrainingArguments(output_dir=tmp_dir, do_train=True, no_mlu=True) " |
| 1118 | tests/test_modeling_tf_common.py:130 | add "import torch_mlu" |
| 1119 | tests/test_image_transforms.py:26 | add "import torch_mlu" |
| 1120 | tests/test_image_processing_common.py:48 | add "import torch_mlu" |
| 1121 | tests/optimization/test_optimization.py:26 | add "import torch_mlu" |
| 1122 | tests/trainer/test_trainer_distributed.py:33 | add "import torch_mlu" |
| 1123 | tests/trainer/test_trainer_distributed.py:87 | change "--nproc_per_node={torch.cuda.device_count()}" to "--nproc_per_node={torch.mlu.device_count()} " |
| 1124 | tests/trainer/test_data_collator.py:38 | add "import torch_mlu" |
| 1125 | tests/trainer/test_trainer_tpu.py:33 | add "import torch_mlu" |
| 1126 | tests/trainer/test_trainer_utils.py:28 | add "import torch_mlu" |
| 1127 | tests/trainer/test_trainer.py:86 | add "import torch_mlu" |
| 1128 | tests/trainer/test_trainer.py:656 | change "trainer = get_regression_trainer(learning_rate=0.1, use_ipex=True, bf16=mix_bf16, no_cuda=True)" to "trainer = get_regression_trainer(learning_rate=0.1, use_ipex=True, bf16=mix_bf16, no_mlu=True) " |
| 1129 | tests/trainer/test_trainer.py:662 | change "learning_rate=0.1, num_train_epochs=1.5, use_ipex=True, bf16=mix_bf16, no_cuda=True" to "learning_rate=0.1, num_train_epochs=1.5, use_ipex=True, bf16=mix_bf16, no_mlu=True " |
| 1130 | tests/trainer/test_trainer.py:669 | change "learning_rate=0.1, max_steps=10, use_ipex=True, bf16=mix_bf16, no_cuda=True" to "learning_rate=0.1, max_steps=10, use_ipex=True, bf16=mix_bf16, no_mlu=True " |
| 1131 | tests/trainer/test_trainer.py:700 | change "n_gpu = max(1, torch.cuda.device_count())" to "n_gpu = max(1, torch.mlu.device_count()) " |
| 1132 | tests/trainer/test_trainer.py:900 | change "a=1.5, b=2.5, use_ipex=True, compute_metrics=AlmostAccuracy(), bf16=mix_bf16, no_cuda=True" to "a=1.5, b=2.5, use_ipex=True, compute_metrics=AlmostAccuracy(), bf16=mix_bf16, no_mlu=True " |
| 1133 | tests/trainer/test_trainer.py:919 | change "no_cuda=True," to "no_mlu=True, " |
| 1134 | tests/trainer/test_trainer.py:938 | change "no_cuda=True," to "no_mlu=True, " |
| 1135 | tests/trainer/test_trainer.py:1019 | change "trainer = get_regression_trainer(a=1.5, b=2.5, use_ipex=True, bf16=mix_bf16, no_cuda=True)" to "trainer = get_regression_trainer(a=1.5, b=2.5, use_ipex=True, bf16=mix_bf16, no_mlu=True) " |
| 1136 | tests/trainer/test_trainer.py:1025 | change "trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, use_ipex=True, bf16=mix_bf16, no_cuda=True)" to "trainer = get_regression_trainer(a=1.5, b=2.5, eval_len=66, use_ipex=True, bf16=mix_bf16, no_mlu=True) " |
| 1137 | tests/trainer/test_trainer.py:1032 | change "a=1.5, b=2.5, double_output=True, use_ipex=True, bf16=mix_bf16, no_cuda=True" to "a=1.5, b=2.5, double_output=True, use_ipex=True, bf16=mix_bf16, no_mlu=True " |
| 1138 | tests/trainer/test_trainer.py:1048 | change "no_cuda=True," to "no_mlu=True, " |
| 1139 | tests/trainer/test_trainer.py:1251 | change "random_torch = not torch.cuda.is_available() or torch.cuda.device_count() <= 1" to "random_torch = not torch.mlu.is_available() or torch.mlu.device_count() <= 1 " |
| 1140 | tests/trainer/test_trainer.py:1253 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 1141 | tests/trainer/test_trainer.py:1306 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 1142 | tests/trainer/test_trainer.py:1535 | change "training_args = TrainingArguments(output_dir="./examples", no_cuda=True)" to "training_args = TrainingArguments(output_dir="./examples", no_mlu=True) " |
| 1143 | tests/trainer/test_trainer.py:1737 | change "if torch.cuda.device_count() > 0:" to "if torch.mlu.device_count() > 0: " |
| 1144 | tests/trainer/test_trainer.py:1743 | change "if torch.cuda.device_count() > 0:" to "if torch.mlu.device_count() > 0: " |
| 1145 | tests/trainer/test_trainer.py:1748 | change "if torch.cuda.device_count() > 0:" to "if torch.mlu.device_count() > 0: " |
| 1146 | tests/trainer/test_trainer.py:1886 | change "a = torch.ones(1024, 1024, device="cuda", requires_grad=True)" to "a = torch.ones(1024, 1024, device="mlu", requires_grad=True) " |
| 1147 | tests/trainer/test_trainer.py:1895 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1148 | tests/trainer/test_trainer.py:1896 | change "torch.cuda.reset_peak_memory_stats()" to "torch.mlu.reset_peak_memory_stats() " |
| 1149 | tests/trainer/test_trainer.py:1899 | change "orig_peak_mem = torch.cuda.max_memory_allocated()" to "orig_peak_mem = torch.mlu.max_memory_allocated() " |
| 1150 | tests/trainer/test_trainer.py:1904 | change "a = torch.ones(1024, 1024, device="cuda", requires_grad=True)" to "a = torch.ones(1024, 1024, device="mlu", requires_grad=True) " |
| 1151 | tests/trainer/test_trainer.py:1914 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1152 | tests/trainer/test_trainer.py:1915 | change "torch.cuda.reset_peak_memory_stats()" to "torch.mlu.reset_peak_memory_stats() " |
| 1153 | tests/trainer/test_trainer.py:1918 | change "peak_mem = torch.cuda.max_memory_allocated()" to "peak_mem = torch.mlu.max_memory_allocated() " |
| 1154 | tests/tokenization/test_tokenization_utils.py:138 | add "import torch_mlu" |
| 1155 | tests/generation/test_beam_search.py:26 | add "import torch_mlu" |
| 1156 | tests/generation/test_flax_utils.py:38 | add "import torch_mlu" |
| 1157 | tests/generation/test_utils.py:30 | add "import torch_mlu" |
| 1158 | tests/generation/test_beam_constraints.py:24 | add "import torch_mlu" |
| 1159 | tests/generation/test_stopping_criteria.py:26 | add "import torch_mlu" |
| 1160 | tests/generation/test_logits_process.py:28 | add "import torch_mlu" |
| 1161 | tests/onnx/test_onnx_v2.py:31 | add "import torch_mlu" |
| 1162 | tests/onnx/test_onnx_v2.py:438 | change "def test_pytorch_export_on_cuda(self, test_name, name, model_name, feature, onnx_config_class_constructor):" to "def test_pytorch_export_on_mlu(self, test_name, name, model_name, feature, onnx_config_class_constructor): " |
| 1163 | tests/onnx/test_onnx_v2.py:439 | change "self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, device="cuda")" to "self._onnx_export(test_name, name, model_name, feature, onnx_config_class_constructor, device="mlu") " |
| 1164 | tests/onnx/test_onnx_v2.py:456 | change "def test_pytorch_export_encoder_decoder_models_on_cuda(" to "def test_pytorch_export_encoder_decoder_models_on_mlu( " |
| 1165 | tests/onnx/test_onnx_v2.py:460 | change "test_name, name, model_name, feature, onnx_config_class_constructor, device="cuda"" to "test_name, name, model_name, feature, onnx_config_class_constructor, device="mlu" " |
| 1166 | tests/models/flava/test_image_processing_flava.py:28 | add "import torch_mlu" |
| 1167 | tests/models/flava/test_modeling_flava.py:49 | add "import torch_mlu" |
| 1168 | tests/models/albert/test_modeling_albert.py:29 | add "import torch_mlu" |
| 1169 | tests/models/esm/test_modeling_esmfold.py:29 | add "import torch_mlu" |
| 1170 | tests/models/esm/test_modeling_esm.py:29 | add "import torch_mlu" |
| 1171 | tests/models/deberta/test_modeling_deberta.py:26 | add "import torch_mlu" |
| 1172 | tests/models/pegasus_x/test_modeling_pegasus_x.py:34 | add "import torch_mlu" |
| 1173 | tests/models/pegasus_x/test_modeling_pegasus_x.py:281 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1174 | tests/models/led/test_modeling_led.py:34 | add "import torch_mlu" |
| 1175 | tests/models/led/test_modeling_led.py:361 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1176 | tests/models/hubert/test_modeling_hubert.py:42 | add "import torch_mlu" |
| 1177 | tests/models/decision_transformer/test_modeling_decision_transformer.py:31 | add "import torch_mlu" |
| 1178 | tests/models/electra/test_modeling_electra.py:29 | add "import torch_mlu" |
| 1179 | tests/models/rag/test_modeling_rag.py:52 | add "import torch_mlu" |
| 1180 | tests/models/rag/test_modeling_rag.py:200 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1181 | tests/models/rag/test_modeling_rag.py:688 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1182 | tests/models/rag/test_modeling_rag.py:1001 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1183 | tests/models/rag/test_modeling_rag.py:1041 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1184 | tests/models/rag/test_retrieval_rag.py:322 | add "import torch_mlu" |
| 1185 | tests/models/speech_encoder_decoder/test_modeling_flax_speech_encoder_decoder.py:52 | add "import torch_mlu" |
| 1186 | tests/models/speech_encoder_decoder/test_modeling_speech_encoder_decoder.py:32 | add "import torch_mlu" |
| 1187 | tests/models/mask2former/test_image_processing_mask2former.py:30 | add "import torch_mlu" |
| 1188 | tests/models/mask2former/test_modeling_mask2former.py:33 | add "import torch_mlu" |
| 1189 | tests/models/convbert/test_modeling_convbert.py:30 | add "import torch_mlu" |
| 1190 | tests/models/mctct/test_modeling_mctct.py:32 | add "import torch_mlu" |
| 1191 | tests/models/mctct/test_feature_extraction_mctct.py:247 | add "import torch_mlu" |
| 1192 | tests/models/videomae/test_image_processing_videomae.py:28 | add "import torch_mlu" |
| 1193 | tests/models/videomae/test_modeling_videomae.py:36 | add "import torch_mlu" |
| 1194 | tests/models/dit/test_modeling_dit.py:23 | add "import torch_mlu" |
| 1195 | tests/models/x_clip/test_modeling_x_clip.py:42 | add "import torch_mlu" |
| 1196 | tests/models/x_clip/test_modeling_x_clip.py:288 | change "# move input tensors to cuda:O" to "# move input tensors to mlu:O " |
| 1197 | tests/models/dinat/test_modeling_dinat.py:31 | add "import torch_mlu" |
| 1198 | tests/models/mgp_str/test_modeling_mgp_str.py:31 | add "import torch_mlu" |
| 1199 | tests/models/mgp_str/test_processor_mgp_str.py:33 | add "import torch_mlu" |
| 1200 | tests/models/oneformer/test_modeling_oneformer.py:34 | add "import torch_mlu" |
| 1201 | tests/models/oneformer/test_image_processing_oneformer.py:30 | add "import torch_mlu" |
| 1202 | tests/models/oneformer/test_processor_oneformer.py:33 | add "import torch_mlu" |
| 1203 | tests/models/mpnet/test_modeling_mpnet.py:28 | add "import torch_mlu" |
| 1204 | tests/models/deit/test_modeling_deit.py:40 | add "import torch_mlu" |
| 1205 | tests/models/deit/test_image_processing_deit.py:28 | add "import torch_mlu" |
| 1206 | tests/models/table_transformer/test_modeling_table_transformer.py:34 | add "import torch_mlu" |
| 1207 | tests/models/xlm/test_modeling_xlm.py:28 | add "import torch_mlu" |
| 1208 | tests/models/xlnet/test_modeling_xlnet.py:29 | add "import torch_mlu" |
| 1209 | tests/models/squeezebert/test_modeling_squeezebert.py:28 | add "import torch_mlu" |
| 1210 | tests/models/detr/test_modeling_detr.py:33 | add "import torch_mlu" |
| 1211 | tests/models/detr/test_image_processing_detr.py:30 | add "import torch_mlu" |
| 1212 | tests/models/opt/test_modeling_opt.py:34 | add "import torch_mlu" |
| 1213 | tests/models/opt/test_modeling_opt.py:281 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1214 | tests/models/opt/test_modeling_opt.py:508 | change "model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()" to "model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).mlu() " |
| 1215 | tests/models/opt/test_modeling_opt.py:513 | change "input_ids = batch["input_ids"].cuda()" to "input_ids = batch["input_ids"].mlu() " |
| 1216 | tests/models/opt/test_modeling_opt.py:514 | change "attention_mask = batch["attention_mask"].cuda()" to "attention_mask = batch["attention_mask"].mlu() " |
| 1217 | tests/models/reformer/test_modeling_reformer.py:35 | add "import torch_mlu" |
| 1218 | tests/models/reformer/test_tokenization_reformer.py:330 | add "import torch_mlu" |
| 1219 | tests/models/gpt_neox/test_modeling_gpt_neox.py:29 | add "import torch_mlu" |
| 1220 | tests/models/conditional_detr/test_image_processing_conditional_detr.py:30 | add "import torch_mlu" |
| 1221 | tests/models/conditional_detr/test_modeling_conditional_detr.py:33 | add "import torch_mlu" |
| 1222 | tests/models/layoutlmv3/test_modeling_layoutlmv3.py:30 | add "import torch_mlu" |
| 1223 | tests/models/layoutlmv3/test_image_processing_layoutlmv3.py:28 | add "import torch_mlu" |
| 1224 | tests/models/layoutlmv3/test_tokenization_layoutlmv3.py:1152 | add "import torch_mlu" |
| 1225 | tests/models/mbart/test_modeling_mbart.py:33 | add "import torch_mlu" |
| 1226 | tests/models/mbart/test_modeling_mbart.py:315 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1227 | tests/models/mbart/test_modeling_mbart.py:360 | change "if "cuda" in torch_device:" to "if "mlu" in torch_device: " |
| 1228 | tests/models/retribert/test_tokenization_retribert.py:344 | add "import torch_mlu" |
| 1229 | tests/models/tvlt/test_feature_extraction_tvlt.py:33 | add "import torch_mlu" |
| 1230 | tests/models/tvlt/test_image_processor_tvlt.py:28 | add "import torch_mlu" |
| 1231 | tests/models/tvlt/test_modeling_tvlt.py:40 | add "import torch_mlu" |
| 1232 | tests/models/sew/test_modeling_sew.py:38 | add "import torch_mlu" |
| 1233 | tests/models/xglm/test_modeling_xglm.py:30 | add "import torch_mlu" |
| 1234 | tests/models/xglm/test_modeling_xglm.py:486 | change "model = XGLMForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).cuda()" to "model = XGLMForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, use_cache=True).mlu() " |
| 1235 | tests/models/xglm/test_modeling_xglm.py:491 | change "input_ids = batch["input_ids"].cuda()" to "input_ids = batch["input_ids"].mlu() " |
| 1236 | tests/models/xglm/test_modeling_xglm.py:492 | change "attention_mask = batch["attention_mask"].cuda()" to "attention_mask = batch["attention_mask"].mlu() " |
| 1237 | tests/models/xglm/test_modeling_flax_xglm.py:41 | add "import torch_mlu" |
| 1238 | tests/models/trajectory_transformer/test_modeling_trajectory_transformer.py:33 | add "import torch_mlu" |
| 1239 | tests/models/xlm_roberta_xl/test_modeling_xlm_roberta_xl.py:29 | add "import torch_mlu" |
| 1240 | tests/models/beit/test_image_processing_beit.py:29 | add "import torch_mlu" |
| 1241 | tests/models/beit/test_modeling_beit.py:35 | add "import torch_mlu" |
| 1242 | tests/models/dpt/test_modeling_dpt_hybrid.py:32 | add "import torch_mlu" |
| 1243 | tests/models/dpt/test_modeling_dpt.py:32 | add "import torch_mlu" |
| 1244 | tests/models/dpt/test_image_processing_dpt.py:28 | add "import torch_mlu" |
| 1245 | tests/models/transfo_xl/test_modeling_transfo_xl.py:30 | add "import torch_mlu" |
| 1246 | tests/models/resnet/test_modeling_resnet.py:31 | add "import torch_mlu" |
| 1247 | tests/models/qdqbert/test_modeling_qdqbert.py:30 | add "import torch_mlu" |
| 1248 | tests/models/clap/test_modeling_clap.py:42 | add "import torch_mlu" |
| 1249 | tests/models/clap/test_feature_extraction_clap.py:31 | add "import torch_mlu" |
| 1250 | tests/models/funnel/test_modeling_funnel.py:29 | add "import torch_mlu" |
| 1251 | tests/models/owlvit/test_image_processing_owlvit.py:28 | add "import torch_mlu" |
| 1252 | tests/models/owlvit/test_modeling_owlvit.py:42 | add "import torch_mlu" |
| 1253 | tests/models/speech_to_text/test_modeling_speech_to_text.py:42 | add "import torch_mlu" |
| 1254 | tests/models/speech_to_text/test_modeling_speech_to_text.py:332 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1255 | tests/models/speech_to_text/test_feature_extraction_speech_to_text.py:239 | add "import torch_mlu" |
| 1256 | tests/models/sew_d/test_modeling_sew_d.py:38 | add "import torch_mlu" |
| 1257 | tests/models/roberta_prelayernorm/test_modeling_roberta_prelayernorm.py:29 | add "import torch_mlu" |
| 1258 | tests/models/deberta_v2/test_modeling_deberta_v2.py:26 | add "import torch_mlu" |
| 1259 | tests/models/plbart/test_modeling_plbart.py:33 | add "import torch_mlu" |
| 1260 | tests/models/plbart/test_modeling_plbart.py:311 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1261 | tests/models/plbart/test_modeling_plbart.py:356 | change "if "cuda" in torch_device:" to "if "mlu" in torch_device: " |
| 1262 | tests/models/bart/test_modeling_bart.py:35 | add "import torch_mlu" |
| 1263 | tests/models/bart/test_modeling_bart.py:390 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1264 | tests/models/bart/test_modeling_bart.py:504 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1265 | tests/models/lilt/test_modeling_lilt.py:29 | add "import torch_mlu" |
| 1266 | tests/models/vision_text_dual_encoder/test_modeling_flax_vision_text_dual_encoder.py:56 | add "import torch_mlu" |
| 1267 | tests/models/vision_text_dual_encoder/test_modeling_vision_text_dual_encoder.py:36 | add "import torch_mlu" |
| 1268 | tests/models/ernie_m/test_modeling_ernie_m.py:29 | add "import torch_mlu" |
| 1269 | tests/models/convnextv2/test_modeling_convnextv2.py:32 | add "import torch_mlu" |
| 1270 | tests/models/bigbird_pegasus/test_modeling_bigbird_pegasus.py:32 | add "import torch_mlu" |
| 1271 | tests/models/bigbird_pegasus/test_modeling_bigbird_pegasus.py:375 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1272 | tests/models/realm/test_modeling_realm.py:31 | add "import torch_mlu" |
| 1273 | tests/models/switch_transformers/test_modeling_switch_transformers.py:31 | add "import torch_mlu" |
| 1274 | tests/models/xlm_prophetnet/test_modeling_xlm_prophetnet.py:24 | add "import torch_mlu" |
| 1275 | tests/models/megatron_bert/test_modeling_megatron_bert.py:32 | add "import torch_mlu" |
| 1276 | tests/models/longt5/test_modeling_longt5.py:33 | add "import torch_mlu" |
| 1277 | tests/models/gpt2/test_modeling_gpt2.py:31 | add "import torch_mlu" |
| 1278 | tests/models/gpt2/test_modeling_flax_gpt2.py:40 | add "import torch_mlu" |
| 1279 | tests/models/openai/test_modeling_openai.py:29 | add "import torch_mlu" |
| 1280 | tests/models/jukebox/test_modeling_jukebox.py:24 | add "import torch_mlu" |
| 1281 | tests/models/jukebox/test_modeling_jukebox.py:169 | change "torch.backends.cuda.matmul.allow_tf32 = False" to "torch.backends.mlu.matmul.allow_tf32 = False " |
| 1282 | tests/models/jukebox/test_modeling_jukebox.py:197 | change "torch.backends.cuda.matmul.allow_tf32 = False" to "torch.backends.mlu.matmul.allow_tf32 = False " |
| 1283 | tests/models/jukebox/test_modeling_jukebox.py:367 | change "labels = [i.cuda() for i in self.prepare_inputs(self.model_id)]" to "labels = [i.mlu() for i in self.prepare_inputs(self.model_id)] " |
| 1284 | tests/models/jukebox/test_modeling_jukebox.py:370 | change "model.priors[0].cuda()" to "model.priors[0].mlu() " |
| 1285 | tests/models/jukebox/test_modeling_jukebox.py:371 | change "zs = [torch.zeros(1, 0, dtype=torch.long).cuda() for _ in range(3)]" to "zs = [torch.zeros(1, 0, dtype=torch.long).mlu() for _ in range(3)] " |
| 1286 | tests/models/jukebox/test_modeling_jukebox.py:377 | change "model.priors[1].cuda()" to "model.priors[1].mlu() " |
| 1287 | tests/models/jukebox/test_modeling_jukebox.py:383 | change "model.priors[2].cuda()" to "model.priors[2].mlu() " |
| 1288 | tests/models/jukebox/test_modeling_jukebox.py:390 | change "model = JukeboxPrior.from_pretrained(prior_id, min_duration=0).eval().half().to("cuda")" to "model = JukeboxPrior.from_pretrained(prior_id, min_duration=0).eval().half().to("mlu") " |
| 1289 | tests/models/jukebox/test_modeling_jukebox.py:392 | change "labels = self.prepare_inputs(prior_id)[0].cuda()" to "labels = self.prepare_inputs(prior_id)[0].mlu() " |
| 1290 | tests/models/jukebox/test_tokenization_jukebox.py:50 | add "import torch_mlu" |
| 1291 | tests/models/vit_mae/test_modeling_vit_mae.py:35 | add "import torch_mlu" |
| 1292 | tests/models/nezha/test_modeling_nezha.py:30 | add "import torch_mlu" |
| 1293 | tests/models/gpt_neox_japanese/test_modeling_gpt_neox_japanese.py:30 | add "import torch_mlu" |
| 1294 | tests/models/vilt/test_modeling_vilt.py:33 | add "import torch_mlu" |
| 1295 | tests/models/vilt/test_image_processing_vilt.py:28 | add "import torch_mlu" |
| 1296 | tests/models/vision_encoder_decoder/test_modeling_flax_vision_encoder_decoder.py:44 | add "import torch_mlu" |
| 1297 | tests/models/vision_encoder_decoder/test_modeling_tf_vision_encoder_decoder.py:58 | add "import torch_mlu" |
| 1298 | tests/models/vision_encoder_decoder/test_modeling_vision_encoder_decoder.py:45 | add "import torch_mlu" |
| 1299 | tests/models/convnext/test_image_processing_convnext.py:28 | add "import torch_mlu" |
| 1300 | tests/models/convnext/test_modeling_convnext.py:31 | add "import torch_mlu" |
| 1301 | tests/models/blip_2/test_modeling_blip_2.py:41 | add "import torch_mlu" |
| 1302 | tests/models/bloom/test_modeling_bloom.py:30 | add "import torch_mlu" |
| 1303 | tests/models/bloom/test_modeling_bloom.py:409 | change "# Please see: https://pytorch.org/docs/stable/notes/cuda.html#reduced-precision-reduction-in-fp16-gemms" to "# Please see: https://pytorch.org/docs/stable/notes/mlu.html#reduced-precision-reduction-in-fp16-gemms " |
| 1304 | tests/models/bloom/test_modeling_bloom.py:426 | change "model = BloomForCausalLM.from_pretrained(path_560m, use_cache=True, revision="gs555750").cuda()" to "model = BloomForCausalLM.from_pretrained(path_560m, use_cache=True, revision="gs555750").mlu() " |
| 1305 | tests/models/bloom/test_modeling_bloom.py:438 | change "greedy_output = model.generate(input_ids.cuda(), max_length=50)" to "greedy_output = model.generate(input_ids.mlu(), max_length=50) " |
| 1306 | tests/models/bloom/test_modeling_bloom.py:446 | change "model = BloomForCausalLM.from_pretrained(path_560m, use_cache=True, revision="gs555750").cuda()" to "model = BloomForCausalLM.from_pretrained(path_560m, use_cache=True, revision="gs555750").mlu() " |
| 1307 | tests/models/bloom/test_modeling_bloom.py:454 | change "input_ids["input_ids"].cuda(), attention_mask=input_ids["attention_mask"], max_length=50, do_sample=False" to "input_ids["input_ids"].mlu(), attention_mask=input_ids["attention_mask"], max_length=50, do_sample=False " |
| 1308 | tests/models/bloom/test_modeling_bloom.py:466 | change "model = BloomForCausalLM.from_pretrained(path_560m, use_cache=True, revision="gs555750").cuda()" to "model = BloomForCausalLM.from_pretrained(path_560m, use_cache=True, revision="gs555750").mlu() " |
| 1309 | tests/models/bloom/test_modeling_bloom.py:477 | change "input_ids["input_ids"].cuda(), attention_mask=input_ids["attention_mask"], max_length=50, do_sample=False" to "input_ids["input_ids"].mlu(), attention_mask=input_ids["attention_mask"], max_length=50, do_sample=False " |
| 1310 | tests/models/bloom/test_modeling_bloom.py:479 | change "greedy_output_without_pad = model.generate(input_ids_without_pad.cuda(), max_length=50, do_sample=False)" to "greedy_output_without_pad = model.generate(input_ids_without_pad.mlu(), max_length=50, do_sample=False) " |
| 1311 | tests/models/bloom/test_modeling_bloom.py:755 | change "is_torch_less_than_1_9, reason="Test failed with torch < 1.9 (`min_cuda` not implemented for `BFloat16`)"" to "is_torch_less_than_1_9, reason="Test failed with torch < 1.9 (`min_mlu` not implemented for `BFloat16`)" " |
| 1312 | tests/models/bloom/test_modeling_bloom.py:758 | change "cuda_available = torch.cuda.is_available()" to "mlu_available = torch.mlu.is_available() " |
| 1313 | tests/models/bloom/test_modeling_bloom.py:779 | change "if cuda_available:" to "if mlu_available: " |
| 1314 | tests/models/bloom/test_modeling_bloom.py:788 | change "cuda_available = torch.cuda.is_available()" to "mlu_available = torch.mlu.is_available() " |
| 1315 | tests/models/bloom/test_modeling_bloom.py:806 | change "if cuda_available:" to "if mlu_available: " |
| 1316 | tests/models/nystromformer/test_modeling_nystromformer.py:29 | add "import torch_mlu" |
| 1317 | tests/models/deformable_detr/test_modeling_deformable_detr.py:34 | add "import torch_mlu" |
| 1318 | tests/models/deformable_detr/test_modeling_deformable_detr.py:655 | change "model.to("cuda")" to "model.to("mlu") " |
| 1319 | tests/models/deformable_detr/test_modeling_deformable_detr.py:658 | change "gpu_outputs = model(pixel_values.to("cuda"), pixel_mask.to("cuda"))" to "gpu_outputs = model(pixel_values.to("mlu"), pixel_mask.to("mlu")) " |
| 1320 | tests/models/deformable_detr/test_image_processing_deformable_detr.py:30 | add "import torch_mlu" |
| 1321 | tests/models/layoutxlm/test_tokenization_layoutxlm.py:1176 | add "import torch_mlu" |
| 1322 | tests/models/codegen/test_modeling_codegen.py:30 | add "import torch_mlu" |
| 1323 | tests/models/codegen/test_modeling_codegen.py:492 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1324 | tests/models/codegen/test_modeling_codegen.py:493 | change "torch.cuda.manual_seed(0)" to "torch.mlu.manual_seed(0) " |
| 1325 | tests/models/codegen/test_modeling_codegen.py:508 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1326 | tests/models/chinese_clip/test_modeling_chinese_clip.py:42 | add "import torch_mlu" |
| 1327 | tests/models/chinese_clip/test_image_processing_chinese_clip.py:28 | add "import torch_mlu" |
| 1328 | tests/models/roberta/test_modeling_roberta.py:30 | add "import torch_mlu" |
| 1329 | tests/models/mobilevit/test_modeling_mobilevit.py:31 | add "import torch_mlu" |
| 1330 | tests/models/mobilevit/test_image_processing_mobilevit.py:28 | add "import torch_mlu" |
| 1331 | tests/models/blip/test_image_processing_blip.py:28 | add "import torch_mlu" |
| 1332 | tests/models/blip/test_modeling_blip.py:42 | add "import torch_mlu" |
| 1333 | tests/models/blip/test_modeling_blip_text.py:29 | add "import torch_mlu" |
| 1334 | tests/models/luke/test_modeling_luke.py:27 | add "import torch_mlu" |
| 1335 | tests/models/glpn/test_image_processing_glpn.py:28 | add "import torch_mlu" |
| 1336 | tests/models/glpn/test_modeling_glpn.py:31 | add "import torch_mlu" |
| 1337 | tests/models/fnet/test_modeling_fnet.py:31 | add "import torch_mlu" |
| 1338 | tests/models/unispeech_sat/test_modeling_unispeech_sat.py:39 | add "import torch_mlu" |
| 1339 | tests/models/mvp/test_modeling_mvp.py:35 | add "import torch_mlu" |
| 1340 | tests/models/mvp/test_modeling_mvp.py:381 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1341 | tests/models/mvp/test_modeling_mvp.py:512 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1342 | tests/models/auto/test_modeling_auto.py:44 | add "import torch_mlu" |
| 1343 | tests/models/speech_to_text_2/test_modeling_speech_to_text_2.py:29 | add "import torch_mlu" |
| 1344 | tests/models/swin2sr/test_image_processing_swin2sr.py:28 | add "import torch_mlu" |
| 1345 | tests/models/swin2sr/test_modeling_swin2sr.py:29 | add "import torch_mlu" |
| 1346 | tests/models/graphormer/test_modeling_graphormer.py:33 | add "import torch_mlu" |
| 1347 | tests/models/megatron_gpt2/test_modeling_megatron_gpt2.py:24 | add "import torch_mlu" |
| 1348 | tests/models/whisper/test_modeling_whisper.py:42 | add "import torch_mlu" |
| 1349 | tests/models/whisper/test_modeling_whisper.py:394 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1350 | tests/models/whisper/test_feature_extraction_whisper.py:37 | add "import torch_mlu" |
| 1351 | tests/models/vit_msn/test_modeling_vit_msn.py:31 | add "import torch_mlu" |
| 1352 | tests/models/levit/test_modeling_levit.py:36 | add "import torch_mlu" |
| 1353 | tests/models/levit/test_image_processing_levit.py:28 | add "import torch_mlu" |
| 1354 | tests/models/bert_generation/test_tokenization_bert_generation.py:213 | add "import torch_mlu" |
| 1355 | tests/models/bert_generation/test_modeling_bert_generation.py:29 | add "import torch_mlu" |
| 1356 | tests/models/lxmert/test_modeling_lxmert.py:32 | add "import torch_mlu" |
| 1357 | tests/models/lxmert/test_modeling_tf_lxmert.py:493 | add "import torch_mlu" |
| 1358 | tests/models/unispeech/test_modeling_unispeech.py:39 | add "import torch_mlu" |
| 1359 | tests/models/wavlm/test_modeling_wavlm.py:38 | add "import torch_mlu" |
| 1360 | tests/models/tapas/test_tokenization_tapas.py:1032 | add "import torch_mlu" |
| 1361 | tests/models/tapas/test_modeling_tapas.py:44 | add "import torch_mlu" |
| 1362 | tests/models/ibert/test_modeling_ibert.py:29 | add "import torch_mlu" |
| 1363 | tests/models/gpt_neo/test_modeling_gpt_neo.py:31 | add "import torch_mlu" |
| 1364 | tests/models/gpt_neo/test_modeling_flax_gpt_neo.py:40 | add "import torch_mlu" |
| 1365 | tests/models/markuplm/test_modeling_markuplm.py:29 | add "import torch_mlu" |
| 1366 | tests/models/markuplm/test_tokenization_markuplm.py:1042 | add "import torch_mlu" |
| 1367 | tests/models/yolos/test_modeling_yolos.py:31 | add "import torch_mlu" |
| 1368 | tests/models/yolos/test_image_processing_yolos.py:30 | add "import torch_mlu" |
| 1369 | tests/models/big_bird/test_tokenization_big_bird.py:182 | add "import torch_mlu" |
| 1370 | tests/models/big_bird/test_modeling_big_bird.py:31 | add "import torch_mlu" |
| 1371 | tests/models/splinter/test_modeling_splinter.py:29 | add "import torch_mlu" |
| 1372 | tests/models/splinter/test_modeling_splinter.py:340 | change "# move input tensors to cuda:O" to "# move input tensors to mlu:O " |
| 1373 | tests/models/maskformer/test_image_processing_maskformer.py:30 | add "import torch_mlu" |
| 1374 | tests/models/maskformer/test_modeling_maskformer_swin.py:32 | add "import torch_mlu" |
| 1375 | tests/models/maskformer/test_modeling_maskformer.py:33 | add "import torch_mlu" |
| 1376 | tests/models/mobilebert/test_modeling_mobilebert.py:29 | add "import torch_mlu" |
| 1377 | tests/models/git/test_modeling_git.py:32 | add "import torch_mlu" |
| 1378 | tests/models/altclip/test_modeling_altclip.py:42 | add "import torch_mlu" |
| 1379 | tests/models/wav2vec2_with_lm/test_processor_wav2vec2_with_lm.py:436 | add "import torch_mlu" |
| 1380 | tests/models/vit/test_image_processing_vit.py:28 | add "import torch_mlu" |
| 1381 | tests/models/vit/test_modeling_vit.py:38 | add "import torch_mlu" |
| 1382 | tests/models/donut/test_modeling_donut_swin.py:31 | add "import torch_mlu" |
| 1383 | tests/models/donut/test_image_processing_donut.py:28 | add "import torch_mlu" |
| 1384 | tests/models/dpr/test_modeling_dpr.py:29 | add "import torch_mlu" |
| 1385 | tests/models/roc_bert/test_modeling_roc_bert.py:29 | add "import torch_mlu" |
| 1386 | tests/models/van/test_modeling_van.py:35 | add "import torch_mlu" |
| 1387 | tests/models/imagegpt/test_modeling_imagegpt.py:40 | add "import torch_mlu" |
| 1388 | tests/models/imagegpt/test_image_processing_imagegpt.py:32 | add "import torch_mlu" |
| 1389 | tests/models/align/test_modeling_align.py:46 | add "import torch_mlu" |
| 1390 | tests/models/visual_bert/test_modeling_visual_bert.py:29 | add "import torch_mlu" |
| 1391 | tests/models/t5/test_modeling_t5.py:39 | add "import torch_mlu" |
| 1392 | tests/models/bridgetower/test_image_processing_bridgetower.py:29 | add "import torch_mlu" |
| 1393 | tests/models/bridgetower/test_modeling_bridgetower.py:38 | add "import torch_mlu" |
| 1394 | tests/models/nat/test_modeling_nat.py:31 | add "import torch_mlu" |
| 1395 | tests/models/encoder_decoder/test_modeling_encoder_decoder.py:34 | add "import torch_mlu" |
| 1396 | tests/models/encoder_decoder/test_modeling_tf_encoder_decoder.py:56 | add "import torch_mlu" |
| 1397 | tests/models/encoder_decoder/test_modeling_flax_encoder_decoder.py:47 | add "import torch_mlu" |
| 1398 | tests/models/bit/test_modeling_bit.py:31 | add "import torch_mlu" |
| 1399 | tests/models/ernie/test_modeling_ernie.py:30 | add "import torch_mlu" |
| 1400 | tests/models/mobilenet_v1/test_image_processing_mobilenet_v1.py:28 | add "import torch_mlu" |
| 1401 | tests/models/mobilenet_v1/test_modeling_mobilenet_v1.py:31 | add "import torch_mlu" |
| 1402 | tests/models/vit_hybrid/test_modeling_vit_hybrid.py:31 | add "import torch_mlu" |
| 1403 | tests/models/wav2vec2/test_modeling_wav2vec2.py:56 | add "import torch_mlu" |
| 1404 | tests/models/wav2vec2/test_feature_extraction_wav2vec2.py:200 | add "import torch_mlu" |
| 1405 | tests/models/fsmt/test_modeling_fsmt.py:33 | add "import torch_mlu" |
| 1406 | tests/models/fsmt/test_modeling_fsmt.py:369 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1407 | tests/models/fsmt/test_modeling_fsmt.py:448 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1408 | tests/models/fsmt/test_modeling_fsmt.py:505 | change "device = 0 if torch_device == "cuda" else -1" to "device = 0 if torch_device == "mlu" else -1 " |
| 1409 | tests/models/biogpt/test_modeling_biogpt.py:30 | add "import torch_mlu" |
| 1410 | tests/models/bort/test_modeling_bort.py:23 | add "import torch_mlu" |
| 1411 | tests/models/mobilenet_v2/test_image_processing_mobilenet_v2.py:28 | add "import torch_mlu" |
| 1412 | tests/models/mobilenet_v2/test_modeling_mobilenet_v2.py:31 | add "import torch_mlu" |
| 1413 | tests/models/blenderbot/test_modeling_blenderbot.py:31 | add "import torch_mlu" |
| 1414 | tests/models/blenderbot/test_modeling_blenderbot.py:276 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1415 | tests/models/blenderbot/test_modeling_blenderbot.py:317 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1416 | tests/models/marian/test_modeling_marian.py:33 | add "import torch_mlu" |
| 1417 | tests/models/marian/test_modeling_marian.py:288 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1418 | tests/models/marian/test_modeling_marian.py:425 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1419 | tests/models/marian/test_modeling_marian.py:597 | change "device = 0 if torch_device == "cuda" else -1" to "device = 0 if torch_device == "mlu" else -1 " |
| 1420 | tests/models/ctrl/test_modeling_ctrl.py:29 | add "import torch_mlu" |
| 1421 | tests/models/ctrl/test_modeling_ctrl.py:233 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1422 | tests/models/ctrl/test_modeling_ctrl.py:259 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1423 | tests/models/flaubert/test_modeling_flaubert.py:28 | add "import torch_mlu" |
| 1424 | tests/models/layoutlm/test_modeling_layoutlm.py:26 | add "import torch_mlu" |
| 1425 | tests/models/bert/test_modeling_bert.py:30 | add "import torch_mlu" |
| 1426 | tests/models/xmod/test_modeling_xmod.py:27 | add "import torch_mlu" |
| 1427 | tests/models/roformer/test_modeling_roformer.py:29 | add "import torch_mlu" |
| 1428 | tests/models/swin/test_modeling_swin.py:31 | add "import torch_mlu" |
| 1429 | tests/models/time_series_transformer/test_modeling_time_series_transformer.py:35 | add "import torch_mlu" |
| 1430 | tests/models/efficientnet/test_modeling_efficientnet.py:30 | add "import torch_mlu" |
| 1431 | tests/models/efficientnet/test_image_processing_efficientnet.py:28 | add "import torch_mlu" |
| 1432 | tests/models/regnet/test_modeling_regnet.py:31 | add "import torch_mlu" |
| 1433 | tests/models/data2vec/test_modeling_data2vec_audio.py:33 | add "import torch_mlu" |
| 1434 | tests/models/data2vec/test_modeling_data2vec_vision.py:32 | add "import torch_mlu" |
| 1435 | tests/models/data2vec/test_modeling_data2vec_text.py:30 | add "import torch_mlu" |
| 1436 | tests/models/distilbert/test_modeling_distilbert.py:28 | add "import torch_mlu" |
| 1437 | tests/models/informer/test_modeling_informer.py:34 | add "import torch_mlu" |
| 1438 | tests/models/m2m_100/test_modeling_m2m_100.py:33 | add "import torch_mlu" |
| 1439 | tests/models/m2m_100/test_modeling_m2m_100.py:319 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1440 | tests/models/audio_spectrogram_transformer/test_modeling_audio_spectrogram_transformer.py:32 | add "import torch_mlu" |
| 1441 | tests/models/audio_spectrogram_transformer/test_feature_extraction_audio_spectrogram_transformer.py:33 | add "import torch_mlu" |
| 1442 | tests/models/clip/test_modeling_flax_clip.py:25 | add "import torch_mlu" |
| 1443 | tests/models/clip/test_modeling_clip.py:50 | add "import torch_mlu" |
| 1444 | tests/models/clip/test_image_processing_clip.py:28 | add "import torch_mlu" |
| 1445 | tests/models/xlm_roberta/test_modeling_xlm_roberta.py:24 | add "import torch_mlu" |
| 1446 | tests/models/perceiver/test_modeling_perceiver.py:39 | add "import torch_mlu" |
| 1447 | tests/models/segformer/test_modeling_segformer.py:31 | add "import torch_mlu" |
| 1448 | tests/models/segformer/test_image_processing_segformer.py:29 | add "import torch_mlu" |
| 1449 | tests/models/longformer/test_modeling_longformer.py:28 | add "import torch_mlu" |
| 1450 | tests/models/rembert/test_modeling_rembert.py:29 | add "import torch_mlu" |
| 1451 | tests/models/upernet/test_modeling_upernet.py:33 | add "import torch_mlu" |
| 1452 | tests/models/prophetnet/test_modeling_prophetnet.py:30 | add "import torch_mlu" |
| 1453 | tests/models/canine/test_modeling_canine.py:30 | add "import torch_mlu" |
| 1454 | tests/models/wav2vec2_conformer/test_modeling_wav2vec2_conformer.py:38 | add "import torch_mlu" |
| 1455 | tests/models/trocr/test_modeling_trocr.py:29 | add "import torch_mlu" |
| 1456 | tests/models/cvt/test_modeling_cvt.py:32 | add "import torch_mlu" |
| 1457 | tests/models/timesformer/test_modeling_timesformer.py:36 | add "import torch_mlu" |
| 1458 | tests/models/blenderbot_small/test_modeling_blenderbot_small.py:31 | add "import torch_mlu" |
| 1459 | tests/models/blenderbot_small/test_modeling_blenderbot_small.py:270 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1460 | tests/models/blenderbot_small/test_modeling_blenderbot_small.py:302 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1461 | tests/models/poolformer/test_modeling_poolformer.py:31 | add "import torch_mlu" |
| 1462 | tests/models/poolformer/test_image_processing_poolformer.py:27 | add "import torch_mlu" |
| 1463 | tests/models/groupvit/test_modeling_groupvit.py:43 | add "import torch_mlu" |
| 1464 | tests/models/groupvit/test_modeling_groupvit.py:175 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 1465 | tests/models/groupvit/test_modeling_groupvit.py:564 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 1466 | tests/models/groupvit/test_modeling_tf_groupvit.py:298 | add "import torch_mlu" |
| 1467 | tests/models/groupvit/test_modeling_tf_groupvit.py:304 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 1468 | tests/models/groupvit/test_modeling_tf_groupvit.py:620 | change "torch.cuda.manual_seed_all(seed)" to "torch.mlu.manual_seed_all(seed) " |
| 1469 | tests/models/speecht5/test_feature_extraction_speecht5.py:34 | add "import torch_mlu" |
| 1470 | tests/models/speecht5/test_modeling_speecht5.py:46 | add "import torch_mlu" |
| 1471 | tests/models/clipseg/test_modeling_clipseg.py:51 | add "import torch_mlu" |
| 1472 | tests/models/layoutlmv2/test_image_processing_layoutlmv2.py:28 | add "import torch_mlu" |
| 1473 | tests/models/layoutlmv2/test_tokenization_layoutlmv2.py:1274 | add "import torch_mlu" |
| 1474 | tests/models/layoutlmv2/test_modeling_layoutlmv2.py:29 | add "import torch_mlu" |
| 1475 | tests/models/swinv2/test_modeling_swinv2.py:30 | add "import torch_mlu" |
| 1476 | tests/models/pegasus/test_modeling_pegasus.py:32 | add "import torch_mlu" |
| 1477 | tests/models/pegasus/test_modeling_pegasus.py:287 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1478 | tests/models/pegasus/test_modeling_pegasus.py:346 | change "if "cuda" not in torch_device:" to "if "mlu" not in torch_device: " |
| 1479 | tests/models/deta/test_modeling_deta.py:33 | add "import torch_mlu" |
| 1480 | tests/models/deta/test_image_processing_deta.py:30 | add "import torch_mlu" |
| 1481 | tests/models/efficientformer/test_image_processing_efficientformer.py:28 | add "import torch_mlu" |
| 1482 | tests/models/efficientformer/test_modeling_efficientformer.py:33 | add "import torch_mlu" |
| 1483 | tests/models/gptj/test_modeling_flax_gptj.py:40 | add "import torch_mlu" |
| 1484 | tests/models/gptj/test_modeling_gptj.py:30 | add "import torch_mlu" |
| 1485 | tests/models/gptj/test_modeling_gptj.py:550 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1486 | tests/models/camembert/test_modeling_camembert.py:23 | add "import torch_mlu" |
| 1487 | tests/models/yoso/test_modeling_yoso.py:29 | add "import torch_mlu" |
| 1488 | tests/pipelines/test_pipelines_depth_estimation.py:34 | add "import torch_mlu" |
| 1489 | tests/pipelines/test_pipelines_text2text_generation.py:30 | add "import torch_mlu" |
| 1490 | tests/pipelines/test_pipelines_feature_extraction.py:34 | add "import torch_mlu" |
| 1491 | tests/pipelines/test_pipelines_automatic_speech_recognition.py:50 | add "import torch_mlu" |
| 1492 | tests/pipelines/test_pipelines_automatic_speech_recognition.py:171 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 1493 | tests/pipelines/test_pipelines_text_classification.py:88 | add "import torch_mlu" |
| 1494 | tests/pipelines/test_pipelines_text_generation.py:254 | add "import torch_mlu" |
| 1495 | tests/pipelines/test_pipelines_common.py:254 | add "import torch_mlu" |
| 1496 | tests/utils/test_generic.py:40 | add "import torch_mlu" |
| 1497 | tests/utils/test_image_utils.py:28 | add "import torch_mlu" |
| 1498 | tests/utils/test_activations.py:22 | add "import torch_mlu" |
| 1499 | tests/utils/test_skip_decorators.py:55 | change "def check_slow_torch_cuda():" to "def check_slow_torch_mlu(): " |
| 1500 | tests/utils/test_skip_decorators.py:57 | change "if run_slow and torch_device == "cuda":" to "if run_slow and torch_device == "mlu": " |
| 1501 | tests/utils/test_skip_decorators.py:68 | change "check_slow_torch_cuda()" to "check_slow_torch_mlu() " |
| 1502 | tests/utils/test_skip_decorators.py:73 | change "check_slow_torch_cuda()" to "check_slow_torch_mlu() " |
| 1503 | tests/utils/test_skip_decorators.py:102 | change "check_slow_torch_cuda()" to "check_slow_torch_mlu() " |
| 1504 | tests/utils/test_skip_decorators.py:108 | change "check_slow_torch_cuda()" to "check_slow_torch_mlu() " |
| 1505 | tests/mixed_int8/test_mixed_int8.py:43 | add "import torch_mlu" |
| 1506 | tests/mixed_int8/test_mixed_int8.py:110 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1507 | tests/mixed_int8/test_mixed_int8.py:190 | change "self.model_8bit.to(torch.device("cuda:0"))" to "self.model_8bit.to(torch.device("mlu:0")) " |
| 1508 | tests/mixed_int8/test_mixed_int8.py:242 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1509 | tests/mixed_int8/test_mixed_int8.py:325 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1510 | tests/mixed_int8/test_mixed_int8.py:355 | change "torch.cuda.empty_cache()" to "torch.mlu.empty_cache() " |
| 1511 | tests/mixed_int8/test_mixed_int8.py:363 | change "# self._clear_cuda_cache()" to "# self._clear_mlu_cache() " |
| 1512 | tests/mixed_int8/test_mixed_int8.py:577 | change "with torch.cuda.amp.autocast():" to "with torch.mlu.amp.autocast(): " |
| 1513 | tests/deepspeed/test_deepspeed.py:275 | change "until this pytest worker exits. This is because the gpu memory allocated by the cuda-kernels" to "until this pytest worker exits. This is because the gpu memory allocated by the mlu-kernels " |
| 1514 | utils/past_ci_versions.py:12 | change ""cuda": "cu113"," to ""mlu": "cu113", " |
| 1515 | utils/past_ci_versions.py:23 | change ""cuda": "cu113"," to ""mlu": "cu113", " |
| 1516 | utils/past_ci_versions.py:34 | change ""cuda": "cu113"," to ""mlu": "cu113", " |
| 1517 | utils/past_ci_versions.py:46 | change ""cuda": "cu111"," to ""mlu": "cu111", " |
| 1518 | utils/past_ci_versions.py:57 | change ""cuda": "cu111"," to ""mlu": "cu111", " |
| 1519 | utils/past_ci_versions.py:68 | change ""cuda": "cu110"," to ""mlu": "cu110", " |
| 1520 | utils/past_ci_versions.py:79 | change ""cuda": "cu101"," to ""mlu": "cu101", " |
| 1521 | utils/past_ci_versions.py:90 | change ""cuda": "cu101"," to ""mlu": "cu101", " |
| 1522 | utils/past_ci_versions.py:101 | change ""cuda": "cu100"," to ""mlu": "cu100", " |
| 1523 | utils/past_ci_versions.py:125 | change "# need another `nvidia:cuda` docker image, otherwise GPU not working" to "# need another `nvidia:mlu` docker image, otherwise GPU not working " |
| 1524 | utils/past_ci_versions.py:131 | change ""base_docker": "nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04"," to ""base_docker": "nvidia/mlu:11.0.3-cudnn8-devel-ubuntu20.04", " |
| 1525 | utils/past_ci_versions.py:150 | change "cuda = """ to "mlu = "" " |
| 1526 | utils/past_ci_versions.py:152 | change "cuda = info["cuda"]" to "mlu = info["mlu"] " |
| 1527 | utils/past_ci_versions.py:153 | change "os.system(f"echo \"export CUDA='{cuda}'\" >> ~/.profile")" to "os.system(f"echo \"export CUDA='{mlu}'\" >> ~/.profile") " |
| 1528 | utils/past_ci_versions.py:154 | change "print(f"echo \"export CUDA='{cuda}'\" >> ~/.profile")" to "print(f"echo \"export CUDA='{mlu}'\" >> ~/.profile") " |
| 1529 | utils/notification_service.py:870 | change ""Torch CUDA extension tests": "run_tests_torch_cuda_extensions_gpu_test_reports"," to ""Torch CUDA extension tests": "run_tests_torch_mlu_extensions_gpu_test_reports", " |
| 1530 | utils/print_env.py:32 | add "import torch_mlu" |
| 1531 | utils/print_env.py:35 | change "print("Cuda available:", torch.cuda.is_available())" to "print("Cuda available:", torch.mlu.is_available()) " |
| 1532 | utils/print_env.py:36 | change "print("Cuda version:", torch.version.cuda)" to "print("Cuda version:", torch.version.mlu) " |
| 1533 | utils/print_env.py:38 | change "print("Number of GPUs available:", torch.cuda.device_count())" to "print("Number of GPUs available:", torch.mlu.device_count()) " |
| 1534 | utils/print_env.py:39 | change "print("NCCL version:", torch.cuda.nccl.version())" to "print("NCCL version:", torch.mlu.cncl.version()) " |
| 1535 | utils/test_module/custom_modeling.py:1 | add "import torch_mlu" |
| 1536 | templates/adding_a_new_model/cookiecutter-template-{{cookiecutter.modelname}}/test_modeling_{{cookiecutter.lowercase_modelname}}.py:31 | add "import torch_mlu" |
| 1537 | templates/adding_a_new_model/cookiecutter-template-{{cookiecutter.modelname}}/test_modeling_{{cookiecutter.lowercase_modelname}}.py:743 | change "if torch_device == "cuda":" to "if torch_device == "mlu": " |
| 1538 | templates/adding_a_new_model/cookiecutter-template-{{cookiecutter.modelname}}/modeling_{{cookiecutter.lowercase_modelname}}.py:23 | add "import torch_mlu" |
| 1539 | templates/adding_a_new_example_script/{{cookiecutter.directory_name}}/run_{{cookiecutter.example_shortcut}}.py:31 | add "import torch_mlu" |
