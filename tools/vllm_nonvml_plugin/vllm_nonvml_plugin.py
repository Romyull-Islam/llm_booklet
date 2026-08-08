"""
Force vLLM to use its CUDA platform when NVML is broken but CUDA is fine.

WHY THIS EXISTS
---------------
vLLM decides which platform it is on in `vllm/platforms/__init__.py`:

    def cuda_platform_plugin() -> str | None:
        pynvml = import_pynvml()
        pynvml.nvmlInit()                  # <-- everything hinges on this
        ...
        except Exception as e:
            if "nvml" not in e.__class__.__name__.lower():
                raise e
            # the only fallback is a Jetson check

On a machine where the NVIDIA kernel module and the userspace NVML library are
different versions (driver upgraded on disk, old module still loaded), that call
raises NVMLError_LibRmVersionMismatch, the plugin returns None, and vLLM exits
with "Failed to infer device type" -- even though the CUDA driver API is fine and
torch can use the GPU normally.

vLLM already has the code to run without NVML. `vllm/platforms/cuda.py` ends with:

    CudaPlatform = NvmlCudaPlatform if nvml_available else NonNvmlCudaPlatform

so on this machine `CudaPlatform` is *already* the non-NVML implementation (the
one used on Jetson). Nothing needs patching. The detection step just needs to be
told CUDA is present.

`resolve_current_platform_cls_qualname()` gives out-of-tree plugins priority over
the builtin ones, so registering this under the `vllm.platform_plugins` entry
point group is enough.

This is a workaround, not a repair. The real fix is to reload the NVIDIA kernel
modules (`sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe
nvidia`) or reboot, after which NVML matches and this plugin deactivates itself.
"""


def cuda_without_nvml() -> str | None:
    """Return vLLM's CUDA platform, but only when NVML is the only thing broken.

    Returns None (so vLLM falls back to its normal detection) whenever:
      - NVML works, i.e. there is nothing to work around, or
      - torch cannot see a CUDA device, i.e. there is genuinely no GPU.
    """
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None

    # If NVML is healthy, stay out of the way and let the builtin plugin run.
    try:
        from vllm.utils.import_utils import import_pynvml

        pynvml = import_pynvml()
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return None
    except Exception:
        pass

    return "vllm.platforms.cuda.CudaPlatform"
