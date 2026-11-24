"""Prints a JSON description of the machine and software stack (recorded with every set of results)."""
import json, os, platform, sys


def main():
    info = {"python": sys.version.split()[0], "os": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "processor": platform.processor()}
    try:
        import numpy, torch, transformers, tokenizers
        info.update(torch=torch.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                    numpy=numpy.__version__, cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "compute_capability": f"{p.major}.{p.minor}", "vram_gib": round(p.total_memory / 2**30, 2),
                           "sm_count": p.multi_processor_count, "bf16_supported": torch.cuda.is_bf16_supported(),
                           "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                           "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                           "fp16_accumulation_switch": hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")}
            try:
                import triton
                info["triton"] = triton.__version__
            except Exception:
                info["triton"] = None
    except Exception as ex:
        info["error"] = repr(ex)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
