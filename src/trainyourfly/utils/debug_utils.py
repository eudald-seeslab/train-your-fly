import logging
import sys
import time

logger = logging.getLogger(__name__)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug("Execution time: %.3f seconds", end_time - start_time)
        return result

    return wrapper


def get_size_of_tensor(x):
    return (x.indices().nelement() * 8 + x.values().nelement() * 4) / 1024**2


def model_summary(model, print_=True):
    param_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if print_:
                logger.debug("%s: %s", name, param.shape)
            param_info[name] = list(param.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_info["total_params"] = total_params
    if print_:
        logger.debug("The model has %d parameters.", total_params)
    return param_info


def debugger_is_active() -> bool:
    """Return if the debugger is currently active."""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None
