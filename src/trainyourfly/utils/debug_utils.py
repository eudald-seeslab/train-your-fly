import logging
import sys
import time

logger = logging.getLogger("trainyourfly")


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
        return result

    return wrapper


def get_size_of_tensor(x):
    return (x.indices().nelement() * 8 + x.values().nelement() * 4) / 1024**2


def model_summary(model, print_=True):
    param_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if print_:
                logger.debug(f"{name}: {param.shape}")
            param_info[name] = list(param.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_info["total_params"] = total_params
    if print_:
        logger.debug(f"The model has {total_params} parameters.")
    return param_info


# Logging
class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    orange = "\x1b[38;5;208m"
    bold_orange = "\x1b[38;5;202m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: bold_orange + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Create logger
def get_logger(name, debug=False):
    log_level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(log_level)
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
        logger.propagate = False

    return logger


def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None
