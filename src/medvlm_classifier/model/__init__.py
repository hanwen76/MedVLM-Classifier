"""Model wrappers and parameter control.

Keep imports lazy so utility scripts can run even when a specific transformers
version does not provide every class used by training helpers.
"""

from .factory import load_model_and_processor_by_id


def load_vlm_and_processor(*args, **kwargs):
    from .vlm import load_vlm_and_processor as _impl

    return _impl(*args, **kwargs)


def freeze_all_except_projector(*args, **kwargs):
    from .vlm import freeze_all_except_projector as _impl

    return _impl(*args, **kwargs)


def summarize_trainable_params(*args, **kwargs):
    from .vlm import summarize_trainable_params as _impl

    return _impl(*args, **kwargs)


__all__ = ["load_vlm_and_processor", "freeze_all_except_projector", "summarize_trainable_params", "load_model_and_processor_by_id"]
