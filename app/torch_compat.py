"""Compat for PyTorch 2.6+ when loading vendored YOLO-NAS weights."""

from __future__ import annotations


def patch_libreyolo_weight_loading() -> None:
    """
    LibreYOLO factory uses load_untrusted_torch_file (weights_only=True).
    Deci YOLO-NAS .pt checkpoints need weights_only=False (trusted, from our repo/LFS).
    """
    import libreyolo.utils.serialization as serialization

    _trusted_load = serialization.load_trusted_torch_file

    def load_untrusted_torch_file(
        path,
        *,
        map_location="cpu",
        context: str = "model weights",
    ):
        return _trusted_load(
            path,
            map_location=map_location,
            context=context,
        )

    serialization.load_untrusted_torch_file = load_untrusted_torch_file
