from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch


class NumericalIntegrityError(FloatingPointError):
    """Fatal NaN/Inf detected in the region TD7 training path."""


def _context_suffix(context):
    return f" | {context}" if context else ""


def iter_named_tensors(value, prefix="value"):
    """Yield floating-point tensors from a nested checkpoint/state structure."""
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            yield prefix, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_tensors(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_named_tensors(item, f"{prefix}[{index}]")


def iter_named_arrays(value, prefix="value"):
    """Yield NumPy/Python floating values from a nested state structure."""
    if torch.is_tensor(value):
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc":
            yield prefix, value
        return
    if isinstance(value, (float, complex, np.floating, np.complexfloating)):
        yield prefix, np.asarray(value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_named_arrays(item, child)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_named_arrays(item, f"{prefix}[{index}]")


def _tensor_failure_detail(name, tensor):
    value = tensor.detach()
    finite = torch.isfinite(value)
    bad = ~finite
    bad_count = int(bad.sum().item())
    nan_count = int(torch.isnan(value).sum().item())
    if value.is_complex():
        posinf_count = int(
            (torch.isposinf(value.real).sum() + torch.isposinf(value.imag).sum()).item()
        )
        neginf_count = int(
            (torch.isneginf(value.real).sum() + torch.isneginf(value.imag).sum()).item()
        )
    else:
        posinf_count = int(torch.isposinf(value).sum().item())
        neginf_count = int(torch.isneginf(value).sum().item())
    coordinates = torch.nonzero(bad, as_tuple=False)[:8].detach().cpu().tolist()

    finite_values = value[finite]
    finite_range = "none"
    if finite_values.numel():
        if value.is_complex():
            magnitudes = finite_values.abs()
            finite_range = (
                f"absmin={float(magnitudes.min().cpu()):.6g},"
                f"absmax={float(magnitudes.max().cpu()):.6g}"
            )
        else:
            finite_range = (
                f"min={float(finite_values.min().cpu()):.6g},"
                f"max={float(finite_values.max().cpu()):.6g},"
                f"absmax={float(finite_values.abs().max().cpu()):.6g}"
            )
    return (
        f"{name}: shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"bad={bad_count}, nan={nan_count}, +inf={posinf_count}, "
        f"-inf={neginf_count}, first_bad={coordinates}, finite_range=({finite_range})"
    )


def assert_finite_tensors(stage, named_tensors, context=None):
    """Raise once per stage after a low-overhead, per-device finite check."""
    items = []
    for name, tensor in named_tensors:
        if not torch.is_tensor(tensor):
            continue
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        items.append((str(name), tensor))
    if not items:
        return

    by_device = defaultdict(list)
    for name, tensor in items:
        by_device[str(tensor.device)].append((name, tensor))

    failed = False
    for device_items in by_device.values():
        checks = [torch.isfinite(tensor.detach()).all() for _, tensor in device_items]
        if not bool(torch.stack(checks).all().item()):
            failed = True
            break
    if not failed:
        return

    bad_items = []
    for name, tensor in items:
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            bad_items.append((name, tensor))
    detail_limit = 12
    details = [
        _tensor_failure_detail(name, tensor)
        for name, tensor in bad_items[:detail_limit]
    ]
    if len(bad_items) > detail_limit:
        details.append(f"... {len(bad_items) - detail_limit} additional bad tensors omitted")
    detail_text = "; ".join(details)
    raise NumericalIntegrityError(
        f"[{stage}] non-finite tensor detected{_context_suffix(context)} | {detail_text}"
    )


def assert_finite_tree(stage, value, context=None):
    assert_finite_tensors(stage, iter_named_tensors(value), context=context)
    assert_finite_arrays(stage, iter_named_arrays(value), context=context)


def assert_finite_modules(stage, named_modules, context=None):
    tensors = []
    for module_name, module in named_modules:
        tensors.extend(
            (f"{module_name}.parameter.{name}", parameter)
            for name, parameter in module.named_parameters()
        )
        tensors.extend(
            (f"{module_name}.buffer.{name}", buffer)
            for name, buffer in module.named_buffers()
        )
    assert_finite_tensors(stage, tensors, context=context)


def _array_failure_detail(name, array):
    value = np.asarray(array)
    finite = np.isfinite(value)
    bad = ~finite
    coordinates = np.argwhere(bad)[:8].tolist()
    finite_values = value[finite]
    finite_range = "none"
    if finite_values.size:
        if np.iscomplexobj(value):
            magnitudes = np.abs(finite_values)
            finite_range = (
                f"absmin={float(magnitudes.min()):.6g},"
                f"absmax={float(magnitudes.max()):.6g}"
            )
        else:
            finite_range = (
                f"min={float(finite_values.min()):.6g},"
                f"max={float(finite_values.max()):.6g},"
                f"absmax={float(np.abs(finite_values).max()):.6g}"
            )
    if np.iscomplexobj(value):
        posinf_count = int(np.isposinf(value.real).sum() + np.isposinf(value.imag).sum())
        neginf_count = int(np.isneginf(value.real).sum() + np.isneginf(value.imag).sum())
    else:
        posinf_count = int(np.isposinf(value).sum())
        neginf_count = int(np.isneginf(value).sum())
    return (
        f"{name}: shape={value.shape}, dtype={value.dtype}, "
        f"bad={int(bad.sum())}, nan={int(np.isnan(value).sum())}, "
        f"+inf={posinf_count}, -inf={neginf_count}, "
        f"first_bad={coordinates}, finite_range=({finite_range})"
    )


def assert_finite_arrays(stage, named_arrays, context=None):
    bad_items = []
    for name, array in named_arrays:
        value = np.asarray(array)
        if value.dtype.kind not in "fc":
            continue
        if not bool(np.isfinite(value).all()):
            bad_items.append((str(name), value))
    if not bad_items:
        return

    detail_limit = 8
    details = "; ".join(
        _array_failure_detail(name, value)
        for name, value in bad_items[:detail_limit]
    )
    if len(bad_items) > detail_limit:
        details += f"; ... {len(bad_items) - detail_limit} additional bad arrays omitted"
    raise NumericalIntegrityError(
        f"[{stage}] non-finite array detected{_context_suffix(context)} | {details}"
    )
