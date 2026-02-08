import sys, os, runpy, threading, trace, importlib.util, types, inspect
from pathlib import Path

# Optional imports for summaries
try:
    import numpy as _np
except Exception:
    _np = None
try:
    import pandas as _pd
except Exception:
    _pd = None
try:
    import scipy.sparse as _sp
except Exception:
    _sp = None
try:
    import torch as _torch
except Exception:
    _torch = None
try:
    import tensorflow as _tf
except Exception:
    _tf = None

import collections as _collections
import collections.abc as _abc

def _safe_str(x, n=30):
    try:
        s = str(x)
    except Exception:
        s = "<unstr>"
    return s[:n]

def _is_numeric_np(arr):
    try:
        return _np is not None and _np.issubdtype(arr.dtype, _np.number)
    except Exception:
        return False

def _is_model(value) -> bool:
    try:
        if _torch is not None:
            import torch.nn as _nn  # local to avoid hard dep if torch missing
            if isinstance(value, _nn.Module):
                return True
    except Exception:
        pass
    try:
        if _tf is not None and hasattr(_tf, "keras"):
            Model = getattr(_tf.keras, "Model", None)
            if Model is not None and isinstance(value, Model):
                return True
    except Exception:
        pass
    return False

def _summarize_model(value) -> str:
    parts = []
    try:
        cls = type(value).__name__
        parts.append(f"type={cls}")
        # framework + mode
        if _is_model(value) and _torch is not None:
            try:
                import torch.nn as _nn  # noqa
                if isinstance(value, _nn.Module):
                    parts.append("framework=torch")
                    parts.append(f"mode={'train' if getattr(value, 'training', False) else 'eval'}")
                    # layers
                    try:
                        layers = max(0, len(list(value.modules())) - 1)
                        parts.append(f"layers={layers}")
                    except Exception:
                        pass
                    # params
                    try:
                        total = 0
                        trainable = 0
                        for p in value.parameters(recurse=True):
                            n = int(p.numel())
                            total += n
                            if bool(getattr(p, "requires_grad", False)):
                                trainable += n
                        parts.append(f"params={total} trainable={trainable} non_trainable={total-trainable}")
                    except Exception:
                        pass
                    # children sample
                    try:
                        kids = [type(m).__name__ for m in list(value.children())[:5]]
                        if kids:
                            parts.append(f"children_sample={kids}{'...' if len(list(value.children()))>5 else ''}")
                    except Exception:
                        pass
                    return "; ".join(parts)
            except Exception:
                pass
        if _is_model(value) and _tf is not None and hasattr(_tf, "keras"):
            try:
                Model = getattr(_tf.keras, "Model", None)
                if Model is not None and isinstance(value, Model):
                    parts.append("framework=tf.keras")
                    # mode best-effort
                    mode = getattr(value, "training", None)
                    if mode is not None:
                        parts.append(f"mode={'train' if bool(mode) else 'eval'}")
                    # layers
                    try:
                        parts.append(f"layers={len(getattr(value, 'layers', []) or [])}")
                    except Exception:
                        pass
                    # params
                    def _shape_prod(shape) -> int:
                        try:
                            dims = list(getattr(shape, "as_list", lambda: list(shape))())
                        except Exception:
                            try:
                                dims = list(shape)
                            except Exception:
                                return 0
                        prod = 1
                        for d in dims:
                            if d is None:
                                return 0
                            prod *= int(d)
                        return int(prod)
                    try:
                        all_w = getattr(value, "weights", []) or []
                        tr_w = getattr(value, "trainable_weights", []) or []
                        ntr_w = getattr(value, "non_trainable_weights", []) or []
                        total = sum(_shape_prod(w.shape) for w in all_w)
                        trainable = sum(_shape_prod(w.shape) for w in tr_w)
                        non_trainable = sum(_shape_prod(w.shape) for w in ntr_w)
                        if total or trainable or non_trainable:
                            parts.append(f"params={total} trainable={trainable} non_trainable={non_trainable}")
                    except Exception:
                        pass
                    # children sample
                    try:
                        kids = [type(l).__name__ for l in (getattr(value, "layers", []) or [])[:5]]
                        if kids:
                            parts.append(f"children_sample={kids}{'...' if len(getattr(value, 'layers', []))>5 else ''}")
                    except Exception:
                        pass
                    return "; ".join(parts)
            except Exception:
                pass
    except Exception:
        pass
    return "; ".join(parts) if parts else "type=Model"

def _is_heavy(value) -> bool:
    """
    Heuristics to decide if a value deserves a structured summary even when not truncated.
    """
    try:
        # pandas heavy types
        if (_pd is not None and isinstance(value, (_pd.DataFrame, _pd.Series))):
            return True
        # numpy arrays
        if (_np is not None and isinstance(value, _np.ndarray)):
            return value.size >= 50  # small arrays can rely on repr
        # torch / tf tensors
        if (_torch is not None and isinstance(value, _torch.Tensor)):
            return value.numel() >= 50
        if (_tf is not None and isinstance(value, getattr(_tf, "Tensor", ()))):
            try:
                return int(getattr(value, "shape", (0,))[-1] or 0) >= 50 or True  # tensors generally heavy
            except Exception:
                return True
        # scipy sparse
        if (_sp is not None and isinstance(value, _sp.spmatrix)):
            return True
        # containers that can explode
        if isinstance(value, (list, tuple)):
            return len(value) >= 50
        if isinstance(value, dict):
            return len(value) >= 50
        # also treat models as heavy
        if _is_model(value):
            return True
    except Exception:
        pass
    return False

def _summarize(value):
    tname = type(value).__name__
    parts = [f"type={tname}"]

    # 4️⃣ torch.Tensor / tensorflow.Tensor
    try:
        if _torch is not None and isinstance(value, _torch.Tensor):
            shape = tuple(value.shape)
            dtype = str(value.dtype)
            try:
                device = str(value.device)
            except Exception:
                device = "unknown"
            parts.append(f"shape={shape}")
            parts.append(f"dtype={dtype}")
            parts.append(f"device={device}")
            # numeric stats (best-effort, CPU sample)
            try:
                numel = value.numel()
                if numel > 0 and value.dtype is not None:
                    t = value.detach()
                    # sample to avoid heavy ops
                    if numel > 100000:
                        t = t.flatten()[:100000]
                    t = t.to("cpu")
                    if t.is_floating_point() or t.dtype in (_torch.int8, _torch.int16, _torch.int32, _torch.int64, _torch.uint8):
                        tf = t.float()
                        mn = float(tf.min().item())
                        mx = float(tf.max().item())
                        mean = float(tf.mean().item())
                        parts.append(f"min={mn:.6g} max={mx:.6g} mean={mean:.6g}")
            except Exception:
                pass
            return "; ".join(parts)
        if _tf is not None and isinstance(value, getattr(_tf, "Tensor", ())):
            try:
                shape = tuple(getattr(value, "shape", ()))
            except Exception:
                shape = ()
            try:
                dtype = str(getattr(value, "dtype", "unknown"))
            except Exception:
                dtype = "unknown"
            device = _safe_str(getattr(value, "device", "unknown"))
            parts.append(f"shape={shape}")
            parts.append(f"dtype={dtype}")
            parts.append(f"device={device}")
            try:
                arr = value.numpy()  # may fail for symbolic tensors
                if arr is not None and _np is not None and _np.size(arr) > 0 and _np.issubdtype(arr.dtype, _np.number):
                    flat = _np.ravel(arr).astype(float, copy=False)
                    mn = _np.nanmin(flat); mx = _np.nanmax(flat); mean = _np.nanmean(flat)
                    parts.append(f"min={mn:.6g} max={mx:.6g} mean={mean:.6g}")
            except Exception:
                pass
            return "; ".join(parts)
    except Exception:
        pass

    # 2️⃣ numpy.ndarray
    if _np is not None and isinstance(value, _np.ndarray):
        try:
            shape = tuple(value.shape)
            parts.append(f"shape={shape}")
            parts.append(f"dtype={value.dtype}")
            parts.append(f"size={value.size}")
            if len(shape) > 2:
                parts.append(f"shape_hint=({shape[0]}, {shape[1]}, ...)")
            if value.dtype == _np.dtype("object"):
                # sample types of first few elements
                try:
                    flat = _np.ravel(value)
                    k = min(10, flat.size)
                    typ_sample = [_safe_str(type(flat[i]).__name__) for i in range(k)]
                    parts.append(f"obj_types_sample={typ_sample}{'...' if flat.size>k else ''}")
                except Exception:
                    pass
            if _np.isfinite(0):  # cheap call to ensure numpy available
                try:
                    has_nan = bool(_np.isnan(value).any())
                    has_inf = bool(_np.isinf(value).any())
                    if has_nan: parts.append("has_nan=True")
                    if has_inf: parts.append("has_inf=True")
                except Exception:
                    pass
            if _is_numeric_np(value):
                try:
                    flat = _np.ravel(value).astype(float, copy=False)
                    mn = _np.nanmin(flat); mx = _np.nanmax(flat); mean = _np.nanmean(flat)
                    parts.append(f"min={mn:.6g} max={mx:.6g} mean={mean:.6g}")
                except Exception:
                    pass
        except Exception:
            pass
        return "; ".join(parts)

    # 3️⃣ pandas.DataFrame
    if _pd is not None and isinstance(value, _pd.DataFrame):
        try:
            parts.append(f"shape={value.shape}")
            try:
                # columns and dtypes
                dtypes = {str(c): str(value[c].dtype) for c in value.columns}
                parts.append(f"cols={list(value.columns)}")
                parts.append(f"dtypes={dtypes}")
            except Exception:
                pass
            # missing info (overall)
            try:
                total = int(value.shape[0] * value.shape[1])
                if total > 0:
                    nmiss = int(value.isna().sum().sum())
                    if nmiss > 0:
                        rate = 100.0 * nmiss / total
                        parts.append(f"missing={nmiss} (≈{rate:.2f}%)")
            except Exception:
                pass
            # index info
            try:
                idx = value.index
                if not isinstance(idx, _pd.RangeIndex):
                    parts.append(f"index={type(idx).__name__}(len={len(idx)})")
            except Exception:
                pass
        except Exception:
            pass
        return "; ".join(parts)

    # 4️⃣ pandas.Series
    if _pd is not None and isinstance(value, _pd.Series):
        try:
            parts.append(f"len={len(value)}")
            parts.append(f"dtype={value.dtype}")
            if value.name is not None:
                parts.append(f"name={value.name}")
            # missing
            try:
                nmiss = int(value.isna().sum())
                if nmiss > 0:
                    rate = 100.0 * nmiss / max(1, len(value))
                    parts.append(f"missing={nmiss} (≈{rate:.2f}%)")
            except Exception:
                pass
            # numeric stats
            try:
                if _np is not None and _np.issubdtype(value.dtype, _np.number) and len(value) > 0:
                    v = _pd.to_numeric(value, errors="coerce")
                    mn = _np.nanmin(v.values.astype(float, copy=False))
                    mx = _np.nanmax(v.values.astype(float, copy=False))
                    mean = _np.nanmean(v.values.astype(float, copy=False))
                    parts.append(f"min={mn:.6g} max={mx:.6g} mean={mean:.6g}")
            except Exception:
                pass
            # category info
            try:
                if str(value.dtype) == "category":
                    parts.append(f"n_categories={len(value.cat.categories)}")
            except Exception:
                pass
        except Exception:
            pass
        return "; ".join(parts)

    # 5️⃣ scipy.sparse 矩阵
    if _sp is not None and isinstance(value, _sp.spmatrix):
        try:
            parts.append(f"shape={value.shape}")
            parts.append(f"nnz={value.nnz}")
            try:
                parts.append(f"format={value.getformat()}")
            except Exception:
                pass
            try:
                density = float(value.nnz) / max(1, (value.shape[0] * value.shape[1]))
                parts.append(f"density={density:.6g}")
            except Exception:
                pass
            try:
                if value.nnz > 0 and _np is not None and _np.issubdtype(value.dtype, _np.number):
                    data = value.data
                    # sample up to 100k
                    if data.size > 100000:
                        data = data[:100000]
                    mn = _np.nanmin(data.astype(float, copy=False))
                    mx = _np.nanmax(data.astype(float, copy=False))
                    mean = _np.nanmean(data.astype(float, copy=False))
                    parts.append(f"min={mn:.6g} max={mx:.6g} mean={mean:.6g}")
            except Exception:
                pass
        except Exception:
            pass
        return "; ".join(parts)

    # 6️⃣ list / tuple
    if isinstance(value, (list, tuple)):
        try:
            n = len(value)
            parts.append(f"len={n}")
            if n == 0:
                parts.append("empty=True")
            else:
                # element types sample
                sample_n = min(5, n)
                def _etype(v):
                    # containers -> just type tag
                    if isinstance(v, (list, tuple, dict)) or (_np is not None and isinstance(v, _np.ndarray)) or (_sp is not None and isinstance(v, _sp.spmatrix)):
                        return type(v).__name__
                    return type(v).__name__
                sample = [_etype(value[i]) for i in range(sample_n)]
                parts.append(f"types_sample={sample}{'...' if n>5 else ''}")
                # homogeneous detection
                try:
                    tset = {type(x).__name__ for x in value}
                    if len(tset) == 1:
                        parts.append(f"homogeneous={next(iter(tset))}")
                except Exception:
                    pass
        except Exception:
            pass
        return "; ".join(parts)

    # 7️⃣ dict
    if isinstance(value, dict):
        try:
            n = len(value)
            parts.append(f"len={n}")
            if n == 0:
                parts.append("empty=True")
            else:
                keys = list(value.keys())
                ksample = keys[:5]
                parts.append(f"keys_sample={[ _safe_str(k, 30) for k in ksample]}{'...' if n>5 else ''}")
                try:
                    ktypes = {type(k).__name__ for k in keys}
                    if len(ktypes) == 1:
                        parts.append(f"key_type={next(iter(ktypes))}")
                except Exception:
                    pass
                try:
                    vtypes = {type(value[k]).__name__ for k in ksample}
                    if len(vtypes) == 1 and n <= 5:
                        parts.append(f"value_type={next(iter(vtypes))}")
                except Exception:
                    pass
        except Exception:
            pass
        return "; ".join(parts)

    # 5️⃣ collections: Counter / defaultdict / OrderedDict
    try:
        if isinstance(value, _collections.Counter):
            parts.append(f"len={len(value)}")
            try:
                top5 = value.most_common(5)
                parts.append(f"top5={[(k, v) for k, v in top5]}")
            except Exception:
                pass
            return "; ".join(parts)
        if isinstance(value, _collections.defaultdict):
            parts.append(f"default_factory={getattr(value.default_factory, '__name__', str(value.default_factory))}")
            parts.append(f"len={len(value)}")
            return "; ".join(parts)
        if isinstance(value, _collections.OrderedDict):
            parts.append(f"len={len(value)}")
            try:
                keys = list(value.keys())
                parts.append(f"keys_sample={[ _safe_str(k, 30) for k in keys[:5]]}{'...' if len(keys)>5 else ''}")
            except Exception:
                pass
            return "; ".join(parts)
    except Exception:
        pass

    # 8️⃣ 其它对象
    try:
        if hasattr(value, "shape"):
            try:
                parts.append(f"shape={getattr(value, 'shape', None)}")
            except Exception:
                pass
        if hasattr(value, "__len__"):
            try:
                parts.append(f"len={len(value)}")
            except Exception:
                pass
        try:
            cn = value.__class__.__name__
            parts.append(f"type={cn}")
        except Exception:
            pass
        # iterators / generators
        try:
            if inspect.isgenerator(value) or isinstance(value, _abc.Iterator):
                parts.append("note=iterator/generator (not materialized)")
                return "; ".join(parts)
        except Exception:
            pass
        # custom instance: sample scalar attrs
        try:
            attrs = {}
            vdict = getattr(value, "__dict__", None)
            if isinstance(vdict, dict):
                for k, v in list(vdict.items())[:5]:
                    if isinstance(v, (int, float, str, bool)):
                        attrs[k] = v if isinstance(v, (int, float, bool)) else _safe_str(v, 40)
            if attrs:
                parts.append(f"attrs_sample={attrs}")
        except Exception:
            pass
    except Exception:
        pass
    return "; ".join(parts)

class VarTracer(trace.Trace):
    def __init__(self, log_path, target_file):
        super().__init__(trace=False, count=False)
        self.log = open(log_path, "w", buffering=1, encoding="utf-8")
        self.target = os.path.abspath(target_file)
        self.prev = {}
        try:
            self.max_repr = int(os.environ.get("TRACE_MAX_REPR", "4000"))
        except Exception:
            self.max_repr = 4000
        self.only_changes = str(os.environ.get("TRACE_ONLY_CHANGES", "1")).lower() in ("1", "true", "yes")
        try:
            self.sample_every = max(1, int(os.environ.get("TRACE_SAMPLE_EVERY", "1")))
        except Exception:
            self.sample_every = 1
        try:
            self.max_per_line = int(os.environ.get("TRACE_MAX_PER_LINE", "0"))  # 0 = unlimited
        except Exception:
            self.max_per_line = 0
        self._line_seen = {}
        self._line_emits = {}
        # Loop-aware sampling state per frame-id
        # state: { 'start_ln': int, 'last_ln': int, 'iter': int, 'emit_iter': bool, 'miss_since_anchor': int }
        self._loops = {}
        # summary policy
        self._summary_always = str(os.environ.get("TRACE_SUMMARY_ALWAYS", "1")).lower() in ("1", "true", "yes")
        # control whether to keep full repr for models
        self._model_repr_full = str(os.environ.get("TRACE_MODEL_REPR", "0")).lower() in ("1", "true", "yes")
        # NEW: call-site grouped sampling across frames
        # _groups: key -> {'iter': int}
        # _frame_emit: frame_id -> bool (whether to emit this whole call frame)
        self._groups = {}
        self._frame_emit = {}

    def globaltrace(self, frame, event, arg):
        return self.localtrace

    def _should_skip(self, name, value) -> bool:
        if (name.startswith("__") and name.endswith("__")) or name.startswith("_") or name.isupper():
            return True
        if name in ("self", "cls") or name.startswith("."):
            return True
        if isinstance(value, types.ModuleType):
            return True
        if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value) or inspect.isroutine(value):
            return True
        return False

    def _loop_should_emit(self, frame, ln) -> bool:
        """
        Loop-aware sampling:
        - Detect iteration boundaries when execution jumps back to the loop anchor (start_ln).
        - If total iterations <= 5: emit all iterations.
        - If > 5: emit only ~5 evenly spaced iterations (quantile targets based on max seen so far).
        Applies to all lines within the same iteration.
        """
        fid = id(frame)
        st = self._loops.get(fid)
        if st is None:
            # Initialize a tentative loop anchor at the first seen line in this frame
            st = {'start_ln': ln, 'last_ln': ln, 'iter': 1, 'emit_iter': True, 'miss_since_anchor': 0}
            self._loops[fid] = st
            return True
        # Heuristic: new iteration when we return to anchor (start_ln) from a higher line
        new_iter = (st['last_ln'] > ln and ln == st['start_ln'])
        # If we have moved far away from anchor for long, reset (loop likely ended)
        if ln != st['start_ln']:
            st['miss_since_anchor'] = st.get('miss_since_anchor', 0) + 1
            if st['miss_since_anchor'] > 1000:
                # reset loop state
                st['start_ln'] = ln
                st['last_ln'] = ln
                st['iter'] = 1
                st['emit_iter'] = True
                st['miss_since_anchor'] = 0
                return True
        else:
            st['miss_since_anchor'] = 0

        if new_iter:
            st['iter'] += 1
            # decide emission for this iteration
            it = st['iter']
            if it <= 5:
                st['emit_iter'] = True
            else:
                # compute ~5 evenly spaced targets given max iter seen so far
                max_it = it
                targets = set(max(1, int(round(k * max_it / 5.0))) for k in range(1, 6))
                st['emit_iter'] = (it in targets)
        # update last line
        st['last_ln'] = ln
        return st['emit_iter']

    def _group_key(self, frame):
        """
        Identify a 'loop group' across separate frames by call-site.
        Key = (callee_file, callee_func, caller_file, caller_lineno)
        """
        try:
            callee_file = os.path.abspath(frame.f_code.co_filename)
            callee_func = frame.f_code.co_name
        except Exception:
            callee_file = frame.f_code.co_filename
            callee_func = frame.f_code.co_name
        caller = frame.f_back
        try:
            caller_file = os.path.abspath(caller.f_code.co_filename) if caller else None
            caller_line = caller.f_lineno if caller else None
        except Exception:
            caller_file = getattr(caller.f_code, "co_filename", None) if caller else None
            caller_line = getattr(caller, "f_lineno", None) if caller else None
        return (callee_file, callee_func, caller_file, caller_line)

    def _group_should_emit(self, it: int) -> bool:
        """
        Iteration policy for grouped calls:
        - <=5: emit all
        - >5: emit only ~5 evenly spaced iterations via dynamic quantile targets
        """
        if it <= 5:
            return True
        targets = set(max(1, int(round(k * it / 5.0))) for k in range(1, 6))
        return it in targets

    def localtrace(self, frame, event, arg):
        # NEW: decide per-call emission on 'call' and cleanup on 'return'
        if event == "call":
            try:
                gk = self._group_key(frame)
                g = self._groups.get(gk, {"iter": 0})
                g["iter"] += 1
                self._groups[gk] = g
                emit = self._group_should_emit(g["iter"])
                self._frame_emit[id(frame)] = emit
            except Exception:
                # default to emit to avoid hiding data on failures
                self._frame_emit[id(frame)] = True
            return self.localtrace
        if event == "return":
            # cleanup frame-scoped decisions and loop state
            try:
                self._frame_emit.pop(id(frame), None)
            except Exception:
                pass
            try:
                self._loops.pop(id(frame), None)
            except Exception:
                pass
            return self.localtrace

        if event == "line":
            try:
                afn = os.path.abspath(frame.f_code.co_filename)
            except Exception:
                afn = frame.f_code.co_filename
            if afn == self.target:
                ln = frame.f_lineno

                # Gate by per-call grouped decision (if present)
                emit_call = self._frame_emit.get(id(frame), True)
                if not emit_call:
                    return self.localtrace

                # Loop-aware sampling gate inside the same frame (classic Python for/while)
                if not self._loop_should_emit(frame, ln):
                    return self.localtrace

                # Per-line sampling and caps
                hits = self._line_seen.get(ln, 0) + 1
                self._line_seen[ln] = hits
                if self.sample_every > 1 and (hits % self.sample_every) != 0:
                    return self.localtrace
                if self.max_per_line > 0 and self._line_emits.get(ln, 0) >= self.max_per_line:
                    return self.localtrace

                current = {}
                for k, v in frame.f_locals.items():
                    if self._should_skip(k, v):
                        continue
                    try:
                        if _is_model(v) and not getattr(self, "_model_repr_full", False):
                            s = f"<{type(v).__name__}>"
                        else:
                            s = repr(v)
                    except Exception:
                        s = "<unrepr>"
                    # attach summary if truncated
                    if isinstance(s, str) and len(s) > self.max_repr:
                        try:
                            s = s[:self.max_repr] + f"...(truncated {len(s)} chars) | " + _summarize(v)
                        except Exception:
                            s = s[:self.max_repr] + "...(truncated)"
                    else:
                        # optionally attach summary for heavy objects even when not truncated
                        try:
                            if self._summary_always and (_is_heavy(v) or _is_model(v)):
                                s = s + " | " + _summarize(v)
                        except Exception:
                            pass
                    current[k] = s

                out = {}
                changed = False
                for k, s in current.items():
                    prev_s = self.prev.get(k)
                    if prev_s == s:
                        out[k] = "..."
                    else:
                        out[k] = s
                        changed = True

                if self.only_changes and not changed:
                    return self.localtrace

                try:
                    self.log.write(f"line {ln} {out}\n")
                    self._line_emits[ln] = self._line_emits.get(ln, 0) + 1
                except Exception:
                    pass

                self.prev = current
        return self.localtrace

def has_main_guard(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
        return "__name__" in s and "__main__" in s
    except Exception:
        return False

def exec_as_module(script_path: str, argv0: str):
    # Execute with __name__ = module name (non-__main__)
    module_name = Path(script_path).stem
    g = {
        "__name__": module_name,
        "__file__": script_path,
        "__package__": None,
        "__cached__": None,
        "__doc__": None,
    }
    with open(script_path, "rb") as f:
        code = compile(f.read(), script_path, "exec")
    sys.argv[0] = argv0
    exec(code, g, g)

def main():
    if len(sys.argv) < 2:
        print("Usage: __trace_bootstrap__.py <script> [args...]", file=sys.stderr)
        sys.exit(2)
    script = sys.argv[1]
    args = sys.argv[2:]
    # Preserve argv[0] as path-like for user code
    sys.argv = [script] + args
    log_path = os.environ.get("TRACE_LOG", "/app/submission/trace.log")
    target_file = os.path.abspath(script)
    tracer = VarTracer(log_path, target_file)
    sys.settrace(tracer.globaltrace)
    threading.settrace(tracer.globaltrace)
    try:
        sp = os.path.abspath(script)
        if has_main_guard(sp):
            runpy.run_path(script, run_name="__main__")
        else:
            exec_as_module(sp, argv0=script)
    finally:
        sys.settrace(None)
        try:
            tracer.log.flush()
            tracer.log.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()