"""PS-09 optimised encoder for BAAI/bge-base-en-v1.5.

Contract (bench.py): setup() is called once and is NOT timed; encode(texts) -> np.ndarray [N, 768], L2-normalised,
in input order. The same encode() is called with the whole passage list (offline) and with one query (live).

Same weights, same architecture, same maths as reference.py. What changes is HOW it runs:
  O1  FP16 on Tensor Cores, fused QKV projection, PyTorch SDPA attention, no pooler
  O2  length-aware scheduler: sort by token length, token-budget batches, pad each batch to a multiple of 8
  O2b batches also end where the padded length drops to a lower multiple of 8 (less padding inside big batches)
  O3  host pipeline: tokenise + build batches in a background thread, pinned memory, async copies, one final sync
  O4  optional torch.compile of the layer stack
  O5  CLS-only last layer (keys/values for all tokens, everything else for the CLS row only)
  O7  live path: one query -> fixed length bucket -> CUDA Graph replay
  P1  precision: FP16 GEMMs/attention with FP32 GEMM reductions; optional FP32 residual stream + LayerNorm (DEFAULT_RES32)
  P2  short inputs (<= SHORT_FP32 tokens) run in FP32 with reference maths, both offline and live. On the 100k
      practice set every FP16 result below cosine 0.9999 was <= 23 tokens long (worst 0.9944 at 16 tokens); above 32
      tokens plain FP16 is >= 0.99999. Short inputs are 33% of passages but only 5.5% of tokens, so FP32 there is cheap.

Every optimisation can be switched off with environment variables, so each phase can be measured on its own
(see run_ablation.py). PS09_STAGE picks a preset; individual PS09_<FLAG> variables override it.

The organisers run this file WITHOUT any PS09_* variables, so the defaults below are the submitted configuration.
Hardware is detected at run time (no GPU model is assumed): FP16 needs CUDA, CUDA Graphs / torch.compile / the
warm-up fall back to the plain path if they fail, and the token budget is halved automatically if warm-up runs out of
memory on a smaller GPU.
"""
import os
import queue
import threading

import numpy as np
import torch
import torch.nn.functional as F

# ---- submitted defaults (scripts/apply_best_config.py --apply edits the first two) ----
DEFAULT_STAGE = 4
DEFAULT_TOKEN_BUDGET = 8192
DEFAULT_RES32 = 0   # FP32 residual stream + LayerNorm on top of FP16 GEMMs (12% slower; superseded by P2, see TRACKER.md)
DEFAULT_SHORT_FP32 = 32   # P2: inputs of at most this many tokens run in FP32 (0 = off)

MODEL = os.environ.get("PS09_MODEL", "BAAI/bge-base-en-v1.5")
MAX_LEN = 512

# Stage presets for the ablation study. Stage 0 behaves like reference.py (FP32, pad to 512, batch 32).
# bucket = O2b: also end a batch where the padded length drops to a lower multiple of 8 (once the batch holds at least
# MIN_FILL x budget tokens). Stage 4 padding 10.0% -> 3.7% on the 100k practice set (scripts/workload_stats.py).
STAGES = {
    0: dict(fp16=0, sort=0, budget=0, bucket=0, overlap=0, cls_only=0, compile=0, graph=0),
    1: dict(fp16=1, sort=0, budget=0, bucket=0, overlap=0, cls_only=0, compile=0, graph=0),   # O1
    2: dict(fp16=1, sort=1, budget=1, bucket=1, overlap=0, cls_only=0, compile=0, graph=0),   # + O2
    3: dict(fp16=1, sort=1, budget=1, bucket=1, overlap=1, cls_only=0, compile=0, graph=1),   # + O3, O7
    4: dict(fp16=1, sort=1, budget=1, bucket=1, overlap=1, cls_only=1, compile=0, graph=1),   # + O5
    5: dict(fp16=1, sort=1, budget=1, bucket=1, overlap=1, cls_only=1, compile=1, graph=1),   # + O4
}
STAGE = int(os.environ.get("PS09_STAGE", DEFAULT_STAGE))
CFG = dict(STAGES[STAGE])
for _k in list(CFG):
    if f"PS09_{_k.upper()}" in os.environ:
        CFG[_k] = int(os.environ[f"PS09_{_k.upper()}"])
TOKEN_BUDGET = int(os.environ.get("PS09_TOKEN_BUDGET", DEFAULT_TOKEN_BUDGET))  # max (batch x padded length) per batch
# Experimental precision switches (default off; measured with validate.py before any use):
BF16 = int(os.environ.get("PS09_BF16", "0"))              # BF16 instead of FP16 (if the GPU supports it)
RES32 = int(os.environ.get("PS09_RES32", DEFAULT_RES32))  # FP32 residual stream + LayerNorm (precision margin)
SHORT_FP32 = int(os.environ.get("PS09_SHORT_FP32", DEFAULT_SHORT_FP32))  # P2 cut-off in tokens (0 = off)
FP16_ACCUM = int(os.environ.get("PS09_FP16_ACCUM", "0"))  # FP16 accumulation in GEMMs (2x on GeForce, less precise)
FIXED_BATCH = 32                                                    # used when budget batching is off
CHUNK = int(os.environ.get("PS09_CHUNK", "16384"))                  # texts tokenised per producer step
MIN_FILL = float(os.environ.get("PS09_MIN_FILL", "0.25"))           # bucket split only once batch >= this x budget
LIVE_BUCKETS = (32, 64, 128)                                        # CUDA Graph lengths for single queries
DEBUG = int(os.environ.get("PS09_DEBUG", "0"))                      # 1 = record max |activation| per layer

_S = {}  # module state filled by setup()
_LOCK = threading.Lock()  # encode() is not re-entrant (shared CUDA Graph buffers); bench.py calls it serially


# ----------------------------------------------------------------------------------------------- model
class Encoder:
    """BERT forward written directly on the checkpoint's weights (so it is independent of the
    transformers version and lets us fuse QKV, choose the attention kernel and trim the last layer)."""

    def __init__(self, hf_model, dtype, device, res32=False):
        self.dtype, self.device = dtype, device
        # res32: GEMMs and attention in FP16/BF16, but the residual stream and LayerNorms in FP32 (precision fallback;
        # scripts/precision_study.py: worst-case 1-cos about 3.7x smaller in CPU emulation)
        self.res32 = bool(res32) and dtype != torch.float32
        sdt = torch.float32 if self.res32 else dtype   # dtype of the residual stream and LayerNorm parameters
        m = hf_model
        e = m.embeddings
        self.word = e.word_embeddings.weight.detach().to(device, dtype)
        self.pos = e.position_embeddings.weight.detach().to(device, dtype)
        self.typ0 = e.token_type_embeddings.weight[0].detach().to(device, dtype)   # all token_type_ids are 0
        self.eln = (e.LayerNorm.weight.detach().to(device, sdt), e.LayerNorm.bias.detach().to(device, sdt))
        self.eps = float(m.config.layer_norm_eps)
        self.heads = m.config.num_attention_heads
        self.layers = []
        for L in m.encoder.layer:
            a, ao = L.attention.self, L.attention.output
            g = lambda t: t.detach().to(device, dtype).contiguous()
            gs = lambda t: t.detach().to(device, sdt).contiguous()
            self.layers.append(dict(
                wqkv=g(torch.cat([a.query.weight, a.key.weight, a.value.weight], 0)),
                bqkv=g(torch.cat([a.query.bias, a.key.bias, a.value.bias], 0)),
                wo=g(ao.dense.weight), bo=g(ao.dense.bias),
                ln1=(gs(ao.LayerNorm.weight), gs(ao.LayerNorm.bias)),
                wi=g(L.intermediate.dense.weight), bi=g(L.intermediate.dense.bias),
                wf=g(L.output.dense.weight), bf=g(L.output.dense.bias),
                ln2=(gs(L.output.LayerNorm.weight), gs(L.output.LayerNorm.bias)),
            ))
        self.d = self.word.shape[1]
        self.dh = self.d // self.heads
        self.maxabs = []

    def _ln(self, x, p):
        return F.layer_norm(x, (self.d,), p[0], p[1], self.eps)  # CUDA LN accumulates in FP32

    def _layer(self, P, h, mask, cls_only):
        B, T, D = h.shape
        H, dh, dt = self.heads, self.dh, self.dtype
        hc = h.to(dt)  # GEMM input (no-op unless res32)
        if not cls_only:
            qkv = F.linear(hc, P["wqkv"], P["bqkv"]).view(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            res = h
        else:  # keys/values need every token; the query and everything after it only the CLS row
            kv = F.linear(hc, P["wqkv"][D:], P["bqkv"][D:]).view(B, T, 2, H, dh).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            res = h[:, :1]
            q = F.linear(hc[:, :1], P["wqkv"][:D], P["bqkv"][:D]).view(B, 1, H, dh).transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        ctx = ctx.transpose(1, 2).reshape(B, -1, D)
        x = self._ln(F.linear(ctx, P["wo"], P["bo"]) + res, P["ln1"])   # FP16 + FP32 -> FP32 when res32
        y = F.linear(F.gelu(F.linear(x.to(dt), P["wi"], P["bi"])), P["wf"], P["bf"])  # exact (erf) GELU, as BERT
        return self._ln(y + x, P["ln2"])

    def forward(self, ids, attn, use_mask=True, cls_only=False):
        """ids, attn: [B, T] int64 on device. Returns L2-normalised CLS embeddings [B, 768] in FP32."""
        T = ids.shape[1]
        if self.res32:
            h = self.word[ids].float() + self.pos[:T].float() + self.typ0.float()
        else:
            h = self.word[ids] + self.pos[:T] + self.typ0
        h = self._ln(h, self.eln)
        mask = None
        if use_mask:  # additive mask, broadcast over heads and query rows
            mask = ((1 - attn[:, None, None, :].to(self.dtype)) * torch.finfo(self.dtype).min)
        n = len(self.layers)
        for i, P in enumerate(self.layers):
            h = self._layer(P, h, mask, cls_only and i == n - 1)
            if DEBUG and not (h.is_cuda and torch.cuda.is_current_stream_capturing()):
                self.maxabs.append(float(h.abs().max()))
        cls = h[:, 0].float()                    # pooling + normalisation in FP32
        return F.normalize(cls, dim=-1)


# ----------------------------------------------------------------------------------------------- setup
def setup():
    if _S:  # idempotent: bench.py calls setup(), encode() calls it only if nobody did
        return
    from transformers import AutoModel, AutoTokenizer
    cuda = torch.cuda.is_available()
    notes = []
    if cuda:
        m = torch.backends.cuda.matmul
        m.allow_tf32 = False   # keep FP32 stages bit-comparable with the reference
        # FP32 reductions inside FP16/BF16 GEMMs (cuBLAS split-K may otherwise reduce in half precision)
        strict = int(os.environ.get("PS09_STRICT_REDUCTION", "1"))
        m.allow_fp16_reduced_precision_reduction = not strict
        if hasattr(m, "allow_bf16_reduced_precision_reduction"):
            m.allow_bf16_reduced_precision_reduction = not strict
        if FP16_ACCUM:
            if hasattr(m, "allow_fp16_accumulation"):
                m.allow_fp16_accumulation = True
            else:
                notes.append("PS09_FP16_ACCUM ignored: needs PyTorch >= 2.7")
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32
    if CFG["fp16"] and cuda:
        dtype = torch.float16
        if BF16:
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                notes.append("PS09_BF16 ignored: GPU has no BF16 support, using FP16")
    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModel.from_pretrained(MODEL).float().eval()
    enc = Encoder(hf, dtype, device, res32=RES32)
    # P2: FP32 copy of the same weights for short inputs (only when the main path is half precision)
    enc32 = Encoder(hf, torch.float32, device) if dtype != torch.float32 and SHORT_FP32 > 0 else None
    del hf
    fast = tok.backend_tokenizer          # Rust tokenizer: batch encoding releases the GIL
    fast.enable_truncation(MAX_LEN)       # max_length counts [CLS]/[SEP], same as the reference call
    fast.no_padding()
    torch.set_grad_enabled(False)
    _S.update(tok=tok, fast=fast, enc=enc, device=device, dtype=dtype, cuda=cuda, notes=notes,
              pad_id=tok.pad_token_id or 0, fwd=enc.forward, graphs={}, compiled=False,
              enc32=enc32, fwd32=enc32.forward if enc32 else None)

    if cuda:
        if CFG["compile"]:
            try:
                _S["fwd"] = torch.compile(enc.forward, dynamic=True)
                _S["compiled"] = True
            except Exception as ex:  # no Triton / unsupported platform -> eager
                notes.append(f"torch.compile unavailable, eager used: {ex!r}"[:300])
        _warmup()
        if CFG["graph"]:
            try:
                _capture_graphs()
            except Exception as ex:  # capture unsupported -> the eager path serves single queries
                _S["graphs"] = {}
                notes.append(f"CUDA Graph capture failed, eager live path used: {ex!r}"[:300])
                torch.cuda.synchronize()


def _warmup():
    """Kernel selection, cuBLAS handles, allocator pools, compile caches -- at the real batch shapes, largest
    first. If the GPU is smaller than expected, halve the token budget until the largest batch fits with headroom."""
    global TOKEN_BUDGET
    dev, cls = _S["device"], bool(CFG["cls_only"])
    total = torch.cuda.get_device_properties(dev).total_memory
    while True:
        try:
            torch.cuda.reset_peak_memory_stats(dev)
            with torch.inference_mode():
                for T in (512, 256, 128, 64, 32, 16, 8):
                    b = max(1, TOKEN_BUDGET // T)
                    ids = torch.full((b, T), 101, device=dev)
                    attn = torch.ones_like(ids)
                    _S["fwd"](ids, attn, True, cls)
                    if _S["compiled"]:
                        _S["fwd"](ids, attn, False, cls)  # unmasked variant used by uniform-length batches
                if _S["fwd32"] is not None:  # P2 FP32 path, only at the short shapes it serves
                    for T in (32, 24, 16, 8):
                        if T <= _pad8(SHORT_FP32):
                            ids = torch.full((max(1, TOKEN_BUDGET // T), T), 101, device=dev)
                            _S["fwd32"](ids, torch.ones_like(ids), True, cls)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(dev)
            if peak > 0.75 * total and TOKEN_BUDGET > 4096:
                raise torch.cuda.OutOfMemoryError(f"warm-up peak {peak / 2**30:.2f} GiB is over 75% of VRAM")
            _S["warmup_peak_bytes"] = int(peak)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if TOKEN_BUDGET <= 4096:
                raise
            TOKEN_BUDGET //= 2
            _S["notes"].append(f"token budget reduced to {TOKEN_BUDGET} to fit this GPU")
        except Exception as ex:
            if not _S["compiled"]:
                raise
            _S["fwd"], _S["compiled"] = _S["enc"].forward, False  # compiler failed at run time -> eager
            _S["notes"].append(f"torch.compile failed in warm-up, eager used: {ex!r}"[:300])


def runtime_info():
    """What actually runs on this machine (for logs and the report)."""
    info = {"stage": STAGE, "cfg": dict(CFG), "token_budget": TOKEN_BUDGET, "short_fp32": SHORT_FP32, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(), "model": MODEL}
    if _S:
        info.update(device=str(_S["device"]), dtype=str(_S["dtype"]).replace("torch.", ""), res32=_S["enc"].res32,
                    short_fp32_active=_S["enc32"] is not None,
                    compiled=_S["compiled"], graph_buckets=sorted(_S["graphs"]), notes=list(_S["notes"]))
        if _S["cuda"]:
            p = torch.cuda.get_device_properties(_S["device"])
            info.update(gpu=p.name, compute_capability=f"{p.major}.{p.minor}", vram_gib=round(p.total_memory / 2**30, 2),
                        cuda_runtime=torch.version.cuda, warmup_peak_gib=round(_S.get("warmup_peak_bytes", 0) / 2**30, 3),
                        strict_reduction=not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    return info


def _capture_graphs():
    """O7: one CUDA Graph per live bucket. Replay launches the whole 12-layer forward in one call.
    P2: a bucket no longer than SHORT_FP32 is captured on the FP32 encoder."""
    device = _S["device"]
    for T in LIVE_BUCKETS:
        enc = _S["enc32"] if _S["enc32"] is not None and T <= SHORT_FP32 else _S["enc"]
        ids = torch.zeros((1, T), dtype=torch.long, device=device)
        attn = torch.zeros((1, T), dtype=torch.long, device=device)
        ids[0, 0], attn[0, 0] = 101, 1
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                enc.forward(ids, attn, True, bool(CFG["cls_only"]))
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            out = enc.forward(ids, attn, True, bool(CFG["cls_only"]))
        host_ids = torch.zeros((1, T), dtype=torch.long).pin_memory()
        host_attn = torch.zeros((1, T), dtype=torch.long).pin_memory()
        _S["graphs"][T] = (g, ids, attn, out, host_ids, host_attn)
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------------------------- batching
def _pad8(n):
    return (n + 7) // 8 * 8


def _plan(lengths):
    """O2: list of index arrays. Sorted by length (longest first, so any OOM shows up at once), cut so that
    batch_size x padded_length <= TOKEN_BUDGET. Without sort/budget: input order, fixed batch of 32."""
    n = len(lengths)
    order = np.argsort(-lengths, kind="stable") if CFG["sort"] else np.arange(n)
    if not CFG["budget"]:
        return [order[i:i + FIXED_BATCH] for i in range(0, n, FIXED_BATCH)]
    bucket = CFG["sort"] and CFG.get("bucket", 0)
    # P2: the first n_long sorted rows run in half precision, the rest (<= SHORT_FP32 tokens) in FP32; never mix them
    n_long = int((lengths > SHORT_FP32).sum()) if CFG["sort"] and _short_on() else n
    if bucket:
        neg_p = -((lengths[order] + 7) // 8 * 8)   # ascending, because order is longest first
        # minimum batch size before a bucket split: MIN_FILL x budget, but never more than a quarter of this call's
        # tokens (small calls would otherwise be forced into one wide, heavily padded batch)
        fill_tokens = MIN_FILL * min(TOKEN_BUDGET, int(lengths.sum()) / 4)
    batches, i = [], 0
    while i < n:
        T = _pad8(int(lengths[order[i]])) if CFG["sort"] else MAX_LEN
        b = max(1, TOKEN_BUDGET // T)
        end = min(n, i + b)
        if bucket:  # first row whose padded length is below T, but keep at least MIN_FILL of the budget
            k = int(np.searchsorted(neg_p, -T, side="right"))
            end = min(end, max(k, i + max(1, int(fill_tokens // T))))
        if i < n_long:
            end = min(end, n_long)
        batches.append(order[i:end])
        i = end
    return batches


def _short_on():
    return SHORT_FP32 > 0 and bool(_S) and _S.get("fwd32") is not None


def _build(enc_list, idx, lengths, offset=0):
    """Pad one batch into pinned host tensors. Stage 0/1 pad to 512 like the reference.
    idx indexes enc_list/lengths (chunk-local); the returned index tensor is idx + offset (position in the call)."""
    T = _pad8(int(lengths[idx].max())) if CFG["sort"] or CFG["budget"] else MAX_LEN
    ids = np.full((len(idx), T), _S["pad_id"], dtype=np.int64)
    attn = np.zeros((len(idx), T), dtype=np.int64)
    for r, j in enumerate(idx):
        e = enc_list[j]
        ids[r, :len(e)] = e
        attn[r, :len(e)] = 1
    uniform = bool(lengths[idx].min() == T)  # no padding at all -> run attention without a mask
    fp32 = _short_on() and bool(lengths[idx].max() <= SHORT_FP32)  # P2: whole batch is short -> FP32 path
    ids_t, attn_t = torch.from_numpy(ids), torch.from_numpy(attn)
    idx_t = torch.from_numpy(np.ascontiguousarray(idx + offset, dtype=np.int64))
    if _S["cuda"]:  # pinned (page-locked) host memory -> truly asynchronous host-to-device copies
        ids_t, attn_t, idx_t = ids_t.pin_memory(), attn_t.pin_memory(), idx_t.pin_memory()
    return ids_t, attn_t, idx_t, (uniform, fp32)


def _tokenise(texts):
    encs = _S["fast"].encode_batch(texts)
    ids = [e.ids for e in encs]
    return ids, np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))


def _put(q, item, stop):
    """Blocking put that gives up if the consumer has stopped (so the thread never hangs after an error)."""
    while not stop.is_set():
        try:
            q.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _producer(texts, q, offset_chunks, stop):
    """O3: runs in a thread. Tokenises chunk k+1 while the GPU works on chunk k."""
    try:
        for start in offset_chunks:
            part = texts[start:start + CHUNK]
            ids, lengths = _tokenise(part)
            for idx in _plan(lengths):
                ids_t, attn_t, idx_t, uni = _build(ids, idx, lengths, offset=start)
                if not _put(q, (ids_t, attn_t, idx_t, uni), stop):
                    return
        _put(q, None, stop)
    except BaseException as ex:  # surface errors in the main thread
        _put(q, ex, stop)


def _run_batch(ids_t, attn_t, mode):
    uniform, fp32 = mode
    dev = _S["device"]
    ids = ids_t.to(dev, non_blocking=True)
    attn = attn_t.to(dev, non_blocking=True)
    fwd = _S["fwd32"] if fp32 else _S["fwd"]
    try:
        return fwd(ids, attn, not uniform, bool(CFG["cls_only"]))
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        h = ids.shape[0] // 2
        if h == 0:
            raise
        return torch.cat([_run_batch(ids_t[:h], attn_t[:h], mode), _run_batch(ids_t[h:], attn_t[h:], mode)])


# ----------------------------------------------------------------------------------------------- encode
def _encode_live(text):
    """O7: single query through a captured CUDA Graph."""
    ids = _S["fast"].encode(text).ids
    T = next((b for b in LIVE_BUCKETS if b >= len(ids)), None)
    if T is None:
        return None
    if _short_on() and (len(ids) <= SHORT_FP32) != (T <= SHORT_FP32):
        return None  # bucket precision does not match this query's (cut-off not on a bucket edge): batch path
    g, s_ids, s_attn, s_out, h_ids, h_attn = _S["graphs"][T]
    h_ids.zero_(); h_attn.zero_()
    h_ids[0, :len(ids)] = torch.tensor(ids)
    h_attn[0, :len(ids)] = 1
    s_ids.copy_(h_ids, non_blocking=True)
    s_attn.copy_(h_attn, non_blocking=True)
    g.replay()
    return s_out.cpu().numpy()


def encode(texts):
    if not _S:
        setup()
    texts = list(texts)
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    with _LOCK, torch.inference_mode():
        return _encode(texts)


def _encode(texts):
    N = len(texts)
    if N == 1 and CFG["graph"] and _S["graphs"]:
        r = _encode_live(texts[0])
        if r is not None:
            return r

    out = torch.empty((N, 768), dtype=torch.float32, device=_S["device"])
    if CFG["overlap"]:
        q, stop = queue.Queue(maxsize=64), threading.Event()
        th = threading.Thread(target=_producer, args=(texts, q, range(0, N, CHUNK), stop), daemon=True)
        th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                ids_t, attn_t, idx_t, uni = item
                out[idx_t.to(_S["device"], non_blocking=True)] = _run_batch(ids_t, attn_t, uni)
        finally:
            stop.set()   # on any error the producer exits instead of blocking on a full queue
            th.join()
    else:
        ids, lengths = _tokenise(texts)
        for idx in _plan(lengths):
            ids_t, attn_t, idx_t, uni = _build(ids, idx, lengths)
            out[idx_t.to(_S["device"])] = _run_batch(ids_t, attn_t, uni)
    return out.cpu().numpy()  # single device -> host copy (and the only sync)
