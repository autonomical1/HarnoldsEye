import bisect
import math
import os
import re
import sys
import json
import subprocess
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# CONFIG
# ============================================================================

ML_MODEL_NAME = "microsoft/codebert-base"

# Pickled `RobertaForSequenceClassification` (or similar) from training; MPS checkpoints load on CPU via patch below.
def _model_pkl_path() -> Path:
    env = os.environ.get("MODEL_PKL")
    return Path(env) if env else Path(__file__).resolve().parent / "model.pkl"


# Tokenizer HF id if config has no usable `name_or_path` (e.g. local Mac path).
CLASSIFIER_TOKENIZER_DEFAULT = os.environ.get(
    "CLASSIFIER_TOKENIZER", "microsoft/codebert-base"
)

# Absolute floor: embeddings often score high for unrelated C/C++; require at least this.
ML_MIN_ABSOLUTE_SIMILARITY = float(
    os.environ.get("ML_MIN_ABSOLUTE_SIMILARITY", "0.78")
)

# Best category must beat the second-best by this margin (reduces random "winner" picks).
ML_MIN_CATEGORY_MARGIN = float(os.environ.get("ML_MIN_CATEGORY_MARGIN", "0.012"))

# Within a single repo scan, only keep snippets whose max similarity is at or above this
# percentile of all snippet scores (e.g. 75 → top ~25% in that clone). Lower = more candidates for Gemini.
ML_REPO_SCORE_PERCENTILE = float(
    os.environ.get("ML_REPO_SCORE_PERCENTILE", "75")
)

# Below this many scored snippets, skip the repo-wide percentile (cutoff becomes the minimum
# score in the clone). Tiny training/demo repos often have many high softmax scores; p88 then
# keeps only a handful and drops obvious vulns (e.g. Damn_Vulnerable_C_Program).
ML_REPO_PERCENTILE_MIN_SNIPPETS = int(
    os.environ.get("ML_REPO_PERCENTILE_MIN_SNIPPETS", "250")
)

# Binary classifiers (e.g. model.pkl with LABEL_0 / LABEL_1): only keep snippets where the
# winning class matches this label. Set to empty to disable (not recommended).
ML_CLASSIFIER_POSITIVE_LABEL = os.environ.get(
    "ML_CLASSIFIER_POSITIVE_LABEL", "LABEL_1"
).strip()

# When using model.pkl classifier, softmax “confidence” is often lower than cosine scores.
ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER = float(
    os.environ.get("ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER", "0.35")
)

# Reference vulnerable C/C++ patterns (embeddings of these are computed at startup)
RISKY_C_CPP_PATTERNS = {
    "buffer_overflow": [
        "strcpy(dest, src);",
        "gets(buffer);",
        "strcat(dest, src);",
        "sprintf(buffer, format, value);",
        "char buf[10]; strcpy(buf, user_input);",
    ],
    "sql_injection": [
        "sqlite3_exec(db, query);",
        'sqlite3_prepare_v2(db, \"SELECT * FROM users WHERE id = \" + id',
        "snprintf(sql, sizeof(sql), \"SELECT * FROM %s\", table);",
    ],
    "command_injection": [
        "system(command);",
        "popen(command, \"r\");",
        "exec(command);",
        "system(user_input);",
    ],
    "use_after_free": [
        "free(ptr); ... ptr->field",
        "delete obj; ... obj->method();",
        "free(p); memcpy(p, src, len);",
    ],
    "memory_leak": [
        "malloc(...); // never freed",
        "new Object(); // never deleted",
        "char* str = strdup(input);",
    ],
    "integer_overflow": [
        "int size = len + extra;",
        "if (a > INT_MAX - b) overflow",
        "unsigned int total = x + y;",
    ],
    "unsafe_function": [
        "strcpy", "strcat", "gets", "sprintf",
        "scanf", "fscanf", "printf(user_input)",
    ],
    "uninitialized_variable": [
        "int x; if (x == 0)",
        "char* p; strcpy(p, src);",
        "FILE* f; fopen(f, name);",
    ],
    "format_string": [
        "printf(user_input);",
        "fprintf(f, format);",
        "snprintf(buf, len, format);",
    ],
    "null_pointer": [
        "ptr->field; // ptr unchecked",
        "if (ptr) ... else ptr->method();",
        "*ptr = value; // ptr may be null",
    ],
}

# ============================================================================
# SCAN PROGRESS (merged into API /api/scan/status while a scan runs)
# ============================================================================

SCAN_PROGRESS: Dict[str, Any] = {}


def _emit_scan_progress(
    progress_callback: Optional[Any],
    **kwargs: Any,
) -> None:
    SCAN_PROGRESS.update(kwargs)
    if progress_callback is not None:
        try:
            progress_callback(dict(SCAN_PROGRESS))
        except Exception:
            pass


# ============================================================================
# ML MODEL
# ============================================================================


def _configure_torch_cpu_threads() -> None:
    """
    PyTorch runs the classifier on CPU by default; matmul uses intra-op threads (not Python threads).
    Call once before heavy inference. Override with TORCH_NUM_THREADS / OMP_NUM_THREADS.
    """
    try:
        import multiprocessing
        import torch
    except Exception:
        return
    explicit = os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if explicit:
        try:
            n = max(1, int(explicit.strip()))
            torch.set_num_threads(n)
        except (TypeError, ValueError):
            pass
    else:
        try:
            cpu = multiprocessing.cpu_count() or 4
            # High RAM hosts can use more intra-op threads; override with TORCH_NUM_THREADS.
            torch.set_num_threads(min(cpu, 64))
        except Exception:
            pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _pickle_load_with_mps_to_cpu(path: Path) -> Any:
    """
    Apple MPS-saved checkpoints reference `mps` storages; Linux cannot allocate them.
    Monkey-patch UntypedStorage.mps to materialize on CPU before unpickling.
    """
    import pickle
    import torch

    orig_mps = torch.UntypedStorage.mps

    def _mps_to_cpu(self: Any) -> Any:
        return torch.UntypedStorage(self.size(), device="cpu").copy_(self, False)

    torch.UntypedStorage.mps = _mps_to_cpu
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    finally:
        torch.UntypedStorage.mps = orig_mps


def _tokenizer_for_classifier(model: Any) -> Any:
    from transformers import AutoTokenizer

    explicit = os.environ.get("CLASSIFIER_TOKENIZER_NAME")
    if explicit:
        return AutoTokenizer.from_pretrained(explicit)

    cfg = model.config
    name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
    if isinstance(name, str) and name and not name.startswith(("/", ".", "~")):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception:
            pass

    print(f"Using tokenizer: {CLASSIFIER_TOKENIZER_DEFAULT}")
    return AutoTokenizer.from_pretrained(CLASSIFIER_TOKENIZER_DEFAULT)


class VulnerabilityScanner:
    """
    If `model.pkl` exists next to this file (or MODEL_PKL), loads a pickled
    Hugging Face sequence classifier (e.g. RobertaForSequenceClassification).
    Otherwise falls back to SentenceTransformer + reference-pattern cosine scores.
    """

    def __init__(self) -> None:
        self._torch_model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None  # SentenceTransformer when embedding mode
        self.pattern_embeddings: Optional[Dict[str, List[Any]]] = None
        self.uses_classifier: bool = False

        _configure_torch_cpu_threads()

        pkl = _model_pkl_path()
        if pkl.is_file():
            try:
                self._init_classifier(pkl)
                self.uses_classifier = True
                return
            except Exception as e:
                print(f"⚠ Could not load trained model {pkl}: {e}")
                print("  Falling back to SentenceTransformer + pattern embeddings.")

        self._init_embedding_scanner()

    def _init_classifier(self, path: Path) -> None:
        import torch

        print(f"Loading trained classifier from {path}...")
        obj = _pickle_load_with_mps_to_cpu(path)
        obj.train(False)
        obj.to(torch.device("cpu"))
        self._torch_model = obj
        self.tokenizer = _tokenizer_for_classifier(obj)
        n_labels = getattr(obj.config, "num_labels", "?")
        print(f"✓ Classifier ready (num_labels={n_labels})")
        print(
            "  Tip: softmax scores often peak below 0.86; set ML_MIN_ABSOLUTE_SIMILARITY "
            "or ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER (e.g. 0.35–0.55) if nothing passes filters."
        )
        try:
            import torch

            print(
                f"  PyTorch CPU threads: {torch.get_num_threads()} "
                f"(set TORCH_NUM_THREADS / OMP_NUM_THREADS to tune)"
            )
        except Exception:
            pass

    def _init_embedding_scanner(self) -> None:
        from sentence_transformers import SentenceTransformer

        print("Loading CodeBERT (SentenceTransformers)...")
        self.model = SentenceTransformer(ML_MODEL_NAME)
        print("Computing reference pattern embeddings...")
        self.pattern_embeddings = {}
        for category, patterns in RISKY_C_CPP_PATTERNS.items():
            self.pattern_embeddings[category] = [
                self.model.encode(p) for p in patterns
            ]
        print(f"✓ Embedding model with {len(RISKY_C_CPP_PATTERNS)} vulnerability categories")

    def score_snippet(self, code_snippet: str) -> Dict:
        """
        Raw scores for find_vulnerable_snippets (percentile + margin filters).

        Classifier mode: max_similarity = max softmax probability across labels.
        Embedding mode: max cosine vs reference patterns per category.
        """
        if self._torch_model is not None:
            return self._score_with_classifier(code_snippet)
        return self._score_with_embeddings(code_snippet)

    def _score_with_classifier(self, code_snippet: str) -> Dict:
        import torch

        model = self._torch_model
        assert self.tokenizer is not None
        max_len = int(os.environ.get("CLASSIFIER_MAX_LENGTH", "512"))
        enc = self.tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
        probs_t = torch.softmax(logits, dim=-1)[0]
        probs = probs_t.cpu().tolist()

        id2label = getattr(model.config, "id2label", None) or {}
        all_scores: Dict[str, float] = {}
        for i, p in enumerate(probs):
            lab = id2label.get(i)
            if lab is None and isinstance(id2label, dict):
                lab = id2label.get(str(i))
            if lab is None:
                lab = str(i)
            elif not isinstance(lab, str):
                lab = str(lab)
            all_scores[lab] = float(p)

        per = sorted(all_scores.values(), reverse=True)
        max_similarity = per[0]
        second_best = per[1] if len(per) > 1 else 0.0
        best_category = max(all_scores, key=all_scores.get)

        return {
            "max_similarity": max_similarity,
            "second_best_similarity": second_best,
            "category_margin": max_similarity - second_best,
            "matched_category": best_category,
            "all_scores": all_scores,
        }

    def score_snippets_classifier_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Batched forward pass — much faster than score_snippet per snippet on CPU."""
        import torch

        if batch_size is None:
            batch_size = max(1, int(os.environ.get("CLASSIFIER_BATCH_SIZE", "48")))
        model = self._torch_model
        tokenizer = self.tokenizer
        assert model is not None and tokenizer is not None
        max_len = int(os.environ.get("CLASSIFIER_MAX_LENGTH", "512"))
        device = next(model.parameters()).device
        id2label = getattr(model.config, "id2label", None) or {}

        def _one_row_probs(probs: List[float]) -> Dict[str, Any]:
            all_scores: Dict[str, float] = {}
            for i, p in enumerate(probs):
                lab = id2label.get(i)
                if lab is None and isinstance(id2label, dict):
                    lab = id2label.get(str(i))
                if lab is None:
                    lab = str(i)
                elif not isinstance(lab, str):
                    lab = str(lab)
                all_scores[lab] = float(p)
            per = sorted(all_scores.values(), reverse=True)
            max_similarity = per[0]
            second_best = per[1] if len(per) > 1 else 0.0
            best_category = max(all_scores, key=all_scores.get)
            return {
                "max_similarity": max_similarity,
                "second_best_similarity": second_best,
                "category_margin": max_similarity - second_best,
                "matched_category": best_category,
                "all_scores": all_scores,
            }

        out: List[Dict[str, Any]] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    padding=True,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs_t = torch.softmax(logits, dim=-1)
                for row in range(probs_t.shape[0]):
                    out.append(_one_row_probs(probs_t[row].cpu().tolist()))
        return out

    def _score_with_embedding_vector(self, snippet_embedding: Any) -> Dict[str, Any]:
        """Score one snippet from a precomputed embedding row (1d array)."""
        import numpy as np

        assert self.pattern_embeddings is not None
        vec = np.atleast_2d(np.asarray(snippet_embedding, dtype=float))

        all_scores: Dict[str, float] = {}
        for category, pattern_embeddings in self.pattern_embeddings.items():
            similarities = [
                cosine_similarity(vec, np.atleast_2d(np.asarray(p_emb, dtype=float)))[
                    0
                ][0]
                for p_emb in pattern_embeddings
            ]
            max_sim = max(similarities) if similarities else 0.0
            all_scores[category] = float(max_sim)

        per_category = sorted(all_scores.values(), reverse=True)
        max_similarity = per_category[0]
        second_best = per_category[1] if len(per_category) > 1 else 0.0
        best_category = max(all_scores, key=all_scores.get)

        return {
            "max_similarity": max_similarity,
            "second_best_similarity": second_best,
            "category_margin": max_similarity - second_best,
            "matched_category": best_category,
            "all_scores": all_scores,
        }

    def _score_with_embeddings(self, code_snippet: str) -> Dict:
        assert self.model is not None
        snippet_embedding = self.model.encode(code_snippet)
        return self._score_with_embedding_vector(snippet_embedding)


# ============================================================================
# CODE EXTRACTION (C/C++ ONLY)
# ============================================================================

def clone_repository(github_url: str) -> str:
    """Clone GitHub repo and return path"""
    temp_dir = tempfile.mkdtemp(prefix="c_scan_")
    
    try:
        print(f"Cloning {github_url}...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                github_url,
                temp_dir,
            ],
            timeout=120,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Repository cloned to {temp_dir}")
        return temp_dir
    
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir)
        raise Exception(f"Clone timeout after 120s")
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        raise Exception(f"Clone failed: {e.stderr}")


def extract_c_cpp_files(repo_path: str) -> List[Dict]:
    """
    Extract all C/C++ files from repo.
    
    Returns:
    [
        {
            "file_path": "src/main.c",
            "absolute_path": "/tmp/.../src/main.c",
            "language": "c" | "cpp",
            "lines": [line1, line2, ...]
        }
    ]
    """
    # Implementation / header-only: set SCAN_INCLUDE_HEADERS=0 to scan .c/.cpp/.cc/.cxx only.
    include_headers = os.environ.get("SCAN_INCLUDE_HEADERS", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    header_exts = {".h", ".hpp", ".hh"}
    c_cpp_extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh"}
    files = []

    for root, dirs, filenames in os.walk(repo_path):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in {
            '.git', '.github', 'build', 'cmake-build-debug',
            '__pycache__', '.pytest_cache', 'node_modules',
            'third_party', 'vendor', 'external', 'deps', 'vcpkg_installed',
        }]
        
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext not in c_cpp_extensions:
                continue
            if not include_headers and ext in header_exts:
                continue
            absolute_path = os.path.join(root, filename)
            relative_path = os.path.relpath(absolute_path, repo_path)

            # Skip large files
            if os.path.getsize(absolute_path) > 1_000_000:
                print(f"⊘ Skipping large file: {relative_path}")
                continue

            try:
                with open(absolute_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                language = (
                    "cpp"
                    if ext in {".cpp", ".cc", ".cxx", ".hpp", ".hh"}
                    else "c"
                )

                files.append({
                    "file_path": relative_path,
                    "absolute_path": absolute_path,
                    "language": language,
                    "lines": lines,
                })

            except Exception as e:
                print(f"⊘ Error reading {relative_path}: {e}")
                continue
    
    mode = "C/C++ sources + headers" if include_headers else "C/C++ sources only (.c/.cpp/.cc/.cxx)"
    print(f"✓ Found {len(files)} file(s) ({mode})")
    return files


# --- Method / function-shaped chunking (brace + paren aware) -----------------

_CONTROL_BEFORE_PAREN = frozenset({
    "if", "while", "for", "switch", "catch", "sizeof", "alignof", "decltype",
    "static_assert",
})

_QUAL_BEFORE_BRACE = (
    "noexcept", "override", "final", "volatile", "const", "mutable",
)


def _line_starts_from_lines(lines: List[str]) -> List[int]:
    starts: List[int] = []
    p = 0
    for ln in lines:
        starts.append(p)
        p += len(ln)
    return starts


def _line_index_for_char(line_starts: List[int], char_idx: int) -> int:
    return bisect.bisect_right(line_starts, char_idx) - 1


def _build_paren_match_map(text: str) -> Dict[int, int]:
    """Map each '(' index to its matching ')' and vice versa (code only; skips strings/comments)."""
    match: Dict[int, int] = {}
    stack: List[int] = []
    i = 0
    n = len(text)
    state = "code"
    while i < n:
        c = text[i]
        if state == "code":
            if c == "/" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "/":
                    state = "line_comment"
                    i += 2
                    continue
                if nxt == "*":
                    state = "block_comment"
                    i += 2
                    continue
            if c == '"':
                state = "string"
                i += 1
                continue
            if c == "'":
                state = "char"
                i += 1
                continue
            if c == "(":
                stack.append(i)
            elif c == ")":
                if stack:
                    o = stack.pop()
                    match[o] = i
                    match[i] = o
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                state = "code"
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                state = "code"
                i += 2
                continue
            i += 1
            continue
        if state == "string":
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                state = "code"
            i += 1
            continue
        if state == "char":
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                state = "code"
            i += 1
            continue
    return match


def _skip_ws_back(text: str, j: int) -> int:
    while j >= 0 and text[j] in " \t\n\r\v\f":
        j -= 1
    return j


def _closing_paren_before_brace(text: str, brace_idx: int) -> int:
    """
    Given index of '{', return index of ')' that immediately precedes the block
    after optional const/noexcept/override/&, or -1 if not a ) { pattern.
    """
    j = brace_idx - 1
    j = _skip_ws_back(text, j)
    if j < 0:
        return -1
    changed = True
    while changed:
        changed = False
        j = _skip_ws_back(text, j)
        if j < 0:
            return -1
        for kw in _QUAL_BEFORE_BRACE:
            L = len(kw)
            if j + 1 >= L and text[j - L + 1 : j + 1].lower() == kw:
                bef = text[j - L] if j >= L else " "
                if bef in " \t\n\r({[<:":
                    j = j - L
                    changed = True
                    break
        if changed:
            continue
        if j >= 1 and text[j - 1 : j + 1] == "&&":
            j -= 2
            changed = True
            continue
        if j >= 0 and text[j] == "&":
            j -= 1
            changed = True
            continue
    j = _skip_ws_back(text, j)
    if j >= 0 and text[j] == ")":
        return j
    return -1


def _callee_token_before_open_paren(text: str, open_paren_idx: int) -> str:
    """Last identifier segment before '(' (after ::), lowercased."""
    j = open_paren_idx - 1
    j = _skip_ws_back(text, j)
    if j < 0:
        return ""
    end = j
    while j >= 0 and (text[j].isalnum() or text[j] == "_"):
        j -= 1
    raw = text[j + 1 : end + 1]
    if "::" in raw:
        raw = raw.split("::")[-1]
    return raw.lower()


def _is_function_like_open_brace(text: str, brace_idx: int, paren_match: Dict[int, int]) -> bool:
    rp = _closing_paren_before_brace(text, brace_idx)
    if rp < 0 or rp not in paren_match:
        return False
    open_p = paren_match[rp]
    callee = _callee_token_before_open_paren(text, open_p)
    if callee in _CONTROL_BEFORE_PAREN:
        return False
    return True


def _header_start_line_0(lines: List[str], brace_line_0: int) -> int:
    """Include preceding lines that belong to the signature (params, attributes, template)."""
    i = brace_line_0
    while i > 0:
        prev = lines[i - 1].rstrip("\n")
        if not prev.strip():
            break
        s = prev.rstrip()
        if s.endswith(";"):
            break
        if s.endswith("\\"):
            i -= 1
            continue
        if s.endswith(",") or s.endswith("("):
            i -= 1
            continue
        if re.search(r"::\s*$", s) or re.search(r">\s*$", s):
            i -= 1
            continue
        st = s.strip()
        if st.startswith("[[") and "]]" in st:
            i -= 1
            continue
        if st.startswith("__attribute__"):
            i -= 1
            continue
        if re.match(r"^\s*template\s*<", s):
            i -= 1
            continue
        if "(" in s and ";" not in s:
            i -= 1
            continue
        break
    return i


def _extract_method_snippets_from_text(
    lines: List[str],
    file_path: str,
    text: str,
    line_starts: List[int],
) -> List[Dict]:
    paren_match = _build_paren_match_map(text)
    snippets: List[Dict] = []
    brace_stack: List[Tuple[int, bool]] = []  # (char_index of '{', is_function_like)

    i = 0
    n = len(text)
    state = "code"

    def line_0_at(pos: int) -> int:
        return _line_index_for_char(line_starts, pos)

    while i < n:
        c = text[i]
        if state == "code":
            if c == "/" and i + 1 < n:
                nxt = text[i + 1]
                if nxt == "/":
                    state = "line_comment"
                    i += 2
                    continue
                if nxt == "*":
                    state = "block_comment"
                    i += 2
                    continue
            if c == '"':
                state = "string"
                i += 1
                continue
            if c == "'":
                state = "char"
                i += 1
                continue
            if c == "{":
                flike = _is_function_like_open_brace(text, i, paren_match)
                brace_stack.append((i, flike))
                i += 1
                continue
            if c == "}":
                if brace_stack:
                    open_idx, flike = brace_stack.pop()
                    if flike:
                        close_idx = i
                        bline = line_0_at(open_idx)
                        h0 = _header_start_line_0(lines, bline)
                        e0 = line_0_at(close_idx)
                        chunk = "".join(lines[h0 : e0 + 1]).strip()
                        if len(chunk) >= 10:
                            snippets.append(
                                {
                                    "code": chunk,
                                    "line_start": h0 + 1,
                                    "line_end": e0 + 1,
                                    "file_path": file_path,
                                }
                            )
                i += 1
                continue
            i += 1
            continue
        if state == "line_comment":
            if c == "\n":
                state = "code"
            i += 1
            continue
        if state == "block_comment":
            if c == "*" and i + 1 < n and text[i + 1] == "/":
                state = "code"
                i += 2
                continue
            i += 1
            continue
        if state == "string":
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == '"':
                state = "code"
            i += 1
            continue
        if state == "char":
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == "'":
                state = "code"
            i += 1
            continue

    return snippets


_C_KEYWORD_NOT_FUNC_NAMES = frozenset(
    {
        "if",
        "while",
        "for",
        "switch",
        "catch",
        "sizeof",
        "return",
        "case",
        "else",
        "struct",
        "union",
        "enum",
    }
)


def _function_name_from_snippet_code(code: str) -> Optional[str]:
    """Best-effort C function name from snippet text before first '{'."""
    i = code.find("{")
    if i < 0:
        return None
    head = code[:i]
    matches = list(re.finditer(r"\b([A-Za-z_]\w*)\s*\(", head))
    for m in reversed(matches):
        name = m.group(1)
        if name not in _C_KEYWORD_NOT_FUNC_NAMES:
            return name
    return None


def _subdivide_oversized_snippets(
    snippets: List[Dict],
    lines: List[str],
    file_path: str,
    file_line_count: int,
) -> List[Dict]:
    """
    On small files, split very large function bodies (e.g. ConnectionHandler) into
    overlapping line windows so ML/Gemini see each command path separately.
    """
    if os.environ.get("SNIPPET_SUBDIVIDE_LARGE_FUNCTIONS", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return snippets
    max_file = max(50, int(os.environ.get("SNIPPET_SMALL_FILE_MAX_LINES", "600")))
    if file_line_count > max_file:
        return snippets
    min_span = max(20, int(os.environ.get("SNIPPET_SUBDIVIDE_MIN_LINES", "80")))
    win = max(12, int(os.environ.get("SNIPPET_SUBCHUNK_LINES", "48")))
    stride = max(1, int(os.environ.get("SNIPPET_SUBCHUNK_STRIDE", "24")))
    if stride > win:
        stride = max(1, win // 2)

    out: List[Dict] = []
    for sn in snippets:
        span = int(sn["line_end"]) - int(sn["line_start"]) + 1
        if span < min_span:
            out.append(sn)
            continue
        ls, le = int(sn["line_start"]), int(sn["line_end"])
        i0 = ls - 1
        i1 = min(len(lines), le)
        sublines = lines[i0:i1]
        if len(sublines) < min_span:
            out.append(sn)
            continue
        for j in range(0, len(sublines), stride):
            chunk = sublines[j : j + win]
            if len(chunk) < 5:
                continue
            code = "".join(chunk).strip()
            if len(code) < 15:
                continue
            out.append(
                {
                    "code": code,
                    "line_start": ls + j,
                    "line_end": ls + j + len(chunk) - 1,
                    "file_path": file_path,
                }
            )
    return out if out else snippets


def _file_function_spans(lines: List[str], file_path: str) -> Dict[str, Tuple[int, int]]:
    """Map C function name -> (line_start, line_end) inclusive 1-based from method-shaped extraction."""
    text = "".join(lines)
    line_starts = _line_starts_from_lines(lines)
    snips = _extract_method_snippets_from_text(lines, file_path, text, line_starts)
    out: Dict[str, Tuple[int, int]] = {}
    for sn in snips:
        name = _function_name_from_snippet_code(sn.get("code") or "")
        if not name:
            continue
        a, b = int(sn["line_start"]), int(sn["line_end"])
        if name in out:
            oa, ob = out[name]
            out[name] = (min(oa, a), max(ob, b))
        else:
            out[name] = (a, b)
    return out


def _expand_findings_with_call_sites(
    findings: List[Dict[str, Any]],
    files: List[Dict],
) -> List[Dict[str, Any]]:
    """
    For each finding whose lines lie inside a named function body, also emit rows at
    out-of-body call sites that invoke that function (same file).
    """
    if os.environ.get("CALL_SITE_ATTRIBUTION", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return findings
    if not findings:
        return findings

    path_lines: Dict[str, List[str]] = {
        f["file_path"]: f["lines"] for f in files if f.get("file_path") and f.get("lines")
    }
    path_funcs: Dict[str, Dict[str, Tuple[int, int]]] = {}
    for fp, lns in path_lines.items():
        path_funcs[fp] = _file_function_spans(lns, fp)

    seen_call: set = set()
    out: List[Dict[str, Any]] = list(findings)

    for f in findings:
        if f.get("finding_role") == "call_site":
            continue
        fp = f.get("file_path") or ""
        if fp not in path_lines:
            continue
        lines = path_lines[fp]
        funcs = path_funcs.get(fp) or {}
        try:
            ls, le = int(f["line_start"]), int(f["line_end"])
        except (TypeError, ValueError, KeyError):
            continue

        containing: List[Tuple[int, str]] = []
        for name, (ds, de) in funcs.items():
            if ds <= ls and le <= de:
                containing.append((de - ds, name))
        if not containing:
            continue
        containing.sort(key=lambda t: (t[0], t[1]))
        sink_name = containing[0][1]

        def_lo, def_hi = funcs[sink_name]
        for ln_1, line_text in enumerate(lines, start=1):
            if def_lo <= ln_1 <= def_hi:
                continue
            if not re.search(r"\b" + re.escape(sink_name) + r"\s*\(", line_text):
                continue
            key = (fp, ln_1, sink_name)
            if key in seen_call:
                continue
            seen_call.add(key)
            vtype = (f.get("vulnerability_type") or f.get("matched_category") or "").strip()
            out.append(
                {
                    "file_path": fp,
                    "line_start": ln_1,
                    "line_end": ln_1,
                    "code": (line_text or "").strip()[:200],
                    "matched_category": f.get("matched_category", ""),
                    "similarity_score": float(f.get("similarity_score") or 0.0),
                    "vulnerability_type": vtype or f.get("matched_category", "unspecified"),
                    "finding_role": "call_site",
                    "sink_line_start": ls,
                    "sink_line_end": le,
                    "sink_function": sink_name,
                }
            )
    return out


def extract_snippets_from_lines(
    lines: List[str],
    file_path: str,
    snippet_size: Optional[int] = None,
) -> List[Dict]:
    """
    Extract snippets aligned to whole functions/methods: signature lines + `{` … `}`.

    Uses paren matching (strings/comments-aware) and treats a block as function-like
    when `{` is preceded by `)` (after const/noexcept/…) and the matching `(` is not
    from if/while/for/switch/catch/….

    Set SNIPPET_MODE=window to restore overlapping fixed-height windows (snippet_size / stride).
    """
    mode = os.environ.get("SNIPPET_MODE", "method").strip().lower()
    if mode == "window":
        if snippet_size is None:
            snippet_size = max(3, int(os.environ.get("SNIPPET_WINDOW_LINES", "10")))
        stride_env = int(os.environ.get("SNIPPET_LINE_STRIDE", "0"))
        stride = stride_env if stride_env > 0 else max(1, snippet_size // 2)
        snippets: List[Dict] = []
        for i in range(0, len(lines), stride):
            end_idx = min(i + snippet_size, len(lines))
            if end_idx - i < 3:
                continue
            snippet_code = "".join(lines[i:end_idx]).strip()
            if not snippet_code or len(snippet_code) < 10:
                continue
            snippets.append(
                {
                    "code": snippet_code,
                    "line_start": i + 1,
                    "line_end": end_idx,
                    "file_path": file_path,
                }
            )
        return snippets

    text = "".join(lines)
    line_starts = _line_starts_from_lines(lines)
    snippets = _extract_method_snippets_from_text(lines, file_path, text, line_starts)
    snippets = _subdivide_oversized_snippets(
        snippets, lines, file_path, len(lines)
    )

    if not snippets and os.environ.get("SNIPPET_WHOLE_FILE_FALLBACK", "1") not in (
        "0",
        "false",
        "no",
    ):
        whole = "".join(lines).strip()
        if len(whole) >= 10:
            snippets.append(
                {
                    "code": whole,
                    "line_start": 1,
                    "line_end": len(lines),
                    "file_path": file_path,
                }
            )

    return snippets


# ============================================================================
# VULNERABILITY DETECTION
# ============================================================================


def _effective_repo_score_percentile(num_snippets: int) -> float:
    """
    Percentile threshold for repo-wide similarity cutoff.
    Return 0 to disable (nearest-rank then uses min score → all snippets eligible
    subject to ML_MIN_* filters). Large clones use ML_REPO_SCORE_PERCENTILE.
    """
    if num_snippets <= 0:
        return 0.0
    if num_snippets < ML_REPO_PERCENTILE_MIN_SNIPPETS:
        return 0.0
    return ML_REPO_SCORE_PERCENTILE


def _percentile_nearest_rank(values: List[float], pct: float) -> float:
    """Nearest-rank percentile, pct in [0, 100]."""
    if not values:
        return 1.0
    s = sorted(values)
    if pct <= 0:
        return s[0]
    if pct >= 100:
        return s[-1]
    k = int(math.ceil(pct / 100.0 * len(s))) - 1
    return s[max(0, min(k, len(s) - 1))]


# Obvious unsafe C library calls: rescue when ML/percentile/classifier drops them (common in
# multi-file training repos where many snippets look alike or the classifier is conservative).
_HEURISTIC_SINK_CALL_RE = re.compile(
    r"\b(strcpy|strcat|gets|sprintf|vsprintf|scanf|sscanf|fscanf)\s*\(",
    re.I,
)


def _heuristic_sink_rescue_enabled() -> bool:
    return os.environ.get("SCAN_HEURISTIC_SINK_RESCUE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _snippet_has_heuristic_c_sink(code: str) -> bool:
    if not code or not code.strip():
        return False
    return _HEURISTIC_SINK_CALL_RE.search(code) is not None


# ============================================================================
# GOOGLE GEMINI — final binary vulnerability check on ML-flagged chunks
# ============================================================================

_GEMINI_SYSTEM_INSTRUCTION_CORE = """You are a senior application security engineer reviewing C/C++ code.
Respond with **only** one JSON object — no markdown, no prose, no code fences.

Required shape:
- "vulnerable": boolean
- "line_numbers": array of [start_line, end_line] pairs (inclusive), using **1-based line numbers in the real source file** exactly as the user message states for the excerpt (not line numbers inside the fenced code block).
- "vulnerability_type": short string when vulnerable is true — use the **most specific accurate** label (snake_case or plain English). Examples: "heap_buffer_overflow", "stack_buffer_overflow", "integer_overflow", "integer_underflow", "use_after_free", "double_free", "unchecked_malloc", "divide_by_zero", "out_of_bounds_read", "out_of_bounds_write", "format_string", "memory_leak", "command_injection", "null_pointer_dereference". Do **not** label everything "stack_buffer_overflow" when the flaw is heap, arithmetic, lifetime, or logic. Use "" when vulnerable is false.

If there is no genuine security issue: {"vulnerable": false, "line_numbers": [], "vulnerability_type": ""}.
If there is: {"vulnerable": true, "line_numbers": [[a,b], ...], "vulnerability_type": "..."} with the **smallest** spans that pinpoint the flaw."""

_GEMINI_SYSTEM_INSTRUCTION_EDUCATIONAL = """

**Educational / deliberately vulnerable repositories:** If the excerpt contains **classic insecure C/C++** — including unchecked stack/heap copies, **integer overflow/underflow** affecting allocation or indexing, **malloc without NULL checks**, **double free / use-after-free**, **divide by zero**, **OOB reads/writes**, **non-NUL-terminated data with %s**, **resource leaks**, **command/format injection** — answer **vulnerable: true** and mark those lines. Do **not** answer false *only* because the project is a lab, CTF, fuzz harness, or training exercise — judge the code on its merits. Use false when the shown code is actually safe (proper bounds, modern APIs) or non-executable commentary only."""


def _gemini_system_instruction_text() -> str:
    """
    System instruction for Gemini. Default is **strict** (core only): best default for
    large OSS and mixed codebases. Optional **educational** appendix (see
    GEMINI_STRICT_SOURCE_CONTEXT) helps when scanning known CTF/lab trees where the model
    tended to answer vulnerable:false just because the repo looked like homework.
    """
    core = _GEMINI_SYSTEM_INSTRUCTION_CORE
    v = os.environ.get("GEMINI_STRICT_SOURCE_CONTEXT", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return core + _GEMINI_SYSTEM_INSTRUCTION_EDUCATIONAL
    return core


def _gemini_verify_enabled() -> bool:
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        return False
    return os.environ.get("GEMINI_VERIFY", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _gemini_error_keeps_finding() -> bool:
    """
    When the Gemini API fails or the response is not parseable:
    True  -> keep the ML finding (optimistic).
    False -> drop the finding (strict; default — avoids silent false positives).
    """
    v = os.environ.get("GEMINI_ON_ERROR", "drop").strip().lower()
    return v in ("keep", "true", "1", "yes")


def _gemini_debug_enabled() -> bool:
    return os.environ.get("GEMINI_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _gemini_user_prompt(
    file_path: str,
    ml_category: str,
    code: str,
    excerpt_line_start: int,
    excerpt_line_end: int,
) -> str:
    ex_vuln = (
        '{"vulnerable": true, "line_numbers": [[12, 14]], '
        '"vulnerability_type": "integer_overflow"}'
    )
    ex_safe = '{"vulnerable": false, "line_numbers": [], "vulnerability_type": ""}'
    return f"""A static/ML scanner flagged the following code region as suspicious.

Decide whether this excerpt contains a **real security vulnerability** relevant to C or C++ (for example: buffer overflow, missing bounds on external input, format-string misuse, command/SQL injection in native code paths, obvious use-after-free, integer overflow leading to unsafe allocation or indexing, or other exploitable flaws). Do **not** treat style, readability, or hypothetical issues with no plausible attack path as vulnerabilities.

If the path or context looks like a lab/fuzzer/training repo but the code still uses dangerous APIs or missing bounds, treat it as **vulnerable** for those constructs (see system instructions).

ML-suggested label (may be wrong): {ml_category}
File path: {file_path}

**Line map:** The excerpt below corresponds to **file lines {excerpt_line_start}-{excerpt_line_end}** (inclusive, 1-based). Every [start,end] in "line_numbers" must fall within that inclusive range and refer to the **real file**, not the line index inside the code fence.

Code:
```
{code}
```

**Output rules (mandatory):**
1. Respond with **only** one JSON object.
2. Shape: {ex_vuln} when there is a real vulnerability, or {ex_safe} when there is not.
3. "line_numbers" must list minimal inclusive spans (file line numbers) covering only the vulnerable code; use [] when vulnerable is false.
4. "vulnerability_type" must name the **primary** flaw with a **specific** category matching the actual bug (heap vs stack, lifetime, arithmetic, etc.); never default to "stack_buffer_overflow" unless the defect is truly an unchecked write on the stack. Use "" when vulnerable is false."""


def _parse_gemini_vulnerability_type(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()[:200]
    return str(raw).strip()[:200]


def _parse_gemini_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse Gemini JSON: vulnerable, line_numbers, vulnerability_type.
    Returns None if unclear. Omits or invalid line_numbers -> [].
    """
    if not response_text or not response_text.strip():
        return None
    s = response_text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "vulnerable" not in obj:
        return None
    v = obj["vulnerable"]
    if isinstance(v, str):
        if v.lower() in ("true", "1", "yes"):
            vuln = True
        elif v.lower() in ("false", "0", "no"):
            vuln = False
        else:
            return None
    elif isinstance(v, bool):
        vuln = v
    else:
        return None

    raw_ln = obj.get("line_numbers", [])
    line_numbers: List[List[int]] = []
    if isinstance(raw_ln, list):
        for pair in raw_ln:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                try:
                    line_numbers.append([int(pair[0]), int(pair[1])])
                except (TypeError, ValueError):
                    continue

    vtype = _parse_gemini_vulnerability_type(obj.get("vulnerability_type"))
    if not vuln:
        return {"vulnerable": False, "line_numbers": [], "vulnerability_type": ""}
    if not vtype:
        vtype = ""
    return {"vulnerable": True, "line_numbers": line_numbers, "vulnerability_type": vtype}


def _sanitize_gemini_line_numbers(
    pairs: List[List[int]],
    chunk_lo: int,
    chunk_hi: int,
) -> List[Tuple[int, int]]:
    """Clamp inclusive file line ranges to [chunk_lo, chunk_hi]; drop invalid."""
    out: List[Tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        try:
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            continue
        if a > b:
            a, b = b, a
        a = max(chunk_lo, min(a, chunk_hi))
        b = max(chunk_lo, min(b, chunk_hi))
        if a > b:
            continue
        out.append((a, b))
    return out


def _code_preview_for_file_lines(
    snippet_text: str,
    chunk_line_start: int,
    chunk_line_end: int,
    want_start: int,
    want_end: int,
    max_len: int = 200,
) -> str:
    """Slice snippet text using file line numbers relative to chunk bounds."""
    lines = snippet_text.splitlines()
    if not lines and snippet_text:
        lines = [snippet_text]
    n = len(lines)
    span = chunk_line_end - chunk_line_start + 1
    if n <= 0 or span <= 0:
        return snippet_text[:max_len]
    # Align snippet lines to file lines when counts match; otherwise map by offset.
    if n != span:
        i0 = max(0, want_start - chunk_line_start)
        i1 = max(0, want_end - chunk_line_start)
    else:
        i0 = want_start - chunk_line_start
        i1 = want_end - chunk_line_start
    i0 = max(0, min(i0, n - 1))
    i1 = max(0, min(i1, n - 1))
    if i0 > i1:
        i0, i1 = i1, i0
    excerpt = "\n".join(lines[i0 : i1 + 1])
    excerpt = excerpt.strip()
    if len(excerpt) > max_len:
        return excerpt[:max_len] + "…"
    return excerpt


def gemini_evaluate_chunk(
    full_code: str,
    file_path: str,
    ml_category: str,
    excerpt_line_start: int,
    excerpt_line_end: int,
) -> Optional[Dict[str, Any]]:
    """
    Ask Gemini for vulnerability decision + tight line_numbers (frontend shape).

    Returns None on API/parse failure (caller uses _gemini_error_keeps_finding()).
    Otherwise {"vulnerable": bool, "line_numbers": [...], "vulnerability_type": str}.
    If no API key, returns {"vulnerable": True, "line_numbers": [], "vulnerability_type": ""} (keep ML span).
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"vulnerable": True, "line_numbers": [], "vulnerability_type": ""}

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    max_chars = max(4000, int(os.environ.get("GEMINI_MAX_CHUNK_CHARS", "120000")))
    code = full_code if len(full_code) <= max_chars else full_code[:max_chars] + "\n/* [truncated] */\n"
    dbg = _gemini_debug_enabled()

    genai.configure(api_key=api_key)
    user_prompt = _gemini_user_prompt(
        file_path,
        ml_category,
        code,
        excerpt_line_start,
        excerpt_line_end,
    )
    try:
        sys_txt = _gemini_system_instruction_text()
        model = genai.GenerativeModel(
            model_name,
            system_instruction=sys_txt,
        )
        prompt = user_prompt
    except TypeError:
        model = genai.GenerativeModel(model_name)
        prompt = _gemini_system_instruction_text() + "\n\n" + user_prompt
    generation_config = {
        "temperature": 0.0,
        "max_output_tokens": 512,
    }
    if dbg:
        print(
            f"→ Gemini request model={model_name!r} file={file_path!r} "
            f"code_chars={len(code)} prompt_chars={len(prompt)}"
        )
    try:
        resp = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        try:
            text = (resp.text or "").strip()
        except ValueError:
            text = ""
        if dbg:
            um = getattr(resp, "usage_metadata", None)
            print(f"← Gemini usage_metadata={um!r}")
            print(f"← Gemini raw text={text!r}")
    except Exception as e:
        print(f"⚠ Gemini API error ({file_path}): {e}")
        return None

    parsed = _parse_gemini_json_response(text)
    if parsed is None:
        print(f"⚠ Gemini unparsable response for {file_path}: {text[:200]!r}")
        return None
    if dbg:
        print(f"← Gemini parsed={parsed!r}")
    return parsed


def gemini_confirms_vulnerability(
    full_code: str,
    file_path: str,
    ml_category: str,
    excerpt_line_start: int = 1,
    excerpt_line_end: int = 1,
) -> bool:
    """Backward-compatible: True if vulnerable (ignores line narrowing)."""
    v = gemini_evaluate_chunk(
        full_code,
        file_path,
        ml_category,
        excerpt_line_start,
        excerpt_line_end,
    )
    if v is None:
        return _gemini_error_keeps_finding()
    return bool(v.get("vulnerable"))


def _gemini_rows_for_single_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    One Gemini round-trip + row materialization. Used sequentially and from worker threads.
    Returns 0..n output findings (line spans) for this ML chunk.
    """
    full = item.get("full_code") or item.get("code") or ""
    chunk_lo = int(item["line_start"])
    chunk_hi = int(item["line_end"])
    try:
        verdict = gemini_evaluate_chunk(
            full,
            item["file_path"],
            str(item.get("matched_category", "")),
            chunk_lo,
            chunk_hi,
        )
    except Exception as e:
        print(f"⚠ Gemini worker error ({item.get('file_path')}): {e}")
        verdict = None

    if verdict is None:
        if not _gemini_error_keeps_finding():
            return []
        verdict = {
            "vulnerable": True,
            "line_numbers": [],
            "vulnerability_type": "",
        }

    if not verdict.get("vulnerable"):
        return []

    raw_pairs = verdict.get("line_numbers") or []
    ranges = _sanitize_gemini_line_numbers(
        raw_pairs if isinstance(raw_pairs, list) else [],
        chunk_lo,
        chunk_hi,
    )
    if not ranges:
        ranges = [(chunk_lo, chunk_hi)]

    gtype = _parse_gemini_vulnerability_type(verdict.get("vulnerability_type"))
    if not gtype:
        gtype = str(item.get("matched_category", "")).strip() or "unspecified"

    rows: List[Dict[str, Any]] = []
    for a, b in ranges:
        out = {k: v for k, v in item.items() if k != "full_code"}
        out["line_start"] = a
        out["line_end"] = b
        out["vulnerability_type"] = gtype
        out["code"] = _code_preview_for_file_lines(
            full, chunk_lo, chunk_hi, a, b, max_len=200
        )
        rows.append(out)
    return rows


def _filter_findings_with_gemini(
    vulnerable_snippets: List[Dict[str, Any]],
    progress_callback: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Keep chunks Gemini marks vulnerable; narrow line ranges to Gemini line_numbers."""
    import time

    n = len(vulnerable_snippets)
    if n == 0:
        return []

    delay_ms = max(0, int(os.environ.get("GEMINI_DELAY_MS", "200")))
    workers = max(1, int(os.environ.get("GEMINI_MAX_CONCURRENT", "16")))
    workers = min(workers, n)

    if workers <= 1:
        kept: List[Dict[str, Any]] = []
        for i, item in enumerate(vulnerable_snippets):
            _emit_scan_progress(
                progress_callback,
                phase="gemini_verify",
                snippets_total=n,
                snippets_scored=i,
                detail=(
                    f"Making a Gemini API call — chunk {i + 1}/{n} (request in flight)"
                ),
            )
            rows = _gemini_rows_for_single_item(item)
            kept.extend(rows)
            _emit_scan_progress(
                progress_callback,
                phase="gemini_verify",
                snippets_total=n,
                snippets_scored=i + 1,
                detail=(
                    f"Making a Gemini API call — finished chunk {i + 1}/{n} "
                    f"({len(kept)} vulnerable line span(s) so far)"
                ),
            )
            if delay_ms and i + 1 < n:
                time.sleep(delay_ms / 1000.0)
        print(f"✓ Gemini kept {len(kept)} vulnerable line span(s) from {n} ML candidate(s)")
        return kept

    # Parallel I/O-bound API calls (bounded to reduce 429 rate limits).
    _emit_scan_progress(
        progress_callback,
        phase="gemini_verify",
        snippets_total=n,
        snippets_scored=0,
        detail=(
            f"Gemini API — running up to {workers} concurrent request(s) for {n} chunk(s)"
        ),
    )

    lock = threading.Lock()
    finished = 0
    spans_kept = 0
    indexed_results: List[Tuple[int, List[Dict[str, Any]]]] = []

    def _run_indexed(idx_item: Tuple[int, Dict[str, Any]]) -> None:
        nonlocal finished, spans_kept
        idx, it = idx_item
        rows = _gemini_rows_for_single_item(it)
        with lock:
            indexed_results.append((idx, rows))
            finished += 1
            spans_kept += len(rows)
            _emit_scan_progress(
                progress_callback,
                phase="gemini_verify",
                snippets_total=n,
                snippets_scored=finished,
                detail=(
                    f"Gemini API — finished {finished}/{n} chunk(s) in parallel "
                    f"({spans_kept} vulnerable line span(s) so far)"
                ),
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_run_indexed, enumerate(vulnerable_snippets)))

    indexed_results.sort(key=lambda x: x[0])
    kept = []
    for _, rows in indexed_results:
        kept.extend(rows)

    print(f"✓ Gemini kept {len(kept)} vulnerable line span(s) from {n} ML candidate(s)")
    return kept


def find_vulnerable_snippets(
    scanner: VulnerabilityScanner,
    files: List[Dict],
    progress_callback: Optional[Any] = None,
) -> List[Dict]:
    """
    Scan all snippets; keep those passing absolute similarity, category margin,
    and repo-wide percentile (reduces false positives from uniformly high embeddings).
    """
    flat: List[Tuple[str, Dict]] = []
    for file_info in files:
        file_path = file_info["file_path"]
        lines = file_info["lines"]
        snippets = extract_snippets_from_lines(lines, file_path)
        for snippet in snippets:
            flat.append((file_path, snippet))

    total = len(flat)
    n_files = len(files)
    _emit_scan_progress(
        progress_callback,
        phase="scoring_snippets",
        snippets_total=total,
        snippets_scored=0,
        c_cpp_files=n_files,
        detail=f"Scoring {total} snippet window(s) from {n_files} C/C++ file(s)",
    )

    score_results: List[Dict[str, Any]] = []
    if getattr(scanner, "uses_classifier", False) and scanner._torch_model is not None:
        texts = [s["code"] for _, s in flat]
        batch_size = max(1, int(os.environ.get("CLASSIFIER_BATCH_SIZE", "48")))
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            score_results.extend(
                scanner.score_snippets_classifier_batch(chunk, batch_size=len(chunk))
            )
            done = min(start + len(chunk), total)
            _emit_scan_progress(
                progress_callback,
                snippets_scored=done,
                detail=f"Classifier: scored {done}/{total} snippets (batch {batch_size})",
            )
    elif scanner.model is not None and scanner.pattern_embeddings is not None:
        texts = [s["code"] for _, s in flat]
        st_batch = max(1, int(os.environ.get("ST_ENCODE_BATCH_SIZE", "32")))
        embs = scanner.model.encode(
            texts,
            batch_size=st_batch,
            show_progress_bar=False,
        )
        for idx, emb_row in enumerate(embs):
            score_results.append(scanner._score_with_embedding_vector(emb_row))
            if (idx + 1) % 64 == 0 or idx + 1 == total:
                _emit_scan_progress(
                    progress_callback,
                    snippets_scored=idx + 1,
                    detail=f"Embeddings: scored {idx + 1}/{total} snippets",
                )
    else:
        for i, (_, snip) in enumerate(flat):
            score_results.append(scanner.score_snippet(snip["code"]))
            if (i + 1) % 50 == 0 or i + 1 == total:
                _emit_scan_progress(
                    progress_callback,
                    snippets_scored=i + 1,
                    detail=f"Scored {i + 1}/{total} snippets",
                )

    scored: List[Tuple[str, Dict, Dict]] = [
        (flat[i][0], flat[i][1], score_results[i]) for i in range(len(flat))
    ]

    max_sims = [r["max_similarity"] for _, _, r in scored]
    eff_pct = _effective_repo_score_percentile(total)
    cutoff = _percentile_nearest_rank(max_sims, eff_pct)
    if eff_pct <= 0:
        print(
            f"ℹ Repo-wide score percentile skipped ({total} snippet(s) "
            f"< {ML_REPO_PERCENTILE_MIN_SNIPPETS}); cutoff={cutoff:.4f} (min score in clone)"
        )
    else:
        print(
            f"ℹ Repo-wide score percentile={eff_pct:.0f}% → cutoff={cutoff:.4f} "
            f"({total} snippet(s))"
        )

    min_abs = (
        ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER
        if getattr(scanner, "uses_classifier", False)
        else ML_MIN_ABSOLUTE_SIMILARITY
    )

    gemini_on = _gemini_verify_enabled()
    heuristic_without_gemini = os.environ.get(
        "SCAN_HEURISTIC_WITHOUT_GEMINI", "0"
    ).strip().lower() in ("1", "true", "yes", "on")
    vulnerable_snippets: List[Dict[str, Any]] = []
    included_keys: set = set()
    for file_path, snippet, r in scored:
        if r["max_similarity"] < min_abs:
            continue
        if r["category_margin"] < ML_MIN_CATEGORY_MARGIN:
            continue
        if r["max_similarity"] < cutoff:
            continue
        if (
            getattr(scanner, "uses_classifier", False)
            and ML_CLASSIFIER_POSITIVE_LABEL
            and r.get("matched_category") != ML_CLASSIFIER_POSITIVE_LABEL
        ):
            continue
        row: Dict[str, Any] = {
            "file_path": file_path,
            "line_start": snippet["line_start"],
            "line_end": snippet["line_end"],
            "code": snippet["code"][:200],
            "matched_category": r["matched_category"],
            "similarity_score": r["max_similarity"],
            "all_scores": r["all_scores"],
        }
        if gemini_on:
            row["full_code"] = snippet["code"]
        vulnerable_snippets.append(row)
        included_keys.add(
            (file_path, int(snippet["line_start"]), int(snippet["line_end"]))
        )

    # Second pass: same files were always scanned; ML gates often drop "obvious" strcpy/scanf
    # chunks in repos full of near-identical exercises (e.g. overflow_with_joy).
    if _heuristic_sink_rescue_enabled() and (gemini_on or heuristic_without_gemini):
        h_max = max(0, int(os.environ.get("SCAN_HEURISTIC_SINK_MAX", "512")))
        require_min_sim = os.environ.get("SCAN_HEURISTIC_REQUIRE_MIN_SIM", "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "",
        )
        h_min_sim = float(os.environ.get("SCAN_HEURISTIC_MIN_SIMILARITY", "0.08"))
        added_h = 0
        for file_path, snippet, r in scored:
            if added_h >= h_max:
                break
            key = (file_path, int(snippet["line_start"]), int(snippet["line_end"]))
            if key in included_keys:
                continue
            if not _snippet_has_heuristic_c_sink(snippet["code"]):
                continue
            if require_min_sim and r["max_similarity"] < h_min_sim:
                continue
            mc = r["matched_category"]
            if getattr(scanner, "uses_classifier", False) and ML_CLASSIFIER_POSITIVE_LABEL:
                mc = ML_CLASSIFIER_POSITIVE_LABEL
            row_h: Dict[str, Any] = {
                "file_path": file_path,
                "line_start": snippet["line_start"],
                "line_end": snippet["line_end"],
                "code": snippet["code"][:200],
                "matched_category": mc,
                "similarity_score": max(float(r["max_similarity"]), h_min_sim),
                "all_scores": r["all_scores"],
                "heuristic_sink_rescue": True,
            }
            if gemini_on:
                row_h["full_code"] = snippet["code"]
            vulnerable_snippets.append(row_h)
            included_keys.add(key)
            added_h += 1
        if added_h:
            print(
                f"ℹ Heuristic sink rescue: +{added_h} snippet(s) "
                f"(strcpy/scanf/… present, ML would have skipped → "
                f"{'Gemini' if gemini_on else 'direct report'})"
            )

    print(f"✓ Scanned {total} snippets (repo percentile cutoff={cutoff:.3f})")
    print(
        f"✓ Found {len(vulnerable_snippets)} suspicious snippet(s) after ML + heuristic filtering"
    )

    if gemini_on and vulnerable_snippets:
        print(
            f"Running Gemini binary verification on {len(vulnerable_snippets)} candidate(s)..."
        )
        vulnerable_snippets = _filter_findings_with_gemini(
            vulnerable_snippets, progress_callback
        )
    elif gemini_on and not vulnerable_snippets:
        print("Gemini verification skipped (no ML candidates).")

    vulnerable_snippets = _expand_findings_with_call_sites(vulnerable_snippets, files)

    return vulnerable_snippets


def consolidate_findings(vulnerable_snippets: List[Dict]) -> Dict:
    """
    Group findings by file and line number.
    
    Returns:
    {
        "files": [
            {
                "file_path": "src/main.c",
                "vulnerabilities": [
                    {
                        "line_start": 42,
                        "line_end": 52,
                        "category": "buffer_overflow",
                        "vulnerability_type": "stack_buffer_overflow",
                        "similarity_score": 0.78
                    }
                ]
            }
        ],
        "summary": {
            "total_files_with_vulns": 3,
            "total_vulnerabilities": 12,
            "by_category": {...}
        }
    }
    """
    # Group by file
    files_dict = {}
    
    for finding in vulnerable_snippets:
        file_path = finding["file_path"]
        
        if file_path not in files_dict:
            files_dict[file_path] = []
        
        vtype = (finding.get("vulnerability_type") or finding["matched_category"] or "").strip()
        vent: Dict[str, Any] = {
            "line_start": finding["line_start"],
            "line_end": finding["line_end"],
            "category": finding["matched_category"],
            "vulnerability_type": vtype or finding["matched_category"],
            "similarity_score": round(finding["similarity_score"], 3),
            "code_preview": finding["code"],
        }
        if finding.get("finding_role"):
            vent["finding_role"] = finding["finding_role"]
        if finding.get("sink_function"):
            vent["sink_function"] = finding["sink_function"]
        if finding.get("sink_line_start") is not None:
            vent["sink_line_start"] = int(finding["sink_line_start"])
            vent["sink_line_end"] = int(
                finding.get("sink_line_end", finding["sink_line_start"])
            )
        files_dict[file_path].append(vent)
    
    # Count by category (prefer Gemini label when present)
    category_counts = {}
    for snippet in vulnerable_snippets:
        cat = (snippet.get("vulnerability_type") or snippet["matched_category"] or "").strip()
        if not cat:
            cat = "unspecified"
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Build output
    files_output = []
    for file_path, vulns in sorted(files_dict.items()):
        files_output.append({
            "file_path": file_path,
            "vulnerability_count": len(vulns),
            "vulnerabilities": vulns
        })
    
    return {
        "files": files_output,
        "summary": {
            "total_files_with_vulns": len(files_dict),
            "total_vulnerabilities": len(vulnerable_snippets),
            "by_category": category_counts
        }
    }


def _context_line_blank(text: str) -> bool:
    return not (text or "").strip()


def _build_finding_code_context(
    lines: List[str],
    line_start: int,
    line_end: int,
    context_before: Optional[int] = None,
    context_after: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Build ~5+5 line window (1-based) for frontend blame-style view.
    Collapses runs of 3+ blank lines into a single ellipsis row.
    """
    if context_before is None:
        context_before = max(0, int(os.environ.get("REPORT_CONTEXT_LINES_BEFORE", "5")))
    if context_after is None:
        context_after = max(0, int(os.environ.get("REPORT_CONTEXT_LINES_AFTER", "5")))
    max_rows = max(11, int(os.environ.get("REPORT_CONTEXT_MAX_ROWS", "40")))

    n = len(lines)
    if n == 0:
        return []

    try:
        a, b = int(line_start), int(line_end)
    except (TypeError, ValueError):
        return []
    if a > b:
        a, b = b, a
    anchor = a
    if anchor < 1 or anchor > n:
        return []

    lo = max(1, anchor - context_before)
    hi = min(n, anchor + context_after)
    if b > anchor + context_after:
        hi = min(n, b + context_after)

    span = hi - lo + 1
    if span > max_rows:
        half = max_rows // 2
        lo = max(1, anchor - half)
        hi = min(n, lo + max_rows - 1)
        lo = max(1, hi - max_rows + 1)

    raw_rows: List[Tuple[int, str]] = []
    for i in range(lo, hi + 1):
        text = lines[i - 1].rstrip("\n\r")
        raw_rows.append((i, text))

    out: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(raw_rows):
        num, text = raw_rows[idx]
        if _context_line_blank(text):
            j = idx + 1
            while j < len(raw_rows) and _context_line_blank(raw_rows[j][1]):
                j += 1
            run_len = j - idx
            if run_len >= 3:
                out.append(
                    {
                        "num": None,
                        "text": "",
                        "ellipsis": True,
                        "label": f"… {run_len} blank lines omitted …",
                    }
                )
                idx = j
                continue
            for k in range(idx, j):
                ln, tx = raw_rows[k]
                row: Dict[str, Any] = {
                    "num": ln,
                    "text": tx,
                    "ellipsis": False,
                }
                if a <= ln <= b:
                    row["in_vuln_range"] = True
                if ln == anchor:
                    row["anchor"] = True
                out.append(row)
            idx = j
            continue

        row = {
            "num": num,
            "text": text,
            "ellipsis": False,
        }
        if a <= num <= b:
            row["in_vuln_range"] = True
        if num == anchor:
            row["anchor"] = True
        out.append(row)
        idx += 1

    return out


def enrich_report_with_code_context(report: Dict[str, Any], files: List[Dict]) -> None:
    """Mutate report: add code_context[] to each vulnerability when source lines exist."""
    by_path: Dict[str, List[str]] = {
        f["file_path"]: f["lines"]
        for f in files
        if f.get("file_path") and f.get("lines") is not None
    }
    for fe in report.get("files") or []:
        path = (fe.get("file_path") or "").strip()
        lns = by_path.get(path)
        if not lns:
            continue
        for v in fe.get("vulnerabilities") or []:
            if not isinstance(v, dict):
                continue
            try:
                ls, le = int(v["line_start"]), int(v["line_end"])
            except (KeyError, TypeError, ValueError):
                continue
            v["code_context"] = _build_finding_code_context(lns, ls, le)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def scan_repository(
    github_url: str,
    output_file: str = "vulnerabilities.json",
    progress_callback: Optional[Any] = None,
) -> Dict:
    """
    Main function: GitHub URL → JSON report
    
    Args:
        github_url: Full GitHub repo URL
        output_file: Where to save JSON results
    
    Returns:
        Vulnerability report dict
    """
    repo_path = None

    try:
        SCAN_PROGRESS.clear()
        _emit_scan_progress(
            progress_callback,
            phase="starting",
            detail="Starting scan",
            c_cpp_files=None,
            snippets_total=None,
            snippets_scored=None,
        )

        print("=" * 60)
        print("C/C++ VULNERABILITY SCANNER")
        print("=" * 60)

        _emit_scan_progress(
            progress_callback,
            phase="cloning",
            detail=f"Cloning {github_url}",
        )
        repo_path = clone_repository(github_url)

        print("\nExtracting C/C++ files...")
        _emit_scan_progress(
            progress_callback,
            phase="extracting_sources",
            detail="Collecting .c/.cpp/.cc/.cxx (+ headers unless SCAN_INCLUDE_HEADERS=0)",
        )
        files = extract_c_cpp_files(repo_path)
        _emit_scan_progress(
            progress_callback,
            phase="sources_ready",
            c_cpp_files=len(files),
            detail=f"{len(files)} C/C++ file(s) to analyze",
        )

        if not files:
            print("⚠ No C/C++ files found in repository")
            return {
                "github_url": github_url,
                "files": [],
                "summary": {
                    "total_files_with_vulns": 0,
                    "total_vulnerabilities": 0,
                    "by_category": {}
                }
            }
        
        print("\nInitializing ML scanner...")
        _emit_scan_progress(
            progress_callback,
            phase="loading_model",
            detail="Loading classifier / embeddings",
        )
        scanner = VulnerabilityScanner()

        print("\nScanning files...")
        _emit_scan_progress(
            progress_callback,
            phase="snippet_scan",
            detail="Running ML on code windows",
        )
        vulnerable_snippets = find_vulnerable_snippets(
            scanner, files, progress_callback=progress_callback
        )

        print("\nConsolidating findings...")
        _emit_scan_progress(
            progress_callback,
            phase="consolidating",
            detail="Merging findings",
        )
        report = consolidate_findings(vulnerable_snippets)
        enrich_report_with_code_context(report, files)

        # Add metadata
        report["github_url"] = github_url
        report["total_files_scanned"] = len(files)

        # Step 6: Save to JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {output_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)
        print(f"Files scanned: {len(files)}")
        print(f"Files with vulnerabilities: {report['summary']['total_files_with_vulns']}")
        print(f"Total vulnerabilities found: {report['summary']['total_vulnerabilities']}")
        if report['summary']['by_category']:
            print("\nBreakdown by category:")
            for cat, count in sorted(report['summary']['by_category'].items()):
                print(f"  - {cat}: {count}")
        
        return report
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return {
            "error": str(e),
            "github_url": github_url,
            "files": [],
            "summary": {
                "total_files_with_vulns": 0,
                "total_vulnerabilities": 0,
                "by_category": {}
            }
        }
    
    finally:
        # Cleanup
        if repo_path and os.path.exists(repo_path):
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(repo_path)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend.py <github_url> [output_file.json]")
        print("\nExample:")
        print("  python backend.py https://github.com/torvalds/linux")
        print("  python backend.py https://github.com/curl/curl vulnerabilities.json")
        sys.exit(1)
    
    github_url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "vulnerabilities.json"
    
    scan_repository(github_url, output_file)