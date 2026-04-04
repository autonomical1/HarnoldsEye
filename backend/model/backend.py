import math
import os
import sys
import json
import subprocess
import tempfile
import shutil
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
    os.environ.get("ML_MIN_ABSOLUTE_SIMILARITY", "0.86")
)

# Best category must beat the second-best by this margin (reduces random "winner" picks).
ML_MIN_CATEGORY_MARGIN = float(os.environ.get("ML_MIN_CATEGORY_MARGIN", "0.028"))

# Within a single repo scan, only keep snippets whose max similarity is at or above this
# percentile of all snippet scores (e.g. 88 → top ~12% most "suspicious" in that clone).
ML_REPO_SCORE_PERCENTILE = float(
    os.environ.get("ML_REPO_SCORE_PERCENTILE", "88")
)

# When using model.pkl classifier, softmax “confidence” is often lower than cosine scores.
ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER = float(
    os.environ.get("ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER", "0.45")
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
            batch_size = max(1, int(os.environ.get("CLASSIFIER_BATCH_SIZE", "16")))
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


def extract_snippets_from_lines(
    lines: List[str],
    file_path: str,
    snippet_size: Optional[int] = None,
) -> List[Dict]:
    """
    Extract overlapping code snippets from file lines.
    
    Returns:
    [
        {
            "code": "...",
            "line_start": 1,
            "line_end": 10,
            "file_path": "src/main.c"
        }
    ]
    """
    if snippet_size is None:
        snippet_size = max(3, int(os.environ.get("SNIPPET_WINDOW_LINES", "10")))
    stride_env = int(os.environ.get("SNIPPET_LINE_STRIDE", "0"))
    stride = stride_env if stride_env > 0 else max(1, snippet_size // 2)
    snippets = []
    
    for i in range(0, len(lines), stride):
        end_idx = min(i + snippet_size, len(lines))
        
        if end_idx - i < 3:  # Skip too-small snippets
            continue
        
        snippet_code = ''.join(lines[i:end_idx]).strip()
        
        if not snippet_code or len(snippet_code) < 10:
            continue
        
        snippets.append({
            "code": snippet_code,
            "line_start": i + 1,  # 1-indexed
            "line_end": end_idx,
            "file_path": file_path
        })
    
    return snippets


# ============================================================================
# VULNERABILITY DETECTION
# ============================================================================


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
    _emit_scan_progress(
        progress_callback,
        phase="scoring_snippets",
        snippets_total=total,
        snippets_scored=0,
        detail=f"Scoring {total} snippet window(s) from {len(files)} C/C++ file(s)",
    )

    score_results: List[Dict[str, Any]] = []
    if getattr(scanner, "uses_classifier", False) and scanner._torch_model is not None:
        texts = [s["code"] for _, s in flat]
        batch_size = max(1, int(os.environ.get("CLASSIFIER_BATCH_SIZE", "16")))
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
    cutoff = _percentile_nearest_rank(max_sims, ML_REPO_SCORE_PERCENTILE)

    min_abs = (
        ML_MIN_ABSOLUTE_SIMILARITY_CLASSIFIER
        if getattr(scanner, "uses_classifier", False)
        else ML_MIN_ABSOLUTE_SIMILARITY
    )

    vulnerable_snippets = []
    for file_path, snippet, r in scored:
        if r["max_similarity"] < min_abs:
            continue
        if r["category_margin"] < ML_MIN_CATEGORY_MARGIN:
            continue
        if r["max_similarity"] < cutoff:
            continue
        vulnerable_snippets.append({
            "file_path": file_path,
            "line_start": snippet["line_start"],
            "line_end": snippet["line_end"],
            "code": snippet["code"][:200],
            "matched_category": r["matched_category"],
            "similarity_score": r["max_similarity"],
            "all_scores": r["all_scores"],
        })

    print(f"✓ Scanned {total} snippets (repo percentile cutoff={cutoff:.3f})")
    print(f"✓ Found {len(vulnerable_snippets)} suspicious snippets after filtering")
    
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
        
        files_dict[file_path].append({
            "line_start": finding["line_start"],
            "line_end": finding["line_end"],
            "category": finding["matched_category"],
            "similarity_score": round(finding["similarity_score"], 3),
            "code_preview": finding["code"]
        })
    
    # Count by category
    category_counts = {}
    for snippet in vulnerable_snippets:
        cat = snippet["matched_category"]
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