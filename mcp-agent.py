from mcp.server.fastmcp import FastMCP
import git
from git import Repo, exc as git_exc
import os
import subprocess
import boto3
import json
import tempfile
import shutil
import logging
import re
import time
import random
import uuid
from collections import deque
import io
import concurrent.futures
import threading
from typing import Optional, Dict
import hmac
import hashlib
import requests

logging.basicConfig(level=logging.DEBUG)

mcp = FastMCP(host="0.0.0.0", stateless_http=True)

#Session ID for context
session_id = str(uuid.uuid4())


def bedrock_converse_with_retry(client, model_id: str, messages: list, inference_config: dict = None,
                                max_attempts: int = 10, initial_backoff: float = 1.0, session_id: str = None):
    """
    Converse with Bedrock using exponential backoff + jitter on throttling (429 / ThrottlingException).
    Supports multi-turn conversations via requestMetadata.sessionId.
    """
    backoff = initial_backoff
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.converse(
                modelId=model_id,
                requestMetadata={"sessionId": session_id} if session_id else None,
                messages=messages,
                inferenceConfig=inference_config or {
                    "temperature": 0.7,
                    "maxTokens": 512
                }
            )
            return response
        except Exception as e:
            err = str(e)
            is_throttle = "ThrottlingException" in err or "Too many requests" in err or "429" in err
            if not is_throttle or attempt == max_attempts:
                raise
            sleep_for = backoff + random.random() * backoff
            logging.warning("Bedrock throttled (attempt %d/%d). Retry in %.2fs", attempt, max_attempts, sleep_for)
            time.sleep(sleep_for)
            backoff *= 2


# --- new: streaming-friendly wrapper that returns decoded text with retry/backoff ---

def bedrock_converse_stream_with_retry(client, model_id: str, messages: list = None, inference_config: dict = None,
                                       max_attempts: int = 10, initial_backoff: float = 1.0, session_id: str = None,
                                       encoding: str = "utf-8"):
    """
    Try to get a streaming 'converse' response when available, falling back to non-streaming converse or invoke_model.
    Returns a dict-like response with key 'body' containing a bytes-like file object (io.BytesIO) so callers can
    call resp.get('body').read() as before.
    """
    backoff = initial_backoff
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            # 1) prefer streaming converse if available
            if hasattr(client, "converse_stream"):
                resp = client.converse_stream(
                    modelId=model_id,
                    requestMetadata={"sessionId": session_id} if session_id else None,
                    messages=messages,
                    inferenceConfig=inference_config or {"temperature": 0.7, "maxTokens": 512}
                )
                pieces = []
                # resp["stream"] is expected to be an iterable of events
                for event in resp.get("stream", []):
                    # common bedrock streaming event shapes include contentBlockDelta / text deltas
                    if isinstance(event, dict):
                        if "contentBlockDelta" in event:
                            delta = event["contentBlockDelta"].get("delta", {})
                            if "text" in delta:
                                pieces.append(delta["text"].encode(encoding, errors="replace"))
                        elif "text" in event:
                            pieces.append(str(event["text"]).encode(encoding, errors="replace"))
                        elif "messageStop" in event:
                            break
                raw = b"".join(pieces)
                return {"body": io.BytesIO(raw)}

            # 2) try non-streaming converse (multi-turn chat)
            if hasattr(client, "converse"):
                resp = client.converse(
                    modelId=model_id,
                    requestMetadata={"sessionId": session_id} if session_id else None,
                    messages=messages,
                    inferenceConfig=inference_config or {"temperature": 0.7, "maxTokens": 512}
                )
                # Many SDKs return response with 'body' that may be bytes or a str or file-like
                body_obj = resp.get("body", None)
                if body_obj is None:
                    # attempt to synthesize text from known keys
                    text = ""
                    if isinstance(resp, dict):
                        # try common keys
                        text = resp.get("completion") or resp.get("output") or resp.get("generated_text") or ""
                    return {"body": io.BytesIO(str(text).encode(encoding, errors="replace"))}
                # if body is file-like, try to read it
                try:
                    if hasattr(body_obj, "read"):
                        raw = body_obj.read()
                        if isinstance(raw, str):
                            raw = raw.encode(encoding, errors="replace")
                        return {"body": io.BytesIO(raw)}
                    if isinstance(body_obj, bytes):
                        return {"body": io.BytesIO(body_obj)}
                    return {"body": io.BytesIO(str(body_obj).encode(encoding, errors="replace"))}
                except Exception:
                    return {"body": io.BytesIO(str(body_obj).encode(encoding, errors="replace"))}

            # 3) final fallback: invoke_model (common boto3 runtime API)
            if hasattr(client, "invoke_model"):
                resp = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps({"messages": messages}) if messages is not None else json.dumps({}),
                    contentType="application/json",
                    accept="application/json",
                    requestMetadata={"sessionId": session_id} if session_id else None,
                )
                body_obj = resp.get("body")
                # try streaming read if available
                try:
                    if hasattr(body_obj, "read"):
                        pieces = []
                        # read in loop defensively
                        while True:
                            chunk = body_obj.read(4096)
                            if not chunk:
                                break
                            if isinstance(chunk, str):
                                pieces.append(chunk.encode(encoding, errors="replace"))
                            else:
                                pieces.append(chunk)
                        raw = b"".join(pieces)
                        return {"body": io.BytesIO(raw)}
                    else:
                        if isinstance(body_obj, bytes):
                            return {"body": io.BytesIO(body_obj)}
                        return {"body": io.BytesIO(str(body_obj or "").encode(encoding, errors="replace"))}
                except Exception:
                    return {"body": io.BytesIO(str(body_obj or "").encode(encoding, errors="replace"))}

            # If client supports none of the above, raise
            raise RuntimeError("Bedrock client does not support converse_stream, converse, or invoke_model")

        except Exception as e:
            last_exc = e
            err = str(e)
            is_throttle = "ThrottlingException" in err or "Too many requests" in err or "429" in err
            if not is_throttle or attempt == max_attempts:
                raise
            sleep_for = backoff + random.random() * backoff
            logging.warning("Bedrock converse throttled (attempt %d/%d). Retry in %.2fs", attempt, max_attempts, sleep_for)
            time.sleep(sleep_for)
            backoff *= 2

    raise last_exc if last_exc else RuntimeError("bedrock_converse_stream_with_retry failed without exception")


@mcp.tool()
def analyze_git_repo(repo_url: str,
                     pr_number: Optional[int] = None,
                     pr_patch: Optional[str] = None,
                     pr_changes: Optional[Dict[str, str]] = None) -> str:
    """
    Analyze a repo. Optional PR parameters:
      - pr_number: fetch and checkout pull/{N}/head
      - pr_patch: apply a patch string to the cloned repo
      - pr_changes: dict of path->content to write and commit
    """
    # use a temp directory to avoid collisions and be cross-platform
    # Option for a Webhook interation
    repo_path = tempfile.mkdtemp(prefix="code_review_repo_")

    # Ensure target dir is empty (previous failed clone may have left files)
    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        os.makedirs(repo_path, exist_ok=True)
    except Exception as ex:
        logging.exception("Failed to prepare repo_path for clone")
        return f"Failed to prepare clone directory: {ex}"

    try:
        try:
            Repo.clone_from(repo_url, repo_path)
        except Exception as clone_exc:
            logging.exception("Error cloning repo")
            return f"Failed to clone repo: {clone_exc}"

        # --- apply PR / patch / file-changes if provided ---
        def apply_pr_to_clone():
            try:
                repo = Repo(repo_path)
                if pr_number is not None:
                    fetch_ref = f"pull/{int(pr_number)}/head:pr-{int(pr_number)}"
                    logging.info("Fetching PR ref %s", fetch_ref)
                    try:
                        repo.remotes.origin.fetch(fetch_ref)
                        repo.git.checkout(f"pr-{int(pr_number)}")
                        return True, f"Checked out PR #{pr_number}"
                    except Exception as e:
                        logging.warning("Fetching PR ref failed: %s", e)
                        # fallback to downloading patch
                        pr_patch_url = None
                        try:
                            # try common GitHub patch URL
                            owner = repo.remotes.origin.url.split("/")[-2]
                        except Exception:
                            pr_patch_url = None
                        if pr_patch_url is None:
                            pr_patch_url = None
                if pr_patch:
                    patch_file = os.path.join(repo_path, "pr.patch")
                    with open(patch_file, "w", encoding="utf-8") as pf:
                        pf.write(pr_patch)
                    res = subprocess.run(["git", "apply", "--index", patch_file], cwd=repo_path, capture_output=True, text=True)
                    if res.returncode != 0:
                        logging.warning("git apply failed: %s", res.stderr)
                        return False, f"git apply failed: {res.stderr}"
                    repo.index.commit(f"Apply PR patch")
                    return True, "Applied patch"
                if pr_changes:
                    for path, content in pr_changes.items():
                        full = os.path.join(repo_path, path)
                        os.makedirs(os.path.dirname(full), exist_ok=True)
                        with open(full, "w", encoding="utf-8") as fh:
                            fh.write(content)
                        repo.git.add(path)
                    repo.index.commit("Apply PR changes via webhook")
                    return True, "Wrote pr_changes and committed"
            except Exception as ex:
                logging.exception("Failed to apply PR changes")
                return False, str(ex)
            return False, "no pr input provided"

        applied, info = apply_pr_to_clone()
        logging.info("PR apply result: %s %s", applied, info)

        # --- New: analyze PR diff directly with Bedrock ---
        def _get_pr_diff_text():
            try:
                repo = Repo(repo_path)
                if pr_patch:
                    return pr_patch
                if pr_changes:
                    parts = []
                    for p, content in pr_changes.items():
                        parts.append(f"--- a/{p}\n+++ b/{p}\n")
                        parts.append(content)
                    return "\n\n".join(parts)
                if pr_number is not None:
                    pr_branch = f"pr-{pr_number}"
                    try:
                        repo.remotes.origin.fetch()
                        repo.remotes.origin.fetch(f"pull/{int(pr_number)}/head:{pr_branch}")
                        repo.remotes.origin.fetch("main")
                        diff = repo.git.diff(f"origin/main...{pr_branch}")
                        if diff and diff.strip():
                            return diff
                    except Exception:
                        pass
                # fallback: show HEAD diff
                try:
                    diff = repo.git.diff("HEAD~10..HEAD")
                    if diff and diff.strip():
                        return diff
                except Exception:
                    pass
                try:
                    return repo.git.show("HEAD")
                except Exception:
                    return ""
            except Exception:
                return ""

        MODEL_TOKEN_LIMIT = int(os.getenv("MODEL_TOKEN_LIMIT", os.getenv("BEDROCK_MODEL_TOKEN_LIMIT", "200000")))
        TOKEN_RESERVE = int(os.getenv("TOKEN_RESERVE", "512"))
        AVG_CHARS_PER_TOKEN = float(os.getenv("AVG_CHARS_PER_TOKEN", "3.0"))

        def analyze_pr_with_bedrock(diff_text: str, pr_id: Optional[int] = None):
            region = os.getenv("AWS_REGION", "us-east-1")
            model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
            bedrock = boto3.client("bedrock-runtime", region_name=region)
            if not diff_text:
                return None
            # small local truncator to avoid exceeding model token window
            def _truncate_for_model(text: str, max_chars: int):
                if len(text) <= max_chars:
                    return text
                cut = text.rfind("\n", 0, max_chars)
                if cut <= 0:
                    cut = max_chars
                return text[:cut] + "\n\n...TRUNCATED...\n\n"

            requested_out = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
            max_out = clamp_tokens_for_model(model_id or "", requested_out) if 'clamp_tokens_for_model' in globals() else requested_out
            # compute safe body size in chars from token heuristic
            safe_chars = int(200000)
            diff_trimmed = _truncate_for_model(diff_text, safe_chars)

            # save trimmed diff to payloads for inspection
            try:
                payload_dir = os.path.join(repo_path, "bedrock_payloads")
                os.makedirs(payload_dir, exist_ok=True)
                pr_diff_fn = os.path.join(payload_dir, f"pr_{pr_id or 'unknown'}_diff.txt")
                with open(pr_diff_fn, "w", encoding="utf-8") as pf:
                    pf.write(diff_trimmed)
                logging.info("Saved trimmed PR diff to %s", pr_diff_fn)
            except Exception:
                logging.debug("Failed to save trimmed PR diff", exc_info=True)

            pr_label = f"PR #{pr_id}" if pr_id else "PR changes"
            pr_prompt = (
                f"{pr_label}: Analyze the proposed changes and their impact on the repository.\n\n"
                "Return ONLY valid JSON with keys: summary (short), impacted_files (list), "
                "issues (list of {title,file,line,desc}), recommendations (list of short actionable steps).\n\n"
                "Be specific to the diff context provided.\n\n"
            )

            try:
                resp = bedrock_converse_stream_with_retry(
                    bedrock,
                    model_id=model_id,
                    messages=[{"role": "user", "content": [{"text": pr_prompt + '\n\nDIFF:\n' + diff_trimmed}]}],
                    inference_config={"temperature": 0.5, "maxTokens": max_out},
                    session_id=session_id
                )
                body_obj = resp.get("body") if isinstance(resp, dict) else resp
                if hasattr(body_obj, "read"):
                    decoded = body_obj.read().decode("utf-8", errors="replace")
                elif isinstance(body_obj, bytes):
                    decoded = body_obj.decode("utf-8", errors="replace")
                else:
                    decoded = str(body_obj or "")
            except Exception as e:
                logging.exception("PR Bedrock analysis failed")
                return {"error": str(e)}

            try:
                parsed = json.loads(decoded)
            except Exception:
                parsed = {"generation": decoded.strip()}
            parsed.setdefault("repo_path", repo_path)
            parsed.setdefault("pr_number", pr_number)
            return parsed

        # Run quick PR-focused analysis and add to the analyses/context so synth uses it
        pr_section = ""
        try:
            pr_diff = _get_pr_diff_text()
            pr_result = analyze_pr_with_bedrock(pr_diff, pr_number)
            pr_section = pr_result
            if pr_result:
                # prepend PR analysis so synth sees it first
                analyses = locals().get("analyses", [])
                if isinstance(analyses, list):
                    analyses.insert(0, pr_result)
                if 'recent_context' in locals():
                    recent_context.appendleft(pr_result) if hasattr(recent_context, "appendleft") else recent_context.append(pr_result)
        except Exception:
            logging.exception("Failed to run PR-specific Bedrock analysis")

        # Run flake8 for linting (if installed)
        if shutil.which("flake8"):
            lint_proc = subprocess.run(["flake8", repo_path], capture_output=True, text=True)
            lint_out = lint_proc.stdout or lint_proc.stderr or ""
        else:
            lint_out = "flake8 not found in PATH — skipping linting.\n"

        # Run radon for code complexity (if installed)
        if shutil.which("radon"):
            radon_proc = subprocess.run(["radon", "cc", repo_path, "-s"], capture_output=True, text=True)
            radon_out = radon_proc.stdout or radon_proc.stderr or ""
        else:
            radon_out = "radon not found in PATH — skipping complexity analysis.\n"

        # Combine results
        report = "\n--- Linting Issues (flake8) ---\n"
        report += lint_out or "No linting issues found.\n"
        report += "\n--- Code Complexity (radon) ---\n"
        report += radon_out or "No complexity issues found.\n"

        # --- Call Bedrock model to summarize / provide suggestions ---
        try:
            region = os.getenv("AWS_REGION", "us-east-1")
            model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")  # set model id in env
            if not model_id:
                report += "\n\n--- Bedrock skipped: no BEDROCK_MODEL_ID set ---\n"
                return report

            bedrock = boto3.client("bedrock-runtime", region_name=region)
            prompt = "Summarize the following code analysis report and provide actionable suggestions:\n\n" + report

            # Build model-specific payload (meta llama instruct expects "prompt")
            if "meta.llama3" in model_id or "llama3" in model_id:
                # Meta Llama3 instruct models expect a top-level "prompt" key.
                # Do not send unsupported keys like max_tokens_to_sample here.
                payload = {"prompt": prompt}
            elif "anthropic" in model_id or "claude" in model_id:
                payload = {"input": prompt, "max_tokens_to_sample": int(os.getenv("BEDROCK_MAX_TOKENS", "512"))}
            else:
                payload = {"prompt": prompt, "max_tokens_to_sample": int(os.getenv("BEDROCK_MAX_TOKENS", "512"))}

            # Example with explicit model and payload
            logging.info(f"Calling Bedrock model {model_id} with payload: {json.dumps(payload)}")
            response = bedrock_converse_stream_with_retry(
                bedrock,
                model_id=model_id,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
                inference_config={
                    "temperature": 0.7,
                    "maxTokens": 512
                },
                session_id=session_id  # Optional for context
            )

            # Read and decode response body robustly
            body_obj = response.get("body")
            body_bytes = None
            try:
                # most boto3 Bedrock responses expose a streaming body with .read()
                if hasattr(body_obj, "read"):
                    body_bytes = body_obj.read()
                else:
                    body_bytes = body_obj
            except Exception:
                body_bytes = body_obj

            if isinstance(body_bytes, bytes):
                decoded = body_bytes.decode("utf-8", errors="replace")
            else:
                decoded = str(body_bytes or "")

            # Try to parse JSON, handle common Bedrock shapes (completion, outputs, output, generated_text)
            summary = ""
            try:
                parsed = json.loads(decoded)
                if isinstance(parsed, dict):
                    if "completion" in parsed:
                        summary = parsed.get("completion", "")
                    elif "outputs" in parsed:
                        texts = []
                        for out in parsed.get("outputs", []):
                            for c in out.get("content", []):
                                if isinstance(c, dict):
                                    if "text" in c:
                                        texts.append(c["text"])
                                    elif c.get("type") == "output_text" and "text" in c:
                                        texts.append(c["text"])
                                elif isinstance(c, str):
                                    texts.append(c)
                        summary = "\n".join(texts)
                    else:
                        summary = parsed.get("output") or parsed.get("generated_text") or parsed.get("results") or json.dumps(parsed)
                else:
                    summary = str(parsed)
            except Exception:
                # not JSON or unexpected shape — use raw decoded text
                summary = decoded

            summary = (summary or "").strip()

            report += f"\n--- Bedrock Model Summary ---\n{summary}\n"

        except Exception as e:
            logging.exception("Error calling Bedrock model")
            report += f"\n\n--- Bedrock analysis skipped: {e.__class__.__name__}: {e} ---\n"

        # --- New per-repo analysis using Bedrock ---
        try:
            # Build a list of text files to send (skip .git, node_modules, binaries)
            def collect_text_file_entries(root_path, max_total_bytes=2_000_000):
                entries = []
                total = 0
                skip_dirs = {".git", "node_modules", "__pycache__"}
                for root, dirs, files in os.walk(root_path):
                    dirs[:] = [d for d in dirs if d not in skip_dirs]
                    for fname in files:
                        full = os.path.join(root, fname)
                        rel = os.path.relpath(full, root_path)
                        try:
                            with open(full, "rb") as fh:
                                raw = fh.read()
                        except Exception:
                            continue
                        if b"\0" in raw:
                            continue
                        try:
                            text = raw.decode("utf-8")
                        except Exception:
                            try:
                                text = raw.decode("latin-1")
                            except Exception:
                                continue
                        entry = f"### {rel}\n{text}\n"
                        size = len(entry.encode("utf-8"))
                        if total + size > max_total_bytes:
                            return entries, True
                        entries.append(entry)
                        total += size
                return entries, False

            file_entries, truncated = collect_text_file_entries(repo_path, max_total_bytes=int(os.getenv("REPO_MAX_BYTES", "2000000")))

            # Chunking / model limits (defaults; override via env)
            CHUNK_BYTES = int(os.getenv("REPO_CHUNK_BYTES", "30000"))   # smaller default
            MODEL_TOKEN_LIMIT = int(os.getenv("MODEL_TOKEN_LIMIT", os.getenv("BEDROCK_MODEL_TOKEN_LIMIT", "8192")))
            TOKEN_RESERVE = int(os.getenv("TOKEN_RESERVE", "512"))
            AVG_CHARS_PER_TOKEN = float(os.getenv("AVG_CHARS_PER_TOKEN", "3.0"))  # conservative heuristic

            # Heuristic token estimation using configurable avg chars/token (more conservative)
            def estimate_tokens(text: str) -> int:
                return max(1, int(len(text) / max(1.0, AVG_CHARS_PER_TOKEN)))

            def split_chunk_by_token_limit(text: str, max_tokens: int):
                # Hard char cap based on heuristic to avoid underestimation
                max_chars = int(max_tokens * AVG_CHARS_PER_TOKEN)
                if len(text) <= max_chars:
                    yield text
                    return
                # iterative split: prefer newline boundaries near mid
                parts = [text]
                while parts:
                    cur = parts.pop(0)
                    if len(cur) <= max_chars:
                        yield cur
                        continue
                    mid = min(len(cur), max_chars)
                    # try to split at a newline before the char cap, else at mid
                    nl = cur.rfind("\n", 0, mid)
                    if nl <= 0:
                        nl = mid
                    a, b = cur[:nl], cur[nl:].lstrip("\n")
                    parts.insert(0, b)
                    parts.insert(0, a)

            # Build raw chunks from collected file entries (size-limited)
            chunks = []
            cur = []
            cur_len = 0
            for e in file_entries:
                elen = len(e.encode("utf-8"))
                if elen > CHUNK_BYTES:
                    b = e.encode("utf-8")
                    for j in range(0, len(b), CHUNK_BYTES):
                        chunks.append(b[j:j + CHUNK_BYTES].decode("utf-8", "ignore"))
                else:
                    if cur_len + elen > CHUNK_BYTES:
                        chunks.append("\n".join(cur))
                        cur = [e]
                        cur_len = elen
                    else:
                        cur.append(e)
                        cur_len += elen
            if cur:
                chunks.append("\n".join(cur))

            # Ensure each chunk fits the model token limit (split further if necessary)
            max_chunk_tokens = MODEL_TOKEN_LIMIT - TOKEN_RESERVE
            expanded_chunks = []
            for c in chunks:
                for sub in split_chunk_by_token_limit(c, max_chunk_tokens):
                    expanded_chunks.append(sub)
            chunks = expanded_chunks

            if not file_entries:
                report += "\n\n--- Repo had no text files to analyze or were filtered out ---\n"
            else:
                analyses = []
                payload_dir = os.path.join(repo_path, "bedrock_payloads")
                try:
                    os.makedirs(payload_dir, exist_ok=True)
                except Exception:
                    logging.exception("Failed to create payload_dir for bedrock payloads")

                # rate limit control: seconds to wait between per-chunk Bedrock calls
                per_chunk_delay = float(os.getenv("PER_CHUNK_DELAY", "3.0"))
                # concurrency for per-chunk calls
                max_workers = max(1, int(os.getenv("PER_CHUNK_CONCURRENCY", "3")))

                # context window: keep last N parsed analyses and use them for final synthesis
                CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "15"))
                recent_context = deque(maxlen=CONTEXT_WINDOW)

                # worker that processes one chunk and returns parsed_obj (or error dict)
                def process_chunk(i, chunk_text):
                    try:
                        part_prompt_instructions = (
                            f"PART {i}/{len(chunks)}\n"
                            "You are an automated code reviewer. For the files below, produce a short JSON object with keys:\n"
                            "  summary: short plain-text summary\n"
                            "  issues: list of short issue descriptions\n"
                            "  recommendations: list of short actionable suggestions\n\n"
                            "Return ONLY valid JSON.\n\n"
                        )
                        available_tokens = max(1, MODEL_TOKEN_LIMIT - TOKEN_RESERVE)

                        def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
                            max_chars = max(64, int(max_tokens * 4))
                            if len(text) <= max_chars:
                                return text
                            cut = text.rfind("\n", 0, max_chars)
                            if cut <= 0:
                                cut = max_chars
                            return text[:cut] + "\n\n...TRUNCATED...\n\n"

                        instr_len_tokens = estimate_tokens(part_prompt_instructions)
                        allowed_tokens_for_chunk = max(10, available_tokens - instr_len_tokens)
                        safe_chunk_text = truncate_text_to_tokens(chunk_text, allowed_tokens_for_chunk)
                        part_prompt = part_prompt_instructions + safe_chunk_text

                        # include model-specific max-output tokens
                        requested_out = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
                        max_out_local = clamp_tokens_for_model(model_id or "", requested_out) if 'clamp_tokens_for_model' in globals() else requested_out

                        if "meta.llama3" in (model_id or "") or "llama3" in (model_id or ""):
                            part_payload = {"prompt": part_prompt}
                        elif "anthropic" in (model_id or "") or "claude" in (model_id or ""):
                            part_payload = {"input": part_prompt, "max_tokens_to_sample": max_out_local}
                        else:
                            part_payload = {"prompt": part_prompt, "max_tokens_to_sample": max_out_local}

                        try:
                            part_fn = os.path.join(payload_dir, f"part_{i:03d}.json")
                            with open(part_fn, "w", encoding="utf-8") as pf:
                                json.dump(part_payload, pf, indent=2, ensure_ascii=False)
                        except Exception:
                            logging.debug("Failed to save part payload", exc_info=True)

                        # call bedrock (streaming wrapper)
                        try:
                            resp = bedrock_converse_stream_with_retry(
                                bedrock,
                                model_id=model_id,
                                messages=[{"role": "user", "content": [{"text": part_prompt}]}],
                                inference_config={"temperature": 0.7, "maxTokens": max_out_local},
                                session_id=session_id
                            )
                            body_obj = resp.get("body") if isinstance(resp, dict) else resp
                            if hasattr(body_obj, "read"):
                                decoded = body_obj.read().decode("utf-8", errors="replace")
                            elif isinstance(body_obj, bytes):
                                decoded = body_obj.decode("utf-8", errors="replace")
                            else:
                                decoded = str(body_obj or "")
                        except Exception as ex:
                            logging.exception("Per-chunk Bedrock call failed (worker)")
                            return {"error": str(ex), "part_index": i}

                        try:
                            parsed = json.loads(decoded)
                        except Exception:
                            parsed = {"generation": decoded.strip()}

                        parsed.setdefault("part_index", i)
                        parsed.setdefault("total_parts", len(chunks))
                        parsed.setdefault("repo_path", repo_path)
                        return parsed
                    finally:
                        try:
                            time.sleep(per_chunk_delay * 0.25)
                        except Exception:
                            pass

                # submit tasks with bounded executor and collect results, then reassemble in order
                results_by_index = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
                    futures = {}
                    for i, chunk_text in enumerate(chunks, start=1):
                        fut = exe.submit(process_chunk, i, chunk_text)
                        futures[fut] = i
                        # space submissions slightly to avoid bursts
                        try:
                            time.sleep(per_chunk_delay / max(1, max_workers))
                        except Exception:
                            pass

                    for fut in concurrent.futures.as_completed(futures):
                        idx = futures[fut]
                        try:
                            res = fut.result()
                        except Exception as e:
                            logging.exception("Per-chunk worker raised", exc_info=True)
                            res = {"error": str(e), "part_index": idx}
                        results_by_index[idx] = res

                # append results in part order to analyses and recent_context
                for idx in range(1, len(chunks) + 1):
                    parsed_obj = results_by_index.get(idx, {"error": "missing result", "part_index": idx})
                    analyses.append(parsed_obj)
                    try:
                        recent_context.append(parsed_obj)
                    except Exception:
                        pass
                
                formatted_prompt = """
                    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
                    You are given multiple short JSON analyses in this session
                    Use the session id to access those JSON's context
                    Combine the analyses into a single plain-text report with sections:
                    
                    Overall Summary:
                      Write 1–2 concise paragraphs summarizing the key findings.
                    
                    Top Issues:
                      List the most critical issues in this format:
                      1. <Short issue title — file/path:line — 1–2 sentence explanation>
                    
                    Remediation Plan:
                      Provide actionable steps in this format:
                      1. <Concrete step>

                    Produce the report now.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """

                # build and persist synthesis payload (include model-specific max output)
                synthesized = ""   # ensure variable always exists
                max_out = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
                synth_payload = {"prompt": formatted_prompt}

                try:
                    synth_fn = os.path.join(payload_dir, "synth_payload.json")
                    with open(synth_fn, "w", encoding="utf-8") as sf:
                        json.dump(synth_payload, sf, indent=2, ensure_ascii=False)
                except Exception:
                    logging.exception("Failed to save synth payload")

                # invoke bedrock for final synthesis (with streaming retry wrapper)
                try:
                    # small pause to reduce throttle risk
                    try:
                        time.sleep(5.0)
                    except Exception:
                        pass

                    # call once (avoid duplicate network call)
                    resp = bedrock_converse_stream_with_retry(
                        bedrock,
                        model_id=model_id or "anthropic.claude-3-haiku-20240307-v1:0",
                        messages=[
                            {"role": "user", "content": [{"text": formatted_prompt }]}
                        ],
                        inference_config={"temperature": 0.3, "maxTokens": max_out},
                        session_id=session_id  # pass session id so Bedrock can use session context
                    )

                    # read body robustly
                    body_obj = resp.get("body") if isinstance(resp, dict) else resp
                    if hasattr(body_obj, "read"):
                        synthesized = body_obj.read().decode("utf-8", errors="replace")
                    elif isinstance(body_obj, bytes):
                        synthesized = body_obj.decode("utf-8", errors="replace")
                    else:
                        synthesized = str(body_obj or "")
                except Exception:
                    logging.exception("Final Bedrock synthesis failed")
                    report += "\n\n--- Bedrock synthesis failed; see agent logs ---\n"

                # Always persist the synthesized report (cleaned) so you can inspect it
                try:
                    if synthesized is None:
                        synthesized = ""
                    # simple cleaning: normalize and collapse excessive newlines
                    cleaned = str(synthesized).replace("\r\n", "\n").replace("\r", "\n")
                    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
                    os.makedirs(payload_dir, exist_ok=True)
                    synth_report_path = os.path.join(payload_dir, "synth_report.txt")
                    with open(synth_report_path, "w", encoding="utf-8") as rf:
                        rf.write(cleaned)
                    logging.info("Saved Bedrock synthesized report: %s", synth_report_path)
                except Exception:
                    logging.exception("Failed to save synthesized report")
                # use cleaned text in final report if available
                try:
                    synthesized = cleaned
                except Exception:
                    synthesized = synthesized or ""

                # --- Final report assembly ---
                # Combine all parts: initial report, per-repo analysis, Bedrock summary, and synthesis
                # include PR-specific analysis if available

                pr_section = json.dumps(pr_section, indent=2, ensure_ascii=False)

                final_report = [
                    "=== Code Review Report ===",
                    "",
                    "=== Pull Request Analysis (Bedrock) ===",
                    pr_section,
                ]

                final_report += [
                    "=== Final Synthesized Report ===",
                    "",
                    synthesized.strip(),
                    "",
                    "=== Linting and Complexity Summary ===",
                    "",
                    summary,
                    "",
                    "=== End of Report ===",
                ]
                # Join all parts with double newlines, ensuring no trailing newlines at the end
                final_report_text = "\n\n".join(final_report).strip()

                # Save the final assembled report to a file
                try:
                    report_fn = os.path.join(repo_path, "code_review_report.txt")
                    with open(report_fn, "w", encoding="utf-8") as rf:
                        rf.write(final_report_text)
                    logging.info("Saved final report to: %s", report_fn)
                except Exception:
                    logging.exception("Failed to save final report")

                return final_report_text

        except Exception as e:
            # Ensure any unexpected error inside the analysis is logged and returned as part of the report
            logging.exception("Unhandled exception in analyze_git_repo")
            try:
                report += f"\n\n--- Internal error: {e.__class__.__name__}: {e} ---\n"
            except Exception:
                report = f"Internal error: {e.__class__.__name__}: {e}"
            return report

    finally:
        # Preserve cloned repo by default. To remove after run set KEEP_PAYLOADS=0 (or false/no).
        try:
            keep = os.getenv("KEEP_PAYLOADS", "1").lower()
        except Exception:
            keep = "1"

        if keep in ("0", "false", "no"):
            try:
                shutil.rmtree(repo_path)
                logging.info("Removed temporary repo directory: %s", repo_path)
            except Exception:
                logging.exception("Failed to remove temporary repo directory")
        else:
            logging.info("Preserving cloned repo at %s (not removed). Set KEEP_PAYLOADS=0 to enable cleanup.", repo_path)

@mcp.tool()
def github_webhook(payload: dict, headers: dict = None) -> str:
    """
    Minimal GitHub webhook receiver.
    Accepts pull_request events and triggers analysis.
    """
    hdrs = headers or {}
    event = hdrs.get("X-GitHub-Event", hdrs.get("x-github-event", ""))
    if event != "pull_request":
        return f"ignored event: {event}"

    action = payload.get("action")
    if action not in ("opened", "synchronize", "reopened", "ready_for_review"):
        return f"ignored action: {action}"

    pr = payload.get("pull_request", {}) or {}
    number = pr.get("number")
    repo = payload.get("repository", {}) or {}
    clone_url = repo.get("clone_url") or repo.get("git_url") or repo.get("ssh_url")

    # Run analysis in a background thread so webhook returns quickly
    import threading
    def worker():
        try:
            analyze_git_repo(clone_url, pr_number=number)
        except Exception as e:
            logging.exception("Webhook analysis failed")
    threading.Thread(target=worker, daemon=True).start()
    return f"accepted PR #{number} for analysis"

if __name__ == "__main__":
    logging.info("Starting MCP agent (streamable-http) on 0.0.0.0:8000")
    # adjust transport/host/port if you need a different setup
    mcp.run(transport="streamable-http")

