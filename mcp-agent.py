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

logging.basicConfig(level=logging.DEBUG)

mcp = FastMCP(host="0.0.0.0", stateless_http=True)

def bedrock_invoke_with_retry(client, max_attempts: int = 10, initial_backoff: float = 1.0, **kwargs):
    """Invoke Bedrock with exponential backoff + jitter on throttling (429 / ThrottlingException)."""
    backoff = initial_backoff
    for attempt in range(1, max_attempts + 1):
        try:
            return client.invoke_model(**kwargs)
        except Exception as e:
            err = str(e)
            is_throttle = "ThrottlingException" in err or "Too many requests" in err or "429" in err
            if not is_throttle or attempt == max_attempts:
                raise
            sleep_for = backoff + random.random() * backoff
            logging.warning("Bedrock throttled (attempt %d/%d). Retry in %.2fs", attempt, max_attempts, sleep_for)
            time.sleep(sleep_for)
            backoff *= 2

@mcp.tool()
def analyze_git_repo(repo_url: str) -> str:
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
        except git_exc.InvalidGitRepositoryError as e:
            logging.exception("InvalidGitRepositoryError cloning repo (gitpython)")
            # directory contents for diagnosis
            try:
                listing = "\n".join(os.listdir(repo_path))
            except Exception as ex:
                listing = f"<error listing dir: {ex}>"
            # fallback to subprocess to capture raw git output
            try:
                proc = subprocess.run(
                    ["git", "clone", repo_url, repo_path],
                    capture_output=True, text=True, check=False
                )
                details = f"subprocess returncode={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            except FileNotFoundError:
                logging.exception("git executable not found for subprocess fallback")
                return "Failed to clone repo: 'git' executable not found. Install Git and ensure it's on PATH (run 'git --version')."
            return f"Failed to clone repo: InvalidGitRepositoryError: {e}\nDirectory listing:\n{listing}\n\nSubprocess output:\n{details}"
        except git_exc.GitCommandError as e:
            # Git command ran but failed (captures stderr/status)
            stderr = getattr(e, "stderr", None)
            logging.exception("GitCommandError cloning repo")
            details = f"{e}"
            if stderr:
                details += f"\nstderr: {stderr}"
            return f"Failed to clone repo: {details}"
        except FileNotFoundError as e:
            # Typically means the 'git' executable is not found on PATH
            logging.exception("git executable not found")
            return "Failed to clone repo: 'git' executable not found. Install Git and ensure it's on PATH (run 'git --version')."
        except Exception as e:
            logging.exception("Unexpected error cloning repo")
            return f"Failed to clone repo: {e.__class__.__name__}: {e}"

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
            model_id = os.getenv("BEDROCK_MODEL_ID", "meta.llama3-70b-instruct-v1:0")  # set model id in env
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
            response = bedrock_invoke_with_retry(
                bedrock,
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json",
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

                for i, chunk_text in enumerate(chunks, start=1):
                    part_prompt_instructions = (
                        f"PART {i}/{len(chunks)}\n"
                        "You are an automated code reviewer. For the files below, produce a short JSON object with keys:\n"
                        "  summary: short plain-text summary\n"
                        "  issues: list of short issue descriptions\n"
                        "  recommendations: list of short actionable suggestions\n\n"
                        "Return ONLY valid JSON.\n\n"
                    )

                    # Ensure part prompt does not exceed model token budget (heuristic ~4 chars/token)
                    available_tokens = max(1, MODEL_TOKEN_LIMIT - TOKEN_RESERVE)
                    def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
                        max_chars = max(64, max_tokens * 4)
                        if len(text) <= max_chars:
                            return text
                        # try to cut at a newline boundary near the truncation point for nicer splits
                        cut = text.rfind("\n", 0, max_chars)
                        if cut <= 0:
                            cut = max_chars
                        return text[:cut] + "\n\n...TRUNCATED...\n\n"

                    # build final part prompt with truncation if needed
                    instr_len_tokens = estimate_tokens(part_prompt_instructions)
                    allowed_tokens_for_chunk = max(10, available_tokens - instr_len_tokens)
                    safe_chunk_text = truncate_text_to_tokens(chunk_text, allowed_tokens_for_chunk)
                    part_prompt = part_prompt_instructions + safe_chunk_text

                    # include model-specific max-output tokens so models don't default to tiny outputs
                    max_out = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
                    if "meta.llama3" in model_id or "llama3" in model_id:
                        # meta llama expects max_output_tokens
                        part_payload = {"prompt": part_prompt}
                    elif "anthropic" in model_id or "claude" in model_id:
                        part_payload = {"input": part_prompt, "max_tokens_to_sample": max_out}
                    else:
                        # other runtimes may accept max_tokens_to_sample
                        part_payload = {"prompt": part_prompt, "max_tokens_to_sample": max_out}

                    try:
                        part_fn = os.path.join(payload_dir, f"part_{i:03d}.json")
                        with open(part_fn, "w", encoding="utf-8") as pf:
                            json.dump(part_payload, pf, indent=2, ensure_ascii=False)
                    except Exception:
                        logging.exception("Failed to save part payload")

                    try:
                        resp = bedrock_invoke_with_retry(
                            bedrock,
                            modelId=model_id,
                            body=json.dumps(part_payload),
                            contentType="application/json",
                            accept="application/json",
                        )
                        body_obj = resp.get("body")
                        if hasattr(body_obj, "read"):
                            decoded = body_obj.read().decode("utf-8", "replace")
                        else:
                            decoded = str(body_obj or "")

                        # ensure payload_dir exists before saving responses
                        try:
                            os.makedirs(payload_dir, exist_ok=True)
                        except Exception:
                            logging.exception("Failed to ensure payload_dir exists")

                        # Save raw response immediately so it's available even if parsing fails
                        try:
                            raw_fn = os.path.join(payload_dir, f"part_{i:03d}_raw.txt")
                            with open(raw_fn, "w", encoding="utf-8") as rf:
                                rf.write(decoded or "")
                            logging.debug("Saved raw part response: %s", raw_fn)
                        except Exception:
                            logging.exception("Failed to save raw part response")

                        parsed = None
                        try:
                            parsed = json.loads(decoded)
                        except Exception:
                            m = re.search(r"\{.*\}", decoded, re.S)
                            if m:
                                try:
                                    parsed = json.loads(m.group(0))
                                except Exception:
                                    parsed = {"raw": decoded.strip()}
                            else:
                                parsed = {"raw": decoded.strip()}

                        # persist the part response (raw + parsed) to a text file for inspection
                        try:
                            resp_fn = os.path.join(payload_dir, f"part_{i:03d}_response.txt")
                            with open(resp_fn, "w", encoding="utf-8") as rf:
                                rf.write("=== RAW RESPONSE ===\n")
                                rf.write(decoded or "")
                                rf.write("\n\n=== PARSED JSON ===\n")
                                try:
                                    rf.write(json.dumps(parsed, indent=2, ensure_ascii=False))
                                except Exception:
                                    rf.write(str(parsed))
                            logging.debug("Saved Bedrock part response: %s", resp_fn)
                        except Exception:
                            logging.exception("Failed to save part response")

                        analyses.append(parsed)
                        # small delay between per-chunk calls to reduce throttle risk
                        try:
                            time.sleep(per_chunk_delay)
                        except Exception:
                            time.sleep(0.5)
                    except Exception as ex:
                        logging.exception("Per-chunk Bedrock call failed")
                        analyses.append({"error": str(ex)})
                        # backoff pause after error to avoid immediate retries
                        try:
                            time.sleep(per_chunk_delay)
                        except Exception:
                            time.sleep(0.5)

                # Build a compact / safe synth prompt that keeps total tokens below the model limit
                # summarize per-chunk analyses to reduce prompt length
                summarized = []
                for a in analyses:
                    entry = {}
                    # prefer explicit keys if present, else try to extract from raw generation
                    if isinstance(a, dict):
                        entry["summary"] = (a.get("summary") or a.get("generation") or "")[:1000]
                        issues = a.get("issues") or []
                        if isinstance(issues, list):
                            entry["issues"] = issues[:5]
                        else:
                            entry["issues"] = [str(issues)[:200]]
                        recs = a.get("recommendations") or []
                        if isinstance(recs, list):
                            entry["recommendations"] = recs[:5]
                        else:
                            entry["recommendations"] = [str(recs)[:200]]
                    else:
                        entry["summary"] = str(a)[:1000]
                        entry["issues"] = []
                        entry["recommendations"] = []
                    summarized.append(entry)

                synth_inputs_trimmed = json.dumps(summarized, indent=2, ensure_ascii=False)
                # ensure synth prompt stays within token budget
                
                formatted_prompt = f"""
                    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
                    You are given multiple short JSON analyses. Combine them into a single plain-text report with the following sections:

                    Overall Summary:
                    Write 1–2 concise paragraphs summarizing the key findings.

                    Top Issues:
                    List the most critical issues in this format:
                    1. <Short issue title — file/path:line — 1–2 sentence explanation>

                    Remediation Plan:
                    Provide actionable steps in this format:
                    1. <Concrete step>

                    JSON Inputs:
                    {synth_inputs_trimmed}
                    Produce the report now.
                    <|eot_id|>
                    <|start_header_id|>assistant<|end_header_id|>
                    """

                # available tokens for synth body (reserve output tokens)
                synth_allowed_tokens = max(64, MODEL_TOKEN_LIMIT - TOKEN_RESERVE - estimate_tokens(formatted_prompt))
                # truncate the JSON dump if necessary (heuristic)
                max_synth_chars = max(256, synth_allowed_tokens * 4)
                if len(synth_inputs_trimmed) > max_synth_chars:
                    synth_inputs_trimmed = synth_inputs_trimmed[:max_synth_chars].rsplit("\n", 1)[0]

                synth_prompt = formatted_prompt.replace("{synth_inputs_trimmed}", synth_inputs_trimmed)

                # build and persist synthesis payload (include model-specific max output)
                synthesized = ""   # ensure variable always exists
                max_out = int(os.getenv("BEDROCK_MAX_TOKENS", "8194"))
                synth_payload = {"prompt": synth_prompt}

                try:
                    synth_fn = os.path.join(payload_dir, "synth_payload.json")
                    with open(synth_fn, "w", encoding="utf-8") as sf:
                        json.dump(synth_payload, sf, indent=2, ensure_ascii=False)
                except Exception:
                    logging.exception("Failed to save synth payload")

                # invoke bedrock for final synthesis (with retry wrapper)
                try:
                    # small pause to reduce throttle risk
                    try:
                        time.sleep(5.0)
                    except Exception:
                        pass

                    resp = bedrock.invoke_model(
                        modelId="meta.llama3-70b-instruct-v1:0",  # or llama3-8b-instruct-v1:0
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(synth_payload)
                    )

                    body_obj = resp.get("body")
                    if hasattr(body_obj, "read"):
                        synthesized = body_obj.read().decode("utf-8", "replace")
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
                final_report = [
                    "=== Code Review Report ===",
                    "",
                    "=== Initial Report (Linting + Complexity) ===",
                    report.strip(),
                    "",
                    "=== Per-Repo Analysis (Bedrock) ===",
                ]

                # Add per-repo analyses (JSON) with formatting
                for i, analysis in enumerate(analyses, start=1):
                    final_report.append(f"--- Analysis Part {i} ---")
                    final_report.append("")
                    final_report.append(json.dumps(analysis, indent=2, ensure_ascii=False))
                    final_report.append("")

                final_report += [
                    "=== Bedrock Model Summary ===",
                    "",
                    summary,
                    "",
                    "=== Final Synthesized Report ===",
                    "",
                    synthesized.strip(),
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

def safe_bedrock_invoke(client, modelId: str, payload_candidates: list, contentType="application/json", accept="application/json", **invoke_kwargs):
    """
    Try a list of payload dicts until one succeeds. On Malformed input / extraneous key errors
    try the next candidate. Uses bedrock_invoke_with_retry for retry/backoff.
    """
    last_exc = None
    for idx, p in enumerate(payload_candidates, start=1):
        try:
            logging.debug("Trying bedrock payload variant %d for model %s (keys=%s)", idx, modelId, list(p.keys()))
            resp = bedrock_invoke_with_retry(
                client,
                modelId=modelId,
                body=json.dumps(p),
                contentType=contentType,
                accept=accept,
                **invoke_kwargs,
            )
            # success
            return resp, p
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # if error indicates malformed input / extraneous key, try next candidate
            if "malformed input" in msg or "extraneous key" in msg or "validationexception" in msg or "not permitted" in msg:
                logging.warning("Payload variant %d rejected by model (%s). Trying next payload variant.", idx, modelId)
                continue
            # otherwise, surface the error immediately
            raise
    # exhausted candidates
    logging.exception("All payload candidates failed for model %s", modelId)
    raise last_exc if last_exc is not None else RuntimeError("No payload candidates provided")

if __name__ == "__main__":
    logging.info("Starting MCP agent (streamable-http) on 0.0.0.0:8000")
    # adjust transport/host/port if you need a different setup
    mcp.run(transport="streamable-http")


