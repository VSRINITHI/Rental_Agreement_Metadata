# Rental Agreement Metadata Extraction

Extracts six target metadata fields from rental and lease agreements supplied as **.docx** or **scanned .png** files, using a hosted Large Language Model (LLM) pipeline.

The extraction pipeline is **genuinely AI/ML-driven**: no regular expressions, keyword-matching heuristics, static template parsers, or hardcoded answers are used for field extraction.

---

## Target Fields

The pipeline extracts exactly six target metadata fields, preserving the spelling used in the dataset column headers:

1. **`Aggrement Value`**: The recurring monthly rent as a plain number (numeric integer/float; deposit and advance amounts excluded).
2. **`Aggrement Start Date`**: The commencement date of the tenancy, standardized to `DD.MM.YYYY`.
3. **`Aggrement End Date`**: The expiration date of the tenancy (`DD.MM.YYYY`), extracted if explicitly stated or derived from start date and lease term.
4. **`Renewal Notice (Days)`**: The termination or renewal notice period expressed as an integer number of days (explicit day counts extracted directly; month periods converted using $1\text{ month} = 30\text{ days}$).
5. **`Party One`**: The landlord / lessor / owner entity (name only; honorifics and parentage excluded).
6. **`Party Two`**: The tenant / lessee entity (name only; honorifics and parentage excluded).

---

## Pipeline Architecture

```
.docx  ──> python-docx (paragraphs + tables in visual document order) ──┐
                                                                         ├─> Raw Document Text (cached to cache/)
.png   ──> NVIDIA Nemotron Parse (OCR layout tokens stripped) ──────────┘
                                │
                                ▼
         Dynamic Few-Shot Retriever (TF-IDF Cosine Similarity)
         • Reads labeled examples from data/train/ and data/train.csv
         • Strict cross-split stem exclusion (target document cannot retrieve itself)
         • Selects top k=2 most similar training examples
                                │
                                ▼
         Prompt Assembly (src/prompt.py)
         • Generic system instructions & field schemas
         • k=2 worked input/output demonstration pairs
         • Target document text enclosed in <DOCUMENT> tags
                                │
                                ▼
         LLM Inference (temperature 0.0, max_tokens 400)
         • Primary model: nvidia/nemotron-3.5-bolt-e4b (NVIDIA NIM)
         • Alternative: Google Gemini via OpenAI-compatible endpoint
                                │
                                ▼
         JSON Validation & Type Coercion (src/client.py)
         • Locates outermost JSON block
         • Enforces presence of all six target keys
         • Coerces numeric strings and maps null variants ("null", "n/a", "") to null
         • Self-Repair: On malformed JSON, triggers 1 corrective re-prompt turn
                                │
                                ▼
         Downstream Artifacts
         • Batch evaluation CSV/JSON predictions (output/)
         • Exact & relaxed recall reports (output/*_report.txt, output/*_errors.csv)
         • REST API endpoint (POST /extract via FastAPI)
```

### Design Principles

1. **Few-Shot In-Context Learning over Hardcoded Rules**: Dataset-specific conventions (e.g., Indian name formats, date arithmetic patterns) are taught dynamically to the model via the $k=2$ most similar training documents. The system prompt contains no dataset-specific names, dates, or values.
2. **Zero Test Data Leakage**:
   - Few-shot retrieval is constructed **strictly** from `data/train/` and `data/train.csv`.
   - The retriever explicitly drops any example matching the target document's stem (`exclude_stem`). Test document `24158401-Rental-Agreement` (which appears in both train labels and test documents) cannot retrieve itself.
   - Test evaluation is performed with frozen prompts without test-driven tuning.
3. **Resilience and Graceful Degradation**:
   - Transience handling: HTTP 408, 429, 500, 502, 503, and 504 errors trigger exponential backoff retries.
   - Parsing resilience: Invalid JSON triggers a single automated repair prompt turn showing the model its syntax error.
   - Document isolation: A failure on a single document is logged as `Status="FAILED"` with `null` fields; it never halts batch processing.
   - Text caching: OCR text is persisted to `cache/<filename>.<sha1>.txt` so scanned images are processed once.

---

## Evaluation Methodology & Metric Definitions

The evaluation script ([src/evaluate.py](src/evaluate.py)) evaluates predictions against ground-truth labels using two recall metrics:

### 1. Exact-Match Recall (Primary Metric)
A prediction is counted as correct if and only if the trimmed prediction string exactly matches the trimmed ground-truth label:
$$\text{Exact Match} = \mathbf{1}\left[\texttt{got.strip()} == \texttt{expected.strip()}\right]$$
- If ground truth has a value and prediction is `null` (or empty), it is scored as a mismatch (`0`).
- If ground truth is empty and prediction is `null`, it is scored as an exact match (`1`).
- Per-field recall is $\frac{\text{Exact Matches}}{\text{Total Labeled Documents}}$.
- **Macro-Average Recall** is the unweighted arithmetic mean across all six target fields:
  $$\text{Macro Recall} = \frac{1}{6} \sum_{i=1}^{6} \text{Recall}_i$$

### 2. Relaxed Recall (Diagnostic Metric)
A prediction is counted as a relaxed match if it matches after case-folding, collapsing whitespace, and stripping leading/trailing punctuation characters (`" .,;"`):
$$\text{Relaxed Match} = \mathbf{1}\left[\texttt{\_relaxed(got)} == \texttt{\_relaxed(expected)}\right]$$

#### Explanation of the 79.2% vs. 83.3% Test Scores
On the 4 held-out test documents (24 total field evaluations):
- **Exact Macro Recall: 79.2%** (19/24 exact matches).
- **Relaxed Macro Recall: 83.3%** (20/24 relaxed matches).
- The difference stems entirely from `Party Two` on document `228094620-Rental-Agreement`: the human ground-truth label contains an annotator typo with a leading period (`.B.Kishore `), whereas the model correctly predicted `B.Kishore`. Under relaxed evaluation, both strip to `b.kishore`, bringing `Party Two` recall from 50.0% (2/4) to 75.0% (3/4) and the macro average from 79.2% to 83.3%.

---

## Model Exploration & Benchmarking

### 1. Initial Connectivity & Latency Probes

Before running dataset evaluations, candidate models were screened via lightweight probes to assess responsiveness, formatting compliance, and latency.

* **NVIDIA Candidate Probes** ([probe.py](probe.py)):
  - `meta/llama-3.1-70b-instruct`: Default model fallback in `config.py` and `.env.example`. While highly capable, elevated queueing latency and occasional read timeouts were observed during initial screening on hosted endpoints (not evaluated as a full benchmark run).
  - `meta/llama-3.2-11b-vision-instruct`: Probed for multimodal inference.
  - `nvidia/nemotron-3.5-bolt-e4b`: Consistently returned responses in ~0.5s–2.0s with zero failures; selected as the primary production model.
  - `nvidia/nemotron-3.5-super-vl-120b-a12b`, `openai/gpt-oss-20b`, `minimaxai/minimax-m3.1-flash`, `poolside/laguna-xs-2.1`, `stepfun-ai/step-5-preview`.
* **Google Gemini Candidate Probes** ([gemini_probe.py](gemini_probe.py)):
  - Tested 8 candidate models for HTTP 200 and non-empty completions:
  - `gemini-flash-lite-latest` (HTTP 200, 1.30s latency, non-empty) — Proceeded to evaluation.
  - `gemini-3.5-flash-lite` (HTTP 200, 0.90s latency, non-empty) — Proceeded to evaluation.
  - `gemini-3-flash-preview` (HTTP 200, 1.99s latency, non-empty) — Proceeded to evaluation.
  - `gemini-3.1-flash-lite-preview` (HTTP 200, 1.21s latency, non-empty).
  - `gemini-2.5-flash` (404), `gemini-2.5-flash-lite` (404), `gemini-2.5-flash-preview-tts` (400), `gemini-2.5-flash-image` (429), `gemini-flash-latest` (timeout).

### 2. Full Training Set Benchmark (Leave-One-Out Evaluation)

All candidate models were benchmarked across the 9 available training documents under identical leave-one-out retrieval ($k=2$) and schema validation rules:

| Model | Provider | Macro Recall | Value | Start Date | End Date | Notice | Party One | Party Two | Sec/Doc | Failures | Output Directory |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **nvidia/nemotron-3.5-bolt-e4b** (Primary) | **NVIDIA** | **55.6%** | 66.7% | 66.7% | 33.3% | 55.6% | **55.6%** | 55.6% | **~2.0s** | **0** | [output/](output/) |
| `gemini-flash-lite-latest` | Google | 53.7% | 66.7% | 66.7% | 33.3% | 66.7% | 33.3% | 55.6% | ~3.7s | 0 | [output_gemini/](output_gemini/) |
| `gemini-3.5-flash-lite` | Google | 53.7% | 66.7% | 66.7% | 33.3% | 66.7% | 33.3% | 55.6% | ~2.5s | 0 | [output_gemini_35_flash_lite/](output_gemini_35_flash_lite/) |
| `gemini-3-flash-preview` | Google | 1.8% | 0.0% | 0.0% | 0.0% | 11.1% | 0.0% | 0.0% | ~10.4s | 9 | *(Failed; run aborted)* |

#### Benchmark Insights
- **Nemotron Bolt vs. Gemini Flash-Lite**: Gemini gained +1 correct prediction on `Renewal Notice (Days)` (6/9 vs 5/9), but dropped -2 predictions on `Party One` (3/9 vs 5/9) because it strictly reproduced the OCR dot in `Antonio Levy S. Ingles. Jr.` instead of normalizing to `, Jr.` as in the human labels.
- **Gemini 3 Flash Preview Failure**: As a reasoning model, its internal thinking tokens exhausted `max_tokens=400`, truncating JSON responses before closing brackets and subsequently encountering HTTP 429 quota exhaustion.
- **Decision Rule**: The project decision rule required a candidate model to beat the baseline by at least 10 percentage points ($\ge 65.6\%$) on train macro recall to replace the primary model. Because Gemini models scored 53.7%, **`nvidia/nemotron-3.5-bolt-e4b` was retained as the sole final model for test evaluation**.

---

## Final Submission Results

* **Final Selected Model**: `nvidia/nemotron-3.5-bolt-e4b`
* **OCR Service**: `nvidia/nemotron-parse` (Nemotron Parse, cached locally in `cache/`)
* **Few-Shot Setting**: `FEWSHOT_K=2` (TF-IDF retrieval, leave-one-out)
* **Active Predictions Directory**: [output/](output/) (baseline backup preserved in [output_nvidia/](output_nvidia/))

### Test Set Performance ([output/test_report.txt](output/test_report.txt))
Evaluated on all 4 held-out test documents:

| Field | Correct / Total | Recall (Exact) | Relaxed Recall | Notes on Mismatches |
|---|---|---|---|---|
| Aggrement Value | 4 / 4 | 100.0% | 100.0% | 100% exact match |
| Aggrement Start Date | 4 / 4 | 100.0% | 100.0% | 100% exact match |
| Aggrement End Date | 2 / 4 | 50.0% | 50.0% | Ground truth on `156155545` assumes 12 months (`14.11.2013`) despite text stating 11 months (`14.10.2013`). |
| Renewal Notice (Days) | 3 / 4 | 75.0% | 75.0% | `228094620` had no explicit notice clause in agreement body. |
| Party One | 4 / 4 | 100.0% | 100.0% | 100% exact match |
| Party Two | 2 / 4 | 50.0% | 75.0% | Exact miss on `228094620` caused by ground-truth leading dot typo (`.B.Kishore ` vs `B.Kishore`). |
| **Macro-Average Recall** | | **79.2%** | **83.3%** | **Primary Exact: 79.2% \| Relaxed Diagnostic: 83.3%** |

### Train Set Performance ([output/train_report.txt](output/train_report.txt))
Evaluated across 9 available training documents under leave-one-out:

| Field | Correct / Total | Recall (Exact) | Relaxed Recall |
|---|---|---|---|
| Aggrement Value | 6 / 9 | 66.7% | 66.7% |
| Aggrement Start Date | 6 / 9 | 66.7% | 66.7% |
| Aggrement End Date | 3 / 9 | 33.3% | 33.3% |
| Renewal Notice (Days) | 5 / 9 | 55.6% | 55.6% |
| Party One | 5 / 9 | 55.6% | 55.6% |
| Party Two | 5 / 9 | 55.6% | 55.6% |
| **Macro-Average Recall** | | **55.6%** | **55.6%** |

---

## Verified Ground-Truth Label Inconsistencies

During error analysis on `train.csv` and `test.csv`, several genuine ground-truth label anomalies were identified:

1. **Impossible Calendar Dates in `train.csv`**:
   - `data/train/47854715-RENTAL-AGREEMENT.docx`: Text specifies an 11-month lease from 1st April 2010. The label lists `31.02.2011` (February 31 does not exist).
   - `data/train/36199312-Rental-Agreement.png`: Label lists `31.04.2011` (April has only 30 days).
   - `data/train/18325926-Rental-Agreement-1.docx`: Label lists `31.11.2009` (November has only 30 days).
2. **Cross-Swapped Dataset Rows in `train.csv`**:
   - The ground-truth rows for `54770958-Rental-Agreement.png` and `54945838-Rental-Agreement.png` are transposed in `train.csv`. Document `54770958` specifies rent of Rs. 5,500 and parties Asha Ramesh / Sadasivuni Deepti, yet `train.csv` assigns it the labels of `54945838` (K. Parthasarathy / Veerabrahmam Bathini).
3. **Missing Training Document File**:
   - Stem `24158401-Rental-Agreement` is listed in `train.csv`, but its document file is absent from `data/train/` (it exists as `data/test/24158401-Rental-Agreement.png` and is labeled in `test.csv`). The pipeline safely logs a warning and excludes it from the retrieval index.
4. **Unattested Label Data**:
   - `data/train/44737744-Maddireddy-Bhargava-Reddy-Rental-Agreement.docx`: Ground truth lists rent `3000` and start date `20.09.2010`. Neither value appears anywhere in the document text (the text states `Rs 9.99.7.9° p.m.` and `.?.??& day of 2010`).

---

## Setup & Configuration

### Prerequisites
- Python 3.10+
- Valid NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (or Google Gemini API key if running Gemini comparison).

### Installation
```bash
# Clone or navigate to the project directory
cd C:\Users\srini\Downloads\metadata-extraction-v2\proj

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Create .env from example template
cp .env.example .env
```

### Environment Configuration ([config.py](config.py))

Configure `.env` using the available settings:

| Variable | Default Value | Description |
|---|---|---|
| `NVIDIA_API_KEY` | *(None / required)* | API key for NVIDIA NIM chat completions and Nemotron Parse OCR. |
| `NVIDIA_MODEL` | `meta/llama-3.1-70b-instruct` | Active chat model name. (Set to `nvidia/nemotron-3.5-bolt-e4b` for final pipeline). |
| `NVIDIA_ENDPOINT` | `https://integrate.api.nvidia.com/v1/chat/completions` | NVIDIA OpenAI-compatible chat completions endpoint. |
| `NVIDIA_PARSE_ENDPOINT` | `https://ai.api.nvidia.com/v1/retrieval/nvidia/nemotron-3.5-parse/parse` | NVIDIA Nemotron Parse document OCR endpoint. |
| `NVIDIA_CONNECT_TIMEOUT`| `15` | Network connection timeout in seconds. |
| `NVIDIA_READ_TIMEOUT` | `120` | HTTP read response timeout in seconds. |
| `NVIDIA_MAX_RETRIES` | `4` | Max retry attempts for transient HTTP status codes (408, 429, 5xx). |
| `FEWSHOT_K` | `2` | Number of worked training examples retrieved per query ($k=2$). |
| `FEWSHOT_MAX_CHARS` | `7000` | Max character slice allowed per retrieved training example. |
| `MAX_TOKENS` | `400` | Token generation limit for the chat completion payload. |
| `WORKERS` | `2` | Concurrent worker thread count (set to `1` for sequential execution). |
| `DATA_DIR` | `data/` | Root directory containing `train/`, `test/`, `train.csv`, and `test.csv`. |
| `OUTPUT_DIR` | `output/` | Destination folder for predictions, reports, and error summaries. |
| `CACHE_DIR` | `cache/` | Cache directory storing extracted plain-text and OCR outputs. |
| `VERBOSE` | `1` | Console logging verbosity (`1` for detailed progress, `0` to silence). |
| `GEMINI_API_KEY` | *(Optional)* | Google Gemini API key used by `run_with_gemini.py`. |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Default model when running the Gemini wrapper. |

---

## Usage & Execution Commands

### 1. Run Final Test Evaluation
Executes extraction on the held-out test split using the configured primary model (`nvidia/nemotron-3.5-bolt-e4b`):
```bash
python run.py --split test
```
*Outputs generated in `output/`*:
- `output/test_predictions.csv`: Model predictions alongside document status and error logs.
- `output/test_predictions.json`: Structured JSON dump of all predictions.
- `output/test_report.txt`: Summary report with exact and relaxed per-field recall.
- `output/test_errors.csv`: Detailed list of every mismatched field, predicted value, and ground-truth label.

### 2. Run Train Evaluation (Leave-One-Out)
Evaluates extraction performance across the training documents, excluding the target document from retrieval:
```bash
python run.py --split train
```

### 3. Smoke Test
Quickly test end-to-end extraction on the first document only:
```bash
python run.py --split test --limit 1
```

### 4. Run Offline Unit Tests
Executes the comprehensive 23-test test suite offline (mocked sessions; zero API calls consumed):
```bash
python -m pytest -q
```

### 5. Benchmark Alternative Gemini Models
Evaluate candidate Gemini models through the isolated wrapper script:
```bash
# Probes available Gemini models:
python gemini_probe.py

# Run training evaluation (outputs saved to isolated directory, e.g. output_gemini_35_flash_lite/):
python run_with_gemini.py gemini-3.5-flash-lite --split train
```

### 6. Probe NVIDIA Models & Service Health
Verify endpoint connectivity and test a tiny sample call against available models:
```bash
python probe.py
python diagnose.py
```

---

## REST API Deployment

The project includes a production-ready asynchronous FastAPI application ([api/main.py](api/main.py)):

### Start the Server
```bash
uvicorn api.main:app --port 8000
```

### Health Check
```bash
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

### Extract Metadata from Document
```bash
curl -X POST "http://localhost:8000/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/95980236-Rental-Agreement.png"
```
**Example JSON Response**:
```json
{
  "file_name": "95980236-Rental-Agreement.png",
  "metadata": {
    "Aggrement Value": 9000,
    "Aggrement Start Date": "01.04.2010",
    "Aggrement End Date": "31.01.2011",
    "Renewal Notice (Days)": 30,
    "Party One": "S.Sakunthala",
    "Party Two": "V.V.Ravi Kian"
  }
}
```

**HTTP Status Codes**:
- `200 OK`: Successful extraction with validated fields.
- `415 Unsupported Media Type`: Uploaded file is not `.docx` or `.png`.
- `422 Unprocessable Entity`: Document text is unreadable or malformed.
- `502 Bad Gateway`: Upstream LLM or OCR service connection failure.

---

## Privacy & Security Considerations

1. **Credential Safety**:
   - Never commit `.env` or hardcode API keys in source files.
   - `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `cache/`, and `*.pyc` are explicitly excluded via [.gitignore](.gitignore).
2. **Google Gemini Privacy Notice**:
   - Per Google's Terms of Service, free-tier requests sent to Google Gemini APIs may be logged and reviewed by human reviewers to train and improve Google products. Do not send confidential contracts via free-tier Gemini endpoints.
3. **NVIDIA NIM API Privacy**:
   - Data-handling and retention policies depend on the applicable hosted service tier and terms. Users should consult NVIDIA's current terms before processing confidential agreements, or deploy containerized NIM models on private infrastructure.
