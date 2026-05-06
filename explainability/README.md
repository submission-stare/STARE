# Explainability Layer: LangGraph Trading Post-Mortem Agent

Explainability Layer is a Python-based intelligent agent system designed to perform a quantitative "post-mortem" analysis of trading model performance. Built using **LangGraph** and **LangChain**, it acts as a senior quantitative analyst by ingesting performance metrics, formulating hypotheses about the model's behavior, generating and executing Python code to validate those hypotheses via data visualization, and ultimately compiling a comprehensive Markdown report.

---

## Architecture Overview

The pipeline is a LangGraph state machine with two supported modes. Before either graph runs, `main.py` resolves the benchmark data path, syncs S3 inputs into `.data_cache/` when needed, initializes a timestamped run log, and, in report mode, loads transaction and portfolio snapshot summaries from the configured benchmark scope.

### Default Report Workflow

The default `--analysis-mode report` graph produces an integrated behavior-and-code-grounded post-mortem:

1. **`code_context_builder`**
   Builds a compact dependency and configuration context from the benchmark entrypoint, benchmark config, code scope, and live report state.

2. **`data_analyst`**
   Routes the behavioral analysis loop based on the current hypotheses, generated plots, and accumulated messages. It is one router in the graph, not the only decision point.

3. **`hypothesis_maker` / `hypothesis_forum`**
   The graph node is named `hypothesis_maker`, but it is implemented by `hypothesis_forum`. This multi-panelist forum reviews scoped benchmark files, transaction summaries, snapshot summaries, documentation, and code context to generate up to the configured hypothesis cap, then deduplicates and peer-reviews them.

4. **`code_claim_explainer`**
   Maps live report claims and hypotheses back to reachable code/config context so later prompts can connect observed behavior to implementation details.

5. **`hypothesis_investigator`**
   Checks which hypothesis tests are feasible from the available scoped CSVs and consolidated `check_input_data` files. It routes either to `code_generator` when executable tests remain or back to `data_analyst` when enough evidence has been gathered or safety limits are reached.

6. **`code_generator`**
   Generates or fixes Python analysis code. It auto-injects a preamble that preloads known benchmark DataFrames, strips redundant imports and file reads, and rewrites `plt.show()` calls into saved PNGs.

7. **`code_executor`**
   Executes the generated code with Python `exec()` while redirecting stdout/stderr and temporarily using `generated_code_results/` as the working directory. This is controlled execution for analysis capture, not a true security sandbox. Runtime errors or missing plots route back to `code_generator` until the retry limit is reached.

8. **`code_hypothesis_investigator`**
   Collects prompt-based code evidence for the live hypotheses and passes that evidence into the final synthesis path.

9. **`consensus_forum`**
   Runs a multi-panelist discussion over the evidence to answer what the agent is doing, how it is doing it, and why it behaves that way.

10. **`report_generator`**
    Writes the final report artifacts: `report.tex`, a readable `report.md`, and a best-effort `report.pdf` through `tectonic` or `pandoc` when available.

### Code-Report Workflow

The separate `--analysis-mode code-report` graph starts from an existing report and produces `code_report.md`. It runs linearly through `code_context_builder`, `code_claim_explainer`, `code_hypothesis_forum`, `code_hypothesis_investigator`, `code_consensus_forum`, and `code_report_generator` to extract claims, map them to code/config paths, formulate code-grounded hypotheses, collect evidence tasks, synthesize code-specific consensus answers, and write recommendations.

### State Management (`state.py`)

The pipeline state is maintained using a shared `TypedDict`. The main state groups are:
- Runtime context: `messages`, `raw_data_path`, `analysis_mode`, `report_path`, benchmark code/config paths, and `next_node`.
- Behavioral evidence: `hypotheses`, `investigation_tests`, `generated_code`, `plot_paths`, `code_fix_retries`, transaction summaries, and snapshot summaries.
- Synthesis outputs: `consensus_answers`, `final_report`, and the report artifacts written by `report_generator`.
- Code-grounding context: extracted report claims, dependency graph data, enriched claim-to-code mappings, code hypotheses, evidence tasks/results, code recommendations, code consensus answers, and `final_code_report`.

---

## Getting Started

### Prerequisites

You need [Conda](https://docs.conda.io/en/latest/) and an [OpenRouter](https://openrouter.ai/) API key.

### Installation

1. Clone the repository and navigate to the project root.
2. Create and activate the Conda environment:
   ```bash
   conda create -n inspector python=3.10 -y
   conda activate inspector
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory and add your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY="your-api-key-here"
   OPENROUTER_MODEL="anthropic/claude-3.5-sonnet" # Optional, defaults to Claude 3.5 Sonnet
   ```

## Usage

### Prerequisites
Make sure you have your OpenRouter API key and AWS Credentials configured in your `.env` file:
```
OPENROUTER_API_KEY=your_openrouter_api_key
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_SESSION_TOKEN=your_token
```

### Running the Agent
The agent can be run directly from the command line. By default it runs the integrated report workflow (`--analysis-mode report`) and writes the report to `report.md`.

#### Method 1: Automatic Inference (Recommended)
If you run the script without data-path arguments, report mode parses `specification/important_paths.md`, uses the first `s3://...` path it finds for the benchmark output, and syncs it into `.data_cache/` with `aws s3 sync`:
```bash
python main.py
```

#### Method 2: Manual Data Path
You can also point the agent to a specific local benchmark-output folder, CSV/JSON file, or a different `s3://` path with `--data-path`:
```bash
python main.py --data-path <S3-benchmark-data-path>
# or locally:
python main.py --data-path .data_cache/
```
When `--data-path` is an S3 URI, the agent still syncs it into `.data_cache/` before analysis. Local paths are used as-is.

The agent scopes its benchmark reads via `config/settings.yaml` under `data_scope`.
By default it reads:
- `run_000/interleaved` for single-run analysis
- `run_000/interleaved/agents_trading/trading_analysis` for transaction and snapshot summaries
- `aggregated_interleaved` for aggregated multi-run summaries

You can override those with environment variables:
```bash
DATA_SCOPE_RUN_DIR=run_000
DATA_SCOPE_TRADING_REGIME_DIR=interleaved
DATA_SCOPE_AGGREGATED_DIR=aggregated_interleaved
```

To generate only the code-grounded report from an existing report, use `code-report` mode:
```bash
python main.py --analysis-mode code-report --report-path report.md
```
This mode writes `code_report.md` and uses `--benchmark-entry`, `--benchmark-config`, and `--code-scope-root` to decide which benchmark code/config paths to inspect. The legacy `--analysis-mode both` option is accepted but is now treated as `report`.

#### Starting Fresh vs Improving Existing Reports
By default, each run of the agent will read the existing `report.md` (if it exists) and **rewrite/improve** it by seamlessly integrating the new findings, code executions, and plots into a single cohesive document.

If you want to discard the old report and start a brand new context, use the `--clear-report` flag:
```bash
python main.py --clear-report
```

### Integration Testing
The old `performance_metrics.csv` basic graph tests are now encapsulated in an Integration Test:
```bash
python -m unittest tests/test_integration.py
```

**3. Review the Output**

- The pipeline will output terminal debug information as the router transitions between nodes.
- Generated `.png` files (e.g., `volatility_comparison.png`) will be captured from `./generated_code_results`.
- A final, consolidated analytical report will be made available at `report.md`. 

---

## Configuration & Debugging

- **Self-Correction:** If generated code crashes or produces no plots, `code_executor` captures the traceback/output and routes directly back to `code_generator` for a bounded fix loop.
- **Model Parameters:** LLM temperatures and model choices can be modified in `utils/llm.py` or `.env`.
- **Output Directory:** Generated analysis charts are currently captured from `generated_code_results/`; the execution helper also supports a configurable working directory.
