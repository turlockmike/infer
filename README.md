# infer

Pipe-friendly LLM agent harness with a bash tool. Works with any OpenAI-compatible provider.

![demo](demo.gif)

```bash
infer "what directory am i in"
cat crash.log | infer "why did this fail"
infer -f main.py "explain this"
infer -j '{"name":"string","pid":0}' "current process info"
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/turlockmike/infer/main/install.sh | sh
```

Downloads a pre-built binary for your platform (macOS arm64/x64, Linux arm64/x64). No runtime required.

**From source** (requires [Bun](https://bun.sh)):

```bash
git clone https://github.com/turlockmike/infer
cd infer
bun install
bun run infer.ts "hello"
```

## Quickstart

Set env vars and go — no config file needed:

```bash
# OpenAI
export INFER_API_KEY=sk-...
export INFER_URL=https://api.openai.com/v1
export INFER_MODEL=gpt-4o

# Anthropic (via OpenAI-compatible proxy, e.g. LiteLLM)
export INFER_API_KEY=sk-ant-...
export INFER_URL=http://localhost:4000/v1
export INFER_MODEL=claude-opus-4-5

# Groq
export INFER_API_KEY=$GROQ_API_KEY
export INFER_URL=https://api.groq.com/openai/v1
export INFER_MODEL=llama-3.3-70b-versatile

# Local (Ollama)
export INFER_URL=http://localhost:11434/v1
export INFER_MODEL=gemma4:latest
export INFER_API_KEY=ollama

infer "what directory am i in"
```

Env vars are overridden by CLI flags and persisted by `infer config set`.

## Usage

```
infer [OPTIONS] [PROMPT]

  -m MODEL          Model name (default: gemma4:latest)
  -u URL            Provider base URL (default: http://localhost:11434/v1)
  -k KEY            API key (default: "ollama" for local providers)
  -s TEXT           System prompt override
  -r ROLE           Named role — loads ~/.config/infer/roles/<name>.md
  -f FILE           File to use as context (prepended to prompt)
  -j [SHAPE]        Output JSON, optionally validated against a shape
  -n N              Max tool-call iterations per run (default: 10)
  -v                Verbose — show tool calls and token stats on stderr
  --stream          Stream tokens as they arrive (default: off)
  --no-sandbox      Use real bash (default: sandboxed via sandbox-exec)
  --allow-network   Enable network access inside the sandbox
```

stdin is context, argument is the instruction:

```bash
cat logs.txt | infer "summarize the errors"
infer -f config.yaml "is this valid?"
```

## Sessions

Use `-S FILE` to persist a conversation across invocations. The file is JSONL — one message per line, human-readable, appendable with standard tools.

```bash
# Start a session
infer -S /tmp/review.jsonl "here is the file" < src/auth.py

# Do something programmatic between turns
issues=$(run_linter src/auth.py)

# Continue the same conversation
echo "$issues" | infer -S /tmp/review.jsonl "the linter found these issues — which ones matter?"

# One more turn
infer -S /tmp/review.jsonl "now write the fix for the most critical one"
```

Sessions also work in REPL mode — load an existing conversation and continue interactively:

```bash
infer -S /tmp/review.jsonl repl
# infer repl  gemma4:latest  (6 messages loaded from /tmp/review.jsonl)
```

The session file is plain JSONL. Inspect, fork, or edit it with standard tools:

```bash
jq '.role + ": " + .content' /tmp/review.jsonl   # read conversation
cp /tmp/review.jsonl /tmp/review-fork.jsonl       # branch from this point
```

## Sandbox

By default, bash commands run in a sandbox — real system binaries with OS-enforced write restrictions. Network access is blocked and writes are restricted to the current directory and `/tmp`.

| Platform | Sandbox backend |
|----------|----------------|
| macOS    | `sandbox-exec` (Seatbelt, built-in) |
| Linux    | `bwrap` (bubblewrap, if installed) |
| Other    | No sandbox — runs unrestricted |

```bash
infer "list files here"                   # sandboxed
infer --allow-network "fetch example.com" # enable network in sandbox
infer --no-sandbox "unrestricted bash"    # bypass sandbox entirely
```

Install bubblewrap on Linux: `apt install bubblewrap` / `dnf install bubblewrap`

## JSON Output

Use `-j` to get structured JSON back. Validation is automatic — if the model returns
invalid JSON or the wrong shape, it gets a correction message and retries.

```bash
# Plain JSON (any shape)
infer -j "current date and time"

# Object shape
infer -j '{"name":"string","pid":0}' "current process info"

# Array of strings
infer -j '["string"]' "list files in /tmp"

# Array of objects
infer -j '[{"file":"string","size":0}]' "files in /tmp with sizes"
```

### Shape syntax

Shapes are JSON examples — types are inferred from the example values:

| Example value | Matches |
|---------------|---------|
| `"string"` | any string |
| `0` | any number |
| `true` | boolean |
| `{}` | any object |
| `[]` | any array |
| `{"key":"string"}` | object with that key as a string |
| `["string"]` | array where every element is a string |

Shapes can nest arbitrarily:

```bash
infer -j '[{"name":"string","tags":["string"],"score":0}]' "top 3 processes"
```

Pipes cleanly into `jq`:

```bash
infer -j '["string"]' "list files in /tmp" | jq '.[]'
```

## Remote Resources

infer has one tool: bash. Remote services are just CLI commands.

**Existing CLIs work directly:**

```bash
# Summarize recent commits
git log --oneline -20 | infer "summarize what changed this week"

# One-liner per merged PR
gh pr list --state merged --limit 10 --json number,title,mergedAt | infer "one sentence per PR"

# Review a PR diff
gh pr diff 42 | infer "what's the risk level of this change?"
```

**No CLI? Use [murl](https://github.com/turlockmike/murl) — curl for MCP servers:**

```bash
# List tools on DeepWiki (GitHub repo documentation, no auth required)
murl https://mcp.deepwiki.com/mcp/tools | infer "what can this service do?"

# Read the architecture of any repo indexed on deepwiki.com
murl https://mcp.deepwiki.com/mcp/tools/read_wiki_structure \
  -d repoName=langchain-ai/langchain | infer "what are the main sections?"
```

The same pattern works for any MCP server: `murl <server>/tools/<name> -d key=value | infer "..."`. The public MCP ecosystem is growing — find servers at [glama.ai/mcp/servers](https://glama.ai/mcp/servers).

## Prompts as Files

A prompt is just a file. Use `-s` to set the system prompt from a file:

```bash
infer -s "$(cat prompts/security-reviewer.md)" < src/auth.py
```

**Skills are a prompt file + a one-line bin script:**

```bash
# 1. Write the prompt
mkdir -p ~/.config/infer/prompts
cat > ~/.config/infer/prompts/review.md << 'EOF'
You are a senior engineer doing a code review. Focus on: correctness, security, and performance.
Flag anything that could fail in production. Be direct. No praise.
EOF

# 2. Write the script
cat > ~/bin/review << 'EOF'
#!/bin/bash
infer -s "$(cat ~/.config/infer/prompts/review.md)" "$@"
EOF
chmod +x ~/bin/review
```

```bash
# Now use it anywhere
cat src/auth.py | review
gh pr diff 42 | review "is this safe to merge?"
git diff HEAD~1 | review "any regressions?"
```

No framework. No config. Prompts are text, skills are scripts, remote services are CLIs.

## Config

```bash
infer config set url http://localhost:11434/v1
infer config set model gemma4:latest
infer config set api_key ollama
infer config show
infer config get model
infer config unset model
```

Local overrides per project: `.infer.json` (config) and `.infer.md` (system prompt, appended to global).

**Roles** — named system prompt presets at `~/.config/infer/roles/<name>.md`:

```bash
echo "Output only code. No explanation." > ~/.config/infer/roles/coder.md
infer -r coder "write a fizzbuzz in python"
```

## Model Compatibility

infer requires a model that supports **tool use** (function calling). Not all models do.

| Model family | Tool use | Notes |
|---|---|---|
| `gemma4` | ✅ | Works well |
| `qwen2.5` | ✅ | Works well |
| `llama3.1+` | ✅ | Works well |
| `phi4` | ✅ | Works well |
| `qwen3` | ⚠️ | Has chain-of-thought mode — append `/no_think` to your prompt or system prompt to suppress it, otherwise responses are slow and verbose |
| `gemma3` | ❌ | Does not support tool use in Ollama — use `gemma4` instead |

If you see `model 'X' does not support tool use`, switch to a different model or variant.

## Providers

| Provider   | URL                                    | Key               |
|------------|----------------------------------------|-------------------|
| Ollama     | `http://localhost:11434/v1`            | `ollama`          |
| LM Studio  | `http://localhost:1234/v1`             | `lm-studio`       |
| Groq       | `https://api.groq.com/openai/v1`       | `$GROQ_API_KEY`   |
| OpenRouter | `https://openrouter.ai/api/v1`         | `$OR_API_KEY`     |
| OpenAI     | `https://api.openai.com/v1`            | `$OPENAI_API_KEY` |

## License

MIT
