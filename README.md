# infer

Pipe-friendly LLM agent harness with a bash tool. Works with any OpenAI-compatible provider.

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

Or manually:

```bash
curl -o /usr/local/bin/infer https://raw.githubusercontent.com/turlockmike/infer/main/infer
chmod +x /usr/local/bin/infer
pip install openai
```

## Usage

```
infer [OPTIONS] [PROMPT]

  -m MODEL     Model name (default: gemma4:latest)
  -u URL       Provider base URL (default: http://localhost:11434/v1)
  -k KEY       API key (default: "ollama" for local providers)
  -r ROLE      Named role — loads ~/.config/infer/roles/<name>.md
  -f FILE      File to use as context (prepended to prompt)
  -j [SHAPE]   Output JSON, optionally validated against a shape
  -v           Verbose — show tool calls and token stats on stderr
```

stdin is context, argument is the instruction:

```bash
cat logs.txt | infer "summarize the errors"
infer -f config.yaml "is this valid?"
```

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
