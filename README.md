# infer

Pipe-friendly LLM agent with a bash tool. Works with any OpenAI-compatible provider.

```bash
infer "what directory am i in"
cat crash.log | infer "why did this fail"
infer -f main.py "explain this"
infer -j '{"name":"string","pid":0}' "current process info"
```

## Install

```bash
pip install openai
curl -o /usr/local/bin/infer https://raw.githubusercontent.com/turlockmike/infer/main/infer
chmod +x /usr/local/bin/infer
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
infer -j '["string"]' "list files in /tmp"
```

## Config

Create `~/.config/infer/config.json`:

```json
{ "url": "http://localhost:11434/v1", "model": "gemma4:latest", "api_key": "ollama" }
```

Local overrides: `.infer.json` (config) and `.infer.md` (system prompt append).

**Roles** — named system prompt presets at `~/.config/infer/roles/<name>.md`:

```bash
echo "Output only code. No explanation." > ~/.config/infer/roles/coder.md
infer -r coder "write a fizzbuzz in python"
```

## Providers

| Provider   | URL                                    | Key             |
|------------|----------------------------------------|-----------------|
| Ollama     | `http://localhost:11434/v1`            | `ollama`        |
| LM Studio  | `http://localhost:1234/v1`             | `lm-studio`     |
| Groq       | `https://api.groq.com/openai/v1`       | `$GROQ_API_KEY` |
| OpenRouter | `https://openrouter.ai/api/v1`         | `$OR_API_KEY`   |
| OpenAI     | `https://api.openai.com/v1`            | `$OPENAI_API_KEY` |

## License

MIT
