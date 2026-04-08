#!/usr/bin/env bun
/**
 * infer — pipe-friendly LLM agent harness with a bash tool.
 * Works with any OpenAI-compatible provider.
 *
 * Usage:
 *   infer "question"
 *   cat file | infer "question about it"
 *   infer -f crash.log "why did this fail"
 *   infer -r coder "fix this"
 *   infer --allow-network "fetch example.com"
 *   infer --no-sandbox "unrestricted bash"
 */

import { Bash, MountableFs, InMemoryFs, ReadWriteFs } from "just-bash";
import OpenAI from "openai";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { spawnSync } from "child_process";
import { parseArgs } from "util";

// --- Config paths ---
const CONFIG_DIR    = join(homedir(), ".config", "infer");
const GLOBAL_CONFIG = join(CONFIG_DIR, "config.json");
const GLOBAL_SYSTEM = join(CONFIG_DIR, "system.md");
const GLOBAL_ROLES  = join(CONFIG_DIR, "roles");
const LOCAL_CONFIG  = ".infer.json";
const LOCAL_SYSTEM  = ".infer.md";

const VALID_KEYS = ["url", "model", "api_key"] as const;
type ConfigKey = typeof VALID_KEYS[number];

const DEFAULTS = {
  url:     "http://localhost:11434/v1",
  model:   "gemma4:latest",
  api_key: "ollama",
};

const DEFAULT_SYSTEM = `You have one tool: bash. Use it for everything.

Your final message is the output of this program — it will be printed to stdout and may be piped into other commands. Be concise and output only what was asked for. No preamble, no commentary.

File conventions:
- Read files with: cat -n <path>
- List directories with: ls -la <path>
- Write files: cat the file first if it exists, then write full content with tee <path> <<'EOF'\\n...\\nEOF
- Never overwrite a file without reading it first unless explicitly told to.`;

const TOOLS: OpenAI.Chat.ChatCompletionTool[] = [{
  type: "function",
  function: {
    name: "bash",
    description: "Run a shell command, return stdout+stderr",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "Shell command to run" }
      },
      required: ["command"],
    },
  },
}];

// --- Config ---
function loadConfig(): typeof DEFAULTS & { system: string } {
  let cfg = { ...DEFAULTS };
  const systemParts: string[] = [];

  for (const [cfgPath, sysPath] of [[GLOBAL_CONFIG, GLOBAL_SYSTEM], [LOCAL_CONFIG, LOCAL_SYSTEM]]) {
    if (existsSync(cfgPath)) Object.assign(cfg, JSON.parse(readFileSync(cfgPath, "utf8")));
    if (existsSync(sysPath)) systemParts.push(readFileSync(sysPath, "utf8").trim());
  }

  return { ...cfg, system: systemParts.length ? systemParts.join("\n\n") : DEFAULT_SYSTEM };
}

function saveConfig(updates: Partial<Record<ConfigKey, string>>) {
  mkdirSync(CONFIG_DIR, { recursive: true });
  const current = existsSync(GLOBAL_CONFIG) ? JSON.parse(readFileSync(GLOBAL_CONFIG, "utf8")) : {};
  writeFileSync(GLOBAL_CONFIG, JSON.stringify({ ...current, ...updates }, null, 2));
}

function runConfigCmd(args: string[]) {
  const [sub, key, value] = args;
  const cfg = loadConfig();

  if (sub === "show") {
    const merged = { url: cfg.url, model: cfg.model, api_key: cfg.api_key };
    for (const [k, v] of Object.entries(merged)) console.log(`${k}=${v}`);

  } else if (sub === "get") {
    if (!VALID_KEYS.includes(key as ConfigKey)) { console.error(`infer config: unknown key '${key}'. Valid: ${VALID_KEYS.join(", ")}`); process.exit(1); }
    console.log((cfg as any)[key] ?? "");

  } else if (sub === "set") {
    if (!VALID_KEYS.includes(key as ConfigKey)) { console.error(`infer config: unknown key '${key}'. Valid: ${VALID_KEYS.join(", ")}`); process.exit(1); }
    saveConfig({ [key]: value } as any);
    console.log(`${key}=${value}`);

  } else if (sub === "unset") {
    mkdirSync(CONFIG_DIR, { recursive: true });
    const current = existsSync(GLOBAL_CONFIG) ? JSON.parse(readFileSync(GLOBAL_CONFIG, "utf8")) : {};
    delete current[key];
    writeFileSync(GLOBAL_CONFIG, JSON.stringify(current, null, 2));
    console.log(`unset ${key}`);

  } else {
    console.error("usage: infer config <show|get|set|unset> [key] [value]");
    process.exit(1);
  }
}

// --- Role ---
function loadRole(name: string): string {
  const path = join(GLOBAL_ROLES, `${name}.md`);
  if (!existsSync(path)) { console.error(`infer: role '${name}' not found at ${path}`); process.exit(1); }
  return readFileSync(path, "utf8").trim();
}

// --- Shape validation ---
export function validateShape(data: unknown, shape: unknown): string | null {
  if (typeof shape === "string") return typeof data !== "string" ? `expected string, got ${typeof data}` : null;
  if (typeof shape === "number") return typeof data !== "number" ? `expected number, got ${typeof data}` : null;
  if (typeof shape === "boolean") return typeof data !== "boolean" ? `expected boolean, got ${typeof data}` : null;
  if (Array.isArray(shape)) {
    if (!Array.isArray(data)) return `expected array, got ${typeof data}`;
    if (shape.length && (data as unknown[]).length) {
      for (let i = 0; i < (data as unknown[]).length; i++) {
        const err = validateShape((data as unknown[])[i], shape[0]);
        if (err) return `[${i}]: ${err}`;
      }
    }
    return null;
  }
  if (shape && typeof shape === "object") {
    if (!data || typeof data !== "object" || Array.isArray(data)) return `expected object, got ${typeof data}`;
    for (const key of Object.keys(shape as object)) {
      if (!(key in (data as object))) return `missing key '${key}'`;
      const err = validateShape((data as any)[key], (shape as any)[key]);
      if (err) return `['${key}']: ${err}`;
    }
    return null;
  }
  return null;
}

// --- Bash execution ---
async function execBash(cmd: string, sandbox: boolean, allowNetwork: boolean, cwd: string): Promise<string> {
  if (!sandbox) {
    const result = spawnSync(cmd, { shell: true, cwd, encoding: "utf8" });
    return ((result.stdout ?? "") + (result.stderr ?? "")).trim() || "(no output)";
  }

  const fs = new MountableFs({ base: new InMemoryFs() });
  fs.mount(cwd, new ReadWriteFs({ root: cwd }));
  fs.mount("/tmp", new ReadWriteFs({ root: "/tmp" }));

  const networkOpts = allowNetwork ? { allowedUrlPrefixes: [""] } : undefined;
  const bash = new Bash({ fs, cwd, ...(networkOpts ? { network: networkOpts } : {}) });

  const result = await bash.exec(cmd);
  return ((result.stdout ?? "") + (result.stderr ?? "")).trim() || "(no output)";
}

// --- Main agent loop ---
export async function run(opts: {
  prompt: string;
  url: string;
  model: string;
  apiKey: string;
  system: string;
  verbose: boolean;
  jsonMode: boolean | string;
  sandbox: boolean;
  allowNetwork: boolean;
  maxSteps?: number;
}): Promise<number> {
  const { prompt, url, model, apiKey, system, verbose, jsonMode, sandbox, allowNetwork, maxSteps = 10 } = opts;
  const cwd = process.cwd();
  const client = new OpenAI({ baseURL: url, apiKey });
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: system },
    { role: "user", content: prompt },
  ];

  if (verbose) process.stderr.write(`model=${model} url=${url} sandbox=${sandbox}\n`);

  for (let step = 1; step <= maxSteps; step++) {
    const resp = await client.chat.completions.create({ model, messages, tools: TOOLS });
    const msg = resp.choices[0].message;

    if (msg.tool_calls?.length) {
      for (const call of msg.tool_calls) {
        const cmd = JSON.parse(call.function.arguments).command as string;
        const output = await execBash(cmd, sandbox, allowNetwork, cwd);
        if (verbose) { process.stderr.write(`+ ${cmd}\n`); process.stderr.write(`${output}\n`); }
        messages.push(msg);
        messages.push({ role: "tool", tool_call_id: call.id, content: output });
      }
    } else {
      let content = msg.content ?? "";

      if (jsonMode) {
        let parsed: unknown;
        try {
          parsed = JSON.parse(content);
        } catch (e: any) {
          process.stderr.write(`infer: invalid JSON: ${e.message}\n`);
          messages.push(msg);
          messages.push({ role: "user", content: `Your response was not valid JSON. Error: ${e.message}\nYou returned: ${content}\nFix it and respond with valid JSON only. No markdown, no code fences, no explanation.` });
          continue;
        }
        if (typeof jsonMode === "string") {
          const err = validateShape(parsed, JSON.parse(jsonMode));
          if (err) {
            process.stderr.write(`infer: shape mismatch: ${err}\n`);
            messages.push(msg);
            messages.push({ role: "user", content: `Your response was invalid. Problem: ${err}\nYou returned: ${content}\nRequired shape: ${jsonMode}\nFix it and respond with valid JSON matching that shape exactly. Nothing else.` });
            continue;
          }
        }
      }

      process.stdout.write(content + "\n");
      if (verbose) process.stderr.write(`[${resp.usage?.completion_tokens} tok, prompt=${resp.usage?.prompt_tokens}]\n`);
      return 0;
    }
  }

  process.stderr.write("infer: max steps reached\n");
  return 1;
}

// --- Entry point ---
if (import.meta.main) {
  const argv = process.argv.slice(2);

  if (argv[0] === "config") { runConfigCmd(argv.slice(1)); process.exit(0); }

  const cfg = loadConfig();
  const hasConfig = existsSync(GLOBAL_CONFIG) || existsSync(LOCAL_CONFIG);
  if (!hasConfig) {
    process.stderr.write("infer: no config found. Run:\n");
    process.stderr.write("  infer config set url http://localhost:11434/v1\n");
    process.stderr.write("  infer config set model gemma4:latest\n");
    process.stderr.write("  infer config set api_key ollama\n\n");
    process.stderr.write("  Proceeding with built-in defaults...\n\n");
  }

  const { values, positionals } = parseArgs({
    args: argv,
    options: {
      model:         { type: "string",  short: "m", default: cfg.model },
      url:           { type: "string",  short: "u", default: cfg.url },
      "api-key":     { type: "string",  short: "k", default: cfg.api_key },
      system:        { type: "string",  short: "s", default: cfg.system },
      role:          { type: "string",  short: "r" },
      file:          { type: "string",  short: "f" },
      json:          { type: "string",  short: "j" },
      verbose:       { type: "boolean", short: "v", default: false },
      "no-sandbox":  { type: "boolean", default: false },
      "allow-network": { type: "boolean", default: false },
    },
    allowPositionals: true,
  });

  let system = values.role ? loadRole(values.role) : (values.system ?? cfg.system);

  let jsonMode: boolean | string = false;
  if (values.json !== undefined) {
    jsonMode = values.json === "" ? true : values.json;
    system += "\n\nRespond with valid JSON only. No markdown, no code fences, no explanation. If the result is a list, return a JSON array with one element per item — never a single string with newlines.";
    if (typeof jsonMode === "string") system += `\n\nThe response must match this shape exactly: ${jsonMode}`;
  }

  const fileContext  = values.file ? readFileSync(values.file, "utf8").trim() : null;
  const stdinData    = process.stdin.isTTY ? null : await Bun.stdin.text().then(t => t.trim() || null);
  const parts        = [fileContext, stdinData].filter(Boolean) as string[];
  const context      = parts.join("\n\n");
  const promptArg    = positionals.join(" ");
  const prompt       = context && promptArg ? `${context}\n\n${promptArg}` : context || promptArg;

  if (!prompt) {
    console.error("usage: infer [options] [prompt]\n\nOptions:\n  -m MODEL  -u URL  -k KEY  -s TEXT  -r ROLE  -f FILE  -j [SHAPE]  -v\n  --no-sandbox  --allow-network\n  config show|get|set|unset");
    process.exit(1);
  }

  const code = await run({
    prompt,
    url:          values.url ?? cfg.url,
    model:        values.model ?? cfg.model,
    apiKey:       values["api-key"] ?? cfg.api_key,
    system,
    verbose:      values.verbose ?? false,
    jsonMode,
    sandbox:      !(values["no-sandbox"] ?? false),
    allowNetwork: values["allow-network"] ?? false,
  });

  process.exit(code);
}
