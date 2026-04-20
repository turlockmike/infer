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
 *   infer --no-network "block network in sandbox"
 *   infer --no-sandbox "unrestricted bash"
 */

import OpenAI from "openai";
import { existsSync, readFileSync, writeFileSync, mkdirSync, appendFileSync, unlinkSync } from "fs";
import { extname, join } from "path";
import { homedir } from "os";
import { spawnSync } from "child_process";
import { parseArgs } from "util";
import { createInterface } from "readline";

const VERSION = "2.0.0";

// --- Config paths ---
// _configDirOverride is only set in tests to inject a temp dir.
let _configDirOverride: string | undefined;
export function _setConfigDir(dir: string | undefined) { _configDirOverride = dir; }
function getConfigDir(): string  { return _configDirOverride ?? join(homedir(), ".config", "infer"); }
function getGlobalConfig(): string { return join(getConfigDir(), "config.json"); }
function getGlobalSystem(): string { return join(getConfigDir(), "system.md"); }
function getGlobalRoles(): string  { return join(getConfigDir(), "roles"); }
function getProfilesDir(): string  { return join(getConfigDir(), "profiles"); }
const LOCAL_CONFIG  = ".infer.json";
const LOCAL_SYSTEM  = ".infer.md";

const VALID_KEYS = ["url", "model", "api_key"] as const;
type ConfigKey = typeof VALID_KEYS[number];

const DEFAULTS = {
  url:     "http://localhost:11434/v1",
  model:   "gemma4:latest",
  api_key: "ollama",
};

const DEFAULT_SYSTEM = `You have one tool: bash. Use it when the task requires system access — reading files, running commands, checking state. Answer directly when no system access is needed.

Never claim you cannot reach an address, port, or service without first attempting it with bash. Always try with curl before saying anything is unreachable. This applies to local network addresses, private IPs, and LAN services — try curl first, report what actually happened.

The user cannot see bash tool output — only your final text response is shown to them. Always include the relevant output, data, or findings directly in your response. Never say "I ran the command" without also reporting what it returned.

Your final message is the output of this program — it will be printed to stdout and may be piped into other commands. Be concise and output only what was asked for. No preamble, no commentary.

File conventions:
- Read files with: cat -n <path>
- List directories with: ls -la <path>
- Write files: cat the file first if it exists, then write full content with tee <path> <<'EOF'\\n...\\nEOF
- Never overwrite a file without reading it first unless explicitly told to.

/no_think`;

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

function profilePath(name: string): string {
  return join(getProfilesDir(), `${name}.json`);
}

function readGlobalConfig(): Record<string, any> {
  const p = getGlobalConfig();
  return existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : {};
}

function writeGlobalConfig(data: Record<string, any>) {
  mkdirSync(getConfigDir(), { recursive: true });
  writeFileSync(getGlobalConfig(), JSON.stringify(data, null, 2));
}

export function loadConfig(): typeof DEFAULTS & { system: string; profile?: string } {
  let cfg: Record<string, any> = { ...DEFAULTS };
  const systemParts: string[] = [];

  // 1. Global config (read first to get the active profile name before applying its values)
  const globalData = readGlobalConfig();

  // 2. Active profile — layer in before global config fields so global wins
  if (globalData.profile && typeof globalData.profile === "string") {
    const pPath = profilePath(globalData.profile);
    if (existsSync(pPath)) {
      const profileData = JSON.parse(readFileSync(pPath, "utf8"));
      for (const k of VALID_KEYS) {
        if (profileData[k] !== undefined) cfg[k] = profileData[k];
      }
    }
  }

  // 3. Global config fields (url/model/api_key override profile)
  for (const k of VALID_KEYS) {
    if (globalData[k] !== undefined) cfg[k] = globalData[k];
  }
  if (globalData.profile) cfg.profile = globalData.profile;

  // 4. Local .infer.json
  if (existsSync(LOCAL_CONFIG)) Object.assign(cfg, JSON.parse(readFileSync(LOCAL_CONFIG, "utf8")));

  // 5. System prompt files
  const globalSystem = getGlobalSystem();
  if (existsSync(globalSystem)) systemParts.push(readFileSync(globalSystem, "utf8").trim());
  if (existsSync(LOCAL_SYSTEM))  systemParts.push(readFileSync(LOCAL_SYSTEM, "utf8").trim());

  // 6. Env var overrides — INFER_URL, INFER_MODEL, INFER_API_KEY
  if (process.env.INFER_URL)     cfg.url     = process.env.INFER_URL;
  if (process.env.INFER_MODEL)   cfg.model   = process.env.INFER_MODEL;
  if (process.env.INFER_API_KEY) cfg.api_key = process.env.INFER_API_KEY;

  return {
    url:     cfg.url,
    model:   cfg.model,
    api_key: cfg.api_key,
    profile: cfg.profile,
    system:  systemParts.length ? systemParts.join("\n\n") : DEFAULT_SYSTEM,
  };
}

function saveConfig(updates: Partial<Record<ConfigKey, string>>) {
  const current = readGlobalConfig();
  writeGlobalConfig({ ...current, ...updates });
}

export function runConfigCmd(args: string[]) {
  const [sub, ...rest] = args;

  // --- profile subcommands ---
  if (sub === "profile") {
    const [profileSub, ...profileRest] = rest;

    if (profileSub === "list") {
      const globalData = readGlobalConfig();
      const active = globalData.profile as string | undefined;
      const profilesDir = getProfilesDir();
      if (!existsSync(profilesDir)) return; // no profiles, print nothing
      const { readdirSync } = require("fs") as typeof import("fs");
      const files = readdirSync(profilesDir).filter((f: string) => f.endsWith(".json"));
      for (const f of files) {
        const name = f.replace(/\.json$/, "");
        console.log(name === active ? `${name} (active)` : name);
      }
      return;
    }

    if (profileSub === "show") {
      const [name] = profileRest;
      if (!name) { console.error("usage: infer config profile show <name>"); process.exit(1); }
      const pPath = profilePath(name);
      if (!existsSync(pPath)) { console.error(`infer config: profile '${name}' not found`); process.exit(1); }
      const data = JSON.parse(readFileSync(pPath, "utf8"));
      for (const k of VALID_KEYS) {
        if (data[k] !== undefined) console.log(`${k}=${data[k]}`);
      }
      return;
    }

    if (profileSub === "add") {
      const [name, ...flagArgs] = profileRest;
      if (!name) { console.error("usage: infer config profile add <name> [--url URL] [--model MODEL] [--api-key KEY]"); process.exit(1); }
      const pPath = profilePath(name);
      if (existsSync(pPath)) { console.error(`infer config: profile '${name}' already exists (use 'edit' to modify)`); process.exit(1); }
      const fields = parseProfileFlags(flagArgs);
      if (Object.keys(fields).length === 0) { console.error("infer config: at least one of --url, --model, --api-key required"); process.exit(1); }
      mkdirSync(getProfilesDir(), { recursive: true });
      writeFileSync(pPath, JSON.stringify(fields, null, 2));
      console.log(`profile '${name}' created`);
      return;
    }

    if (profileSub === "edit") {
      const [name, ...flagArgs] = profileRest;
      if (!name) { console.error("usage: infer config profile edit <name> [--url URL] [--model MODEL] [--api-key KEY]"); process.exit(1); }
      const pPath = profilePath(name);
      if (!existsSync(pPath)) { console.error(`infer config: profile '${name}' not found`); process.exit(1); }
      const current = JSON.parse(readFileSync(pPath, "utf8"));
      const fields = parseProfileFlags(flagArgs);
      if (Object.keys(fields).length === 0) { console.error("infer config: at least one of --url, --model, --api-key required"); process.exit(1); }
      writeFileSync(pPath, JSON.stringify({ ...current, ...fields }, null, 2));
      console.log(`profile '${name}' updated`);
      return;
    }

    if (profileSub === "rm") {
      const [name] = profileRest;
      if (!name) { console.error("usage: infer config profile rm <name>"); process.exit(1); }
      const pPath = profilePath(name);
      if (!existsSync(pPath)) { console.error(`infer config: profile '${name}' not found`); process.exit(1); }
      unlinkSync(pPath);
      // Clear active profile if it was this one
      const globalData = readGlobalConfig();
      if (globalData.profile === name) {
        delete globalData.profile;
        writeGlobalConfig(globalData);
      }
      console.log(`profile '${name}' removed`);
      return;
    }

    console.error("usage: infer config profile <list|show|add|edit|rm> ...");
    process.exit(1);
  }

  // --- use <name> / use --none ---
  if (sub === "use") {
    const [target] = rest;
    if (target === "--none") {
      const globalData = readGlobalConfig();
      delete globalData.profile;
      writeGlobalConfig(globalData);
      console.log("active profile cleared");
      return;
    }
    if (!target) { console.error("usage: infer config use <name> | --none"); process.exit(1); }
    if (!existsSync(profilePath(target))) { console.error(`infer config: profile '${target}' not found`); process.exit(1); }
    const globalData = readGlobalConfig();
    globalData.profile = target;
    writeGlobalConfig(globalData);
    console.log(`profile '${target}' activated`);
    return;
  }

  // --- existing subcommands ---
  const [key, value] = rest;
  const cfg = loadConfig();

  if (sub === "show") {
    console.log(`profile=${cfg.profile ?? "(none)"}`);
    console.log(`url=${cfg.url}`);
    console.log(`model=${cfg.model}`);
    console.log(`api_key=${cfg.api_key}`);

  } else if (sub === "get") {
    if (!VALID_KEYS.includes(key as ConfigKey)) { console.error(`infer config: unknown key '${key}'. Valid: ${VALID_KEYS.join(", ")}`); process.exit(1); }
    console.log((cfg as any)[key] ?? "");

  } else if (sub === "set") {
    if (!VALID_KEYS.includes(key as ConfigKey)) { console.error(`infer config: unknown key '${key}'. Valid: ${VALID_KEYS.join(", ")}`); process.exit(1); }
    saveConfig({ [key]: value } as any);
    console.log(`${key}=${value}`);

  } else if (sub === "unset") {
    const globalData = readGlobalConfig();
    // allow unsetting 'profile' as a special key
    if (key === "profile") {
      delete globalData.profile;
      writeGlobalConfig(globalData);
      console.log("unset profile");
    } else {
      if (!VALID_KEYS.includes(key as ConfigKey)) { console.error(`infer config: unknown key '${key}'. Valid: ${VALID_KEYS.join(", ")}, profile`); process.exit(1); }
      delete globalData[key];
      writeGlobalConfig(globalData);
      console.log(`unset ${key}`);
    }

  } else {
    console.error("usage: infer config <show|get|set|unset|use|profile> ...");
    process.exit(1);
  }
}

// Parse --url / --model / --api-key flags from a flat arg array
function parseProfileFlags(flagArgs: string[]): Partial<Record<ConfigKey, string>> {
  const result: Partial<Record<ConfigKey, string>> = {};
  for (let i = 0; i < flagArgs.length; i++) {
    const arg = flagArgs[i];
    if (arg === "--url"     && flagArgs[i + 1]) { result.url     = flagArgs[++i]; }
    else if (arg === "--model"   && flagArgs[i + 1]) { result.model   = flagArgs[++i]; }
    else if (arg === "--api-key" && flagArgs[i + 1]) { result.api_key = flagArgs[++i]; }
  }
  return result;
}

// --- Role ---
function loadRole(name: string): string {
  const path = join(getGlobalRoles(), `${name}.md`);
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

// --- Images ---
// Map common image extensions to MIME types. Data URLs in OpenAI-compatible APIs
// need the MIME type correct — providers use it to dispatch to the right decoder.
const IMAGE_MIME: Record<string, string> = {
  ".jpg":  "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png":  "image/png",
  ".gif":  "image/gif",
  ".webp": "image/webp",
};

/**
 * Read an image file and return an OpenAI-compatible image_url content part
 * using a base64 data URL. Works with any provider that accepts the OpenAI
 * multimodal content schema — Ollama, OpenAI, OpenRouter, LM Studio, etc.
 *
 * Errors are thrown (not process.exit) so callers can decide how to handle them.
 */
export function encodeImagePart(path: string): { type: "image_url"; image_url: { url: string } } {
  if (!existsSync(path)) throw new Error(`infer: image file not found: ${path}`);
  const ext = extname(path).toLowerCase();
  const mime = IMAGE_MIME[ext];
  if (!mime) throw new Error(`infer: unsupported image extension '${ext}' (supported: ${Object.keys(IMAGE_MIME).join(", ")})`);
  const b64 = readFileSync(path).toString("base64");
  return { type: "image_url", image_url: { url: `data:${mime};base64,${b64}` } };
}

// Strip chain-of-thought blocks that thinking models emit before their final response.
// Most open-source thinking models (DeepSeek-R1, Qwen3, GLM-Z1/5.1, nemotron-cascade)
// adopted <think>...</think>. A few use the longer <thinking> variant.
// /no_think in the system prompt suppresses generation for qwen3/GLM; stripping is
// a defensive fallback for models that ignore it or use custom suppression tokens.
function stripThinking(content: string): string {
  return content
    .replace(/<thinking>[\s\S]*?<\/thinking>/gi, "")
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .trim();
}

// --- Session (JSONL) ---
// System message is config, not conversation — excluded from session files.
// Each line is a full OpenAI message object: {"role":"user","content":"..."} etc.

export function loadSession(path: string): OpenAI.Chat.ChatCompletionMessageParam[] {
  if (!existsSync(path)) return [];
  return readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map(l => JSON.parse(l) as OpenAI.Chat.ChatCompletionMessageParam)
    .filter(m => m.role !== "system");
}

export function appendToSession(path: string, messages: OpenAI.Chat.ChatCompletionMessageParam[]) {
  const lines = messages.map(m => JSON.stringify(m)).join("\n") + "\n";
  appendFileSync(path, lines, "utf8");
}

// --- Bash execution ---
export async function execBash(cmd: string, sandbox: boolean, allowNetwork: boolean, cwd: string): Promise<string> {
  const unsandboxed = () => {
    const result = spawnSync(cmd, { shell: true, cwd, encoding: "utf8" });
    return ((result.stdout ?? "") + (result.stderr ?? "")).trim() || "(no output)";
  };

  if (!sandbox) return unsandboxed();

  if (process.platform === "darwin") {
    // macOS Seatbelt — real binaries, OS-enforced write restriction
    const realCwd = spawnSync("realpath", [cwd], { encoding: "utf8" }).stdout.trim();
    const profile = [
      "(version 1)", "(allow default)",
      allowNetwork ? "" : "(deny network*)",
      "(deny file-write*)",
      `(allow file-write* (subpath "/private/tmp"))`,
      `(allow file-write* (subpath "${realCwd}"))`,
    ].filter(Boolean).join("");
    const result = spawnSync("sandbox-exec", ["-p", profile, "bash", "-c", cmd], { cwd, encoding: "utf8" });
    return ((result.stdout ?? "") + (result.stderr ?? "")).trim() || "(no output)";
  }

  if (process.platform === "linux") {
    // Linux: use bwrap (bubblewrap) if available
    const bwrapBin = spawnSync("which", ["bwrap"], { encoding: "utf8" }).stdout.trim();
    if (bwrapBin) {
      const realCwd = spawnSync("realpath", [cwd], { encoding: "utf8" }).stdout.trim();
      const args = [
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--bind", realCwd, realCwd,
        ...(allowNetwork ? [] : ["--unshare-net"]),
        "bash", "-c", cmd,
      ];
      const result = spawnSync("bwrap", args, { cwd, encoding: "utf8" });
      return ((result.stdout ?? "") + (result.stderr ?? "")).trim() || "(no output)";
    }
  }

  // Fallback: no sandbox available on this platform
  return unsandboxed();
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
  stream?: boolean;
  maxSteps?: number;
  initialMessages?: OpenAI.Chat.ChatCompletionMessageParam[];
  images?: string[];
}): Promise<{ code: number; messages: OpenAI.Chat.ChatCompletionMessageParam[] }> {
  const { prompt, url, model, apiKey, system, verbose, jsonMode, sandbox, allowNetwork, maxSteps = 10, initialMessages, images } = opts;
  const cwd = process.cwd();
  const client = new OpenAI({ baseURL: url, apiKey });
  // Build user content: multimodal array when images attached, plain string otherwise.
  // Providers that don't support vision will reject the array shape — the error
  // surfaces from the API call, which is the right place for it.
  const userContent: OpenAI.Chat.ChatCompletionUserMessageParam["content"] = images && images.length
    ? [
        ...(prompt ? [{ type: "text" as const, text: prompt }] : []),
        ...images.map(encodeImagePart),
      ]
    : prompt;
  // Prior session messages (no system) + fresh system prepended + new user turn appended
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: system },
    ...(initialMessages ?? []),
    { role: "user", content: userContent },
  ];
  const sessionStart = 1 + (initialMessages?.length ?? 0); // index of new user message

  const stream = (opts.stream ?? false) && !jsonMode;
  if (verbose) process.stderr.write(`model=${model} url=${url} sandbox=${sandbox} stream=${stream}\n`);

  for (let step = 1; step <= maxSteps; step++) {
    let content = "";
    let toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = [];
    // Wrap API call to catch "does not support tools" and give an actionable error
    const callApi = async <T>(fn: () => Promise<T>): Promise<T> => {
      try { return await fn(); } catch (e: any) {
        if (typeof e?.message === "string" && /does not support tools/i.test(e.message)) {
          process.stderr.write(`infer: model '${model}' does not support tool use.\n`);
          process.stderr.write(`  Try a different variant — e.g. a larger quantisation or an instruct model.\n`);
          process.stderr.write(`  Ollama: check 'ollama list' and pick a model tagged as supporting tools.\n`);
          process.exit(1);
        }
        throw e;
      }
    };

    if (stream) {
      const resp = await callApi(() => client.chat.completions.create({ model, messages, tools: TOOLS, stream: true }));
      for await (const chunk of resp) {
        const delta = chunk.choices[0]?.delta;
        if (delta?.content) { process.stdout.write(delta.content); content += delta.content; }
        if (delta?.tool_calls) {
          for (const tc of delta.tool_calls) {
            if (!toolCalls[tc.index]) toolCalls[tc.index] = { id: "", type: "function", function: { name: "", arguments: "" } };
            if (tc.id) toolCalls[tc.index].id += tc.id;
            if (tc.function?.name) toolCalls[tc.index].function.name += tc.function.name;
            if (tc.function?.arguments) toolCalls[tc.index].function.arguments += tc.function.arguments;
          }
        }
      }
      content = stripThinking(content); // strip any thinking tokens from history
      if (content) process.stdout.write("\n");
      else if (!toolCalls.length) process.stderr.write("infer: warning: model returned empty response\n");
    } else {
      const resp = await callApi(() => client.chat.completions.create({ model, messages, tools: TOOLS }));
      const msg = resp.choices[0].message;
      // Some models (e.g. Gemma4) return reasoning in a non-standard field and leave content empty
      content = stripThinking(msg.content || (msg as any).reasoning || "");
      toolCalls = (msg.tool_calls ?? []) as OpenAI.Chat.ChatCompletionMessageToolCall[];
      if (verbose) process.stderr.write(`[${resp.usage?.completion_tokens} tok, prompt=${resp.usage?.prompt_tokens}]\n`);
    }

    if (toolCalls.length) {
      if (stream && content) { /* already printed above */ }
      const assistantMsg: OpenAI.Chat.ChatCompletionMessageParam = { role: "assistant", content, tool_calls: toolCalls };
      messages.push(assistantMsg);
      for (const call of toolCalls) {
        const cmd = JSON.parse(call.function.arguments).command as string;
        const output = await execBash(cmd, sandbox, allowNetwork, cwd);
        if (verbose) { process.stderr.write(`+ ${cmd}\n`); process.stderr.write(`${output}\n`); }
        messages.push({ role: "tool", tool_call_id: call.id, content: output });
      }
    } else {
      // If content is empty with no tool calls, force a final response rather than returning nothing
      if (!content && messages.some(m => m.role === "tool")) {
        messages.push({ role: "user", content: "Please provide your response to the user based on the tool results above." });
        continue;
      }

      if (!content && !stream) process.stderr.write("infer: warning: model returned empty response\n");

      if (!stream) {

        if (jsonMode) {
          let parsed: unknown;
          try {
            parsed = JSON.parse(content);
          } catch (e: any) {
            process.stderr.write(`infer: invalid JSON: ${e.message}\n`);
            const assistantMsg: OpenAI.Chat.ChatCompletionMessageParam = { role: "assistant", content };
            messages.push(assistantMsg);
            messages.push({ role: "user", content: `Your response was not valid JSON. Error: ${e.message}\nYou returned: ${content}\nFix it and respond with valid JSON only. No markdown, no code fences, no explanation.` });
            continue;
          }
          if (typeof jsonMode === "string") {
            const err = validateShape(parsed, JSON.parse(jsonMode));
            if (err) {
              process.stderr.write(`infer: shape mismatch: ${err}\n`);
              const assistantMsg: OpenAI.Chat.ChatCompletionMessageParam = { role: "assistant", content };
              messages.push(assistantMsg);
              messages.push({ role: "user", content: `Your response was invalid. Problem: ${err}\nYou returned: ${content}\nRequired shape: ${jsonMode}\nFix it and respond with valid JSON matching that shape exactly. Nothing else.` });
              continue;
            }
          }
        }
        console.log(content);
      }
      messages.push({ role: "assistant", content });
      return { code: 0, messages };
    }
  }

  process.stderr.write("infer: max steps reached\n");
  return { code: 1, messages };
}

// --- REPL ---
export async function runRepl(opts: {
  url: string;
  model: string;
  apiKey: string;
  system: string;
  verbose: boolean;
  sandbox: boolean;
  allowNetwork: boolean;
  stream?: boolean;
  session?: string;
  _rl?: any; // injectable readline interface (for testing)
}): Promise<void> {
  const { url, model, apiKey, system, verbose, sandbox, allowNetwork, stream = false, session, _rl } = opts;
  const cwd = process.cwd();
  const client = new OpenAI({ baseURL: url, apiKey });

  const priorMessages = session ? loadSession(session) : [];
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: system },
    ...priorMessages,
  ];

  const rl = _rl ?? createInterface({ input: process.stdin, output: process.stdout });
  const sessionNote = session
    ? (priorMessages.length ? ` (${priorMessages.length} messages loaded from ${session})` : ` (new session: ${session})`)
    : "";
  process.stdout.write(`infer repl  ${model}${sessionNote}  Ctrl+C or 'exit' to quit\n\n`);

  const readLine = () => new Promise<string | null>((resolve) => {
    const onClose = () => resolve(null);
    rl.once("close", onClose);
    rl.question("> ", (answer) => {
      rl.removeListener("close", onClose);
      resolve(answer);
    });
  });

  rl.on("SIGINT", () => { process.stdout.write("\n"); rl.close(); });

  while (true) {
    const input = await readLine();
    if (input === null || input.trim() === "exit" || input.trim() === "quit") break;
    if (!input.trim()) continue;

    messages.push({ role: "user", content: input });
    const turnStart = messages.length - 1; // index of this turn's user message

    let replied = false;
    while (!replied) {
      let content = "";
      let toolCalls: OpenAI.Chat.ChatCompletionMessageToolCall[] = [];
      const callApi = async <T>(fn: () => Promise<T>): Promise<T> => {
        try { return await fn(); } catch (e: any) {
          if (typeof e?.message === "string" && /does not support tools/i.test(e.message)) {
            process.stderr.write(`infer: model '${model}' does not support tool use.\n`);
            process.stderr.write(`  Try a different variant — e.g. a larger quantisation or an instruct model.\n`);
            process.stderr.write(`  Ollama: check 'ollama list' and pick a model tagged as supporting tools.\n`);
            process.exit(1);
          }
          throw e;
        }
      };

      if (stream) {
        const resp = await callApi(() => client.chat.completions.create({ model, messages, tools: TOOLS, stream: true }));
        for await (const chunk of resp) {
          const delta = chunk.choices[0]?.delta;
          if (delta?.content) { process.stdout.write(delta.content); content += delta.content; }
          if (delta?.tool_calls) {
            for (const tc of delta.tool_calls) {
              if (!toolCalls[tc.index]) toolCalls[tc.index] = { id: "", type: "function", function: { name: "", arguments: "" } };
              if (tc.id) toolCalls[tc.index].id += tc.id;
              if (tc.function?.name) toolCalls[tc.index].function.name += tc.function.name;
              if (tc.function?.arguments) toolCalls[tc.index].function.arguments += tc.function.arguments;
            }
          }
        }
        content = stripThinking(content); // strip any thinking tokens from history
        if (content) process.stdout.write("\n");
      } else {
        const resp = await callApi(() => client.chat.completions.create({ model, messages, tools: TOOLS }));
        const msg = resp.choices[0].message;
        content = stripThinking(msg.content || (msg as any).reasoning || "");
        toolCalls = (msg.tool_calls ?? []) as OpenAI.Chat.ChatCompletionMessageToolCall[];
        if (verbose) process.stderr.write(`[${resp.usage?.completion_tokens} tok, prompt=${resp.usage?.prompt_tokens}]\n`);
      }

      if (toolCalls.length) {
        const assistantMsg: OpenAI.Chat.ChatCompletionMessageParam = { role: "assistant", content, tool_calls: toolCalls };
        messages.push(assistantMsg);
        for (const call of toolCalls) {
          const cmd = JSON.parse(call.function.arguments).command as string;
          const output = await execBash(cmd, sandbox, allowNetwork, cwd);
          if (verbose) { process.stderr.write(`+ ${cmd}\n`); process.stderr.write(`${output}\n`); }
          messages.push({ role: "tool", tool_call_id: call.id, content: output });
        }
      } else {
        // Force a final response if content is empty after tool calls
        if (!content && messages.some(m => m.role === "tool")) {
          messages.push({ role: "user", content: "Please provide your response to the user based on the tool results above." });
          continue;
        }
        if (!stream) process.stdout.write(content + "\n");
        const assistantMsg: OpenAI.Chat.ChatCompletionMessageParam = { role: "assistant", content };
        messages.push(assistantMsg);
        // Persist this turn: user message + any tool call pairs + final assistant
        if (session) appendToSession(session, messages.slice(turnStart));
        replied = true;
      }
    }

    process.stdout.write("\n");
  }

  rl.close();
}

// --- Entry point ---
if (import.meta.main) {
  const argv = process.argv.slice(2);

  if (argv[0] === "--version" || argv[0] === "-V") {
    console.log(VERSION);
    process.exit(0);
  }

  if (argv[0] === "--help" || argv[0] === "-h") {
    console.log(`infer — pipe-friendly LLM agent harness with a bash tool

Usage:
  infer [OPTIONS] PROMPT
  infer                     enter REPL (interactive mode)
  infer repl                enter REPL explicitly
  infer config <cmd>        manage config

Options:
  -m, --model MODEL         model name (env: INFER_MODEL, default: gemma4:latest)
  -u, --url URL             provider base URL (env: INFER_URL)
  -k, --api-key KEY         API key (env: INFER_API_KEY)
  -s, --system TEXT         system prompt override
  -r, --role NAME           load role from ~/.config/infer/roles/<name>.md
  -f, --file FILE           file to use as context (prepended to prompt)
  -i, --image FILE          attach an image (repeatable); requires a vision model
  -j, --json [SHAPE]        output JSON, optionally validated against a shape
  -S, --session FILE        JSONL session file — load prior turns, append this turn
  -n, --max-steps N         max tool-call iterations per run (default: 10)
  -v, --verbose             show tool calls and token stats on stderr
      --stream              stream tokens as they arrive (default: off)
      --no-sandbox          use real bash (default: sandboxed via sandbox-exec)
      --no-network          block network access inside the sandbox (default: network allowed)
      --allow-network       [deprecated, no-op — network is now allowed by default]
  -h, --help                show this help
  -V, --version             show version

Config commands:
  infer config show                          show resolved config (including active profile)
  infer config get <key>                     get a config value
  infer config set <key> <value>             set a config value
  infer config unset <key>                   unset a config value
  infer config use <name>                    activate a named profile
  infer config use --none                    clear the active profile
  infer config profile list                  list profiles
  infer config profile show <name>           show a profile's values
  infer config profile add <name> [flags]    create a profile (--url, --model, --api-key)
  infer config profile edit <name> [flags]   merge fields into an existing profile
  infer config profile rm <name>             delete a profile

Examples:
  infer "what directory am i in"
  cat crash.log | infer "why did this fail"
  infer -f main.py "explain this"
  infer -i frame.jpg "describe what you see"
  infer -i a.png -i b.png "compare these images"
  infer -j '{"name":"string","pid":0}' "current process info"
  infer --stream "write a poem"
  infer repl
  infer -S /tmp/s.jsonl "first question"
  infer -S /tmp/s.jsonl "follow-up question"
  infer config profile add tower --url http://192.168.4.30:11434/v1 --model gemma4:latest
  infer config use tower
  infer config use --none`);
    process.exit(0);
  }

  if (argv[0] === "config") { runConfigCmd(argv.slice(1)); process.exit(0); }

  const isReplCmd = argv[0] === "repl";
  const cfg = loadConfig();
  const hasConfig = existsSync(getGlobalConfig()) || existsSync(LOCAL_CONFIG)
    || !!(process.env.INFER_URL || process.env.INFER_MODEL || process.env.INFER_API_KEY);
  if (!hasConfig) {
    process.stderr.write("infer: no config found. Set env vars or run:\n");
    process.stderr.write("  export INFER_API_KEY=sk-...   # or INFER_URL / INFER_MODEL\n\n");
    process.stderr.write("  infer config set url http://localhost:11434/v1\n");
    process.stderr.write("  infer config set model gemma4:latest\n");
    process.stderr.write("  infer config set api_key ollama\n\n");
    process.stderr.write("  Proceeding with built-in defaults...\n\n");
  }

  const { values, positionals } = parseArgs({
    args: isReplCmd ? argv.slice(1) : argv,
    options: {
      model:         { type: "string",  short: "m", default: cfg.model },
      url:           { type: "string",  short: "u", default: cfg.url },
      "api-key":     { type: "string",  short: "k", default: cfg.api_key },
      system:        { type: "string",  short: "s", default: cfg.system },
      role:          { type: "string",  short: "r" },
      file:          { type: "string",  short: "f" },
      image:         { type: "string",  short: "i", multiple: true },
      json:          { type: "string",  short: "j" },
      session:       { type: "string",  short: "S" },
      verbose:       { type: "boolean", short: "v", default: false },
      stream:        { type: "boolean", default: false },
      "no-sandbox":  { type: "boolean", default: false },
      "no-network":    { type: "boolean", default: false },
      "allow-network": { type: "boolean", default: false }, // deprecated no-op (kept for back-compat)
      "max-steps":   { type: "string",  short: "n", default: "10" },
    },
    allowPositionals: true,
  });

  let system = values.role ? loadRole(values.role) : (values.system ?? cfg.system);

  let jsonMode: boolean | string = false;
  if (values.json !== undefined) {
    jsonMode = values.json === "" ? true : values.json;
    const jsonInstruction = typeof jsonMode === "string"
      ? `Respond with valid JSON only matching this exact shape: ${jsonMode}. No markdown, no code fences, no explanation.`
      : `Respond with valid JSON only. No markdown, no code fences, no explanation. If the result is a list, return a JSON array with one element per item — never a single string with newlines.`;
    system = jsonInstruction + "\n\n" + system;
  }

  const fileContext  = values.file ? readFileSync(values.file, "utf8").trim() : null;
  const stdinData    = process.stdin.isTTY ? null : await Bun.stdin.text().then(t => t.trim() || null);
  const parts        = [fileContext, stdinData].filter(Boolean) as string[];
  const context      = parts.join("\n\n");
  const promptArg    = positionals.join(" ");
  const basePrompt   = context && promptArg ? `${context}\n\n${promptArg}` : context || promptArg;
  const prompt       = jsonMode && basePrompt
    ? `${basePrompt}\n\nRespond with JSON only.`
    : basePrompt;

  const images = (values.image as string[] | undefined) ?? [];

  // REPL only when there's nothing to infer from — no prompt, no images, and a TTY.
  // An image-only invocation (`infer -r vision -i frame.jpg`) is a valid one-shot
  // call, not an invitation to REPL.
  const replMode = isReplCmd || (!prompt && !images.length && !!process.stdin.isTTY);

  if (replMode) {
    await runRepl({
      url:          values.url ?? cfg.url,
      model:        values.model ?? cfg.model,
      apiKey:       values["api-key"] ?? cfg.api_key,
      system,
      verbose:      values.verbose ?? false,
      stream:       values.stream ?? false,
      sandbox:      !(values["no-sandbox"] ?? false),
      allowNetwork: !(values["no-network"] ?? false),
      session:      values.session,
    });
    process.exit(0);
  }

  if (!prompt && !images.length) {
    console.error("usage: infer [options] [prompt]\n\nOptions:\n  -m MODEL  -u URL  -k KEY  -s TEXT  -r ROLE  -f FILE  -i IMAGE  -j [SHAPE]  -S FILE  -n N  -v\n  --stream  --no-sandbox  --no-network\n  config show|get|set|unset|use|profile  repl");
    process.exit(1);
  }

  const sessionFile = values.session;
  const initialMessages = sessionFile ? loadSession(sessionFile) : undefined;

  const { code, messages: resultMessages } = await run({
    prompt,
    url:             values.url ?? cfg.url,
    model:           values.model ?? cfg.model,
    apiKey:          values["api-key"] ?? cfg.api_key,
    system,
    verbose:         values.verbose ?? false,
    stream:          values.stream ?? false,
    jsonMode,
    sandbox:         !(values["no-sandbox"] ?? false),
    allowNetwork:    !(values["no-network"] ?? false),
    maxSteps:        parseInt(values["max-steps"] ?? "10", 10),
    initialMessages,
    images,
  });

  if (sessionFile) {
    // Append only the new messages from this turn (skip system + prior session messages)
    const newMessages = resultMessages.slice(1 + (initialMessages?.length ?? 0));
    appendToSession(sessionFile, newMessages);
  }

  process.exitCode = code;
}
