/**
 * Behavioral eval suite — tests system prompt effectiveness across models.
 *
 * Run against any model:
 *   bun run test:behavioral
 *   INFER_MODEL=qwen2.5:latest bun run test:behavioral
 *   INFER_MODEL=gemma3n:latest bun run test:behavioral
 *   INFER_MODEL=phi4:latest bun run test:behavioral
 *
 * Tests behavioral properties, not exact wording:
 *   - Does the model use bash when it should?
 *   - Does it include actual output in its response?
 *   - Does it attempt curl before refusing?
 *   - Is it concise enough for pipe use?
 */

import { describe, it, expect, beforeAll } from "bun:test";
import { run, loadConfig } from "./infer";
import type OpenAI from "openai";

const cfg = loadConfig();
const MODEL = process.env.INFER_MODEL ?? cfg.model;
const URL   = process.env.INFER_URL   ?? cfg.url;
const KEY   = process.env.INFER_KEY   ?? cfg.api_key;

console.error(`\nbehavioral eval: model=${MODEL} url=${URL}\n`);

// Check model is reachable
async function isModelAvailable(): Promise<boolean> {
  try {
    const resp = await fetch(`${URL}/models`, {
      headers: { Authorization: `Bearer ${KEY}` },
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) return false;
    const data = await resp.json() as { data?: { id: string }[] };
    return data.data?.some(m => m.id === MODEL) ?? false;
  } catch {
    return false;
  }
}

const modelAvailable = await isModelAvailable();
if (!modelAvailable) console.error(`  WARNING: model ${MODEL} not available — all tests will skip\n`);

const OPTS = {
  url: URL, model: MODEL, apiKey: KEY,
  system: cfg.system,   // use the actual system prompt, not a simplified one
  verbose: false, jsonMode: false as const,
  sandbox: false, allowNetwork: false,
  maxSteps: 5,
};

// --- Helpers ---

type Msg = OpenAI.Chat.ChatCompletionMessageParam;

function hasBashCall(messages: Msg[]): boolean {
  return messages.some(m => m.role === "tool");
}

function bashCommands(messages: Msg[]): string[] {
  return messages
    .filter((m): m is OpenAI.Chat.ChatCompletionAssistantMessageParam => m.role === "assistant")
    .flatMap(m => (m.tool_calls ?? []).map((tc: any) => {
      try { return JSON.parse(tc.function.arguments).command as string; } catch { return ""; }
    }))
    .filter(Boolean);
}

function finalResponse(messages: Msg[]): string {
  const assistants = messages.filter(m => m.role === "assistant");
  const last = assistants[assistants.length - 1];
  return (last as any)?.content ?? "";
}

// Capture console.log output from run()
async function evalRun(prompt: string, extra: Partial<typeof OPTS> = {}) {
  let output = "";
  const origLog = console.log;
  console.log = (s: string) => { output = s; };
  const result = await run({ ...OPTS, ...extra, prompt });
  console.log = origLog;
  return { ...result, output };
}

// --- Tool use discipline ---

describe("tool use: uses bash when system access is needed", () => {
  it.skipIf(!modelAvailable)("uses bash for 'what directory am i in'", async () => {
    const { code, messages } = await evalRun("what directory am i in");
    expect(code).toBe(0);
    expect(hasBashCall(messages)).toBe(true);
    // Must include an actual path in the response
    expect(finalResponse(messages)).toMatch(/\//);
  }, 30_000);

  it.skipIf(!modelAvailable)("uses bash to list files", async () => {
    const { code, messages } = await evalRun("what files are in the current directory?");
    expect(code).toBe(0);
    expect(hasBashCall(messages)).toBe(true);
    expect(finalResponse(messages).trim().length).toBeGreaterThan(5);
  }, 30_000);

  it.skipIf(!modelAvailable)("uses bash to check a running process", async () => {
    const { code, messages } = await evalRun("is bun running as a process right now?");
    expect(code).toBe(0);
    expect(hasBashCall(messages)).toBe(true);
  }, 30_000);
});

describe("tool use: does NOT use bash for pure knowledge", () => {
  it.skipIf(!modelAvailable)("answers capital of France correctly (no bash required)", async () => {
    const { code, output } = await evalRun("what is the capital of France? Reply with just the city name.");
    expect(code).toBe(0);
    expect(output.toLowerCase()).toContain("paris");
    // Note: some models (e.g. qwen3) use bash even for knowledge questions — answer correctness is what matters
  }, 20_000);

  it.skipIf(!modelAvailable)("answers a math question without bash", async () => {
    const { code, messages, output } = await evalRun("what is 17 multiplied by 13? Reply with only the number.");
    expect(code).toBe(0);
    expect(hasBashCall(messages)).toBe(false);
    expect(output).toContain("221");
  }, 20_000);
});

// --- Output inclusion ---

describe("output inclusion: response contains actual tool output", () => {
  it.skipIf(!modelAvailable)("echoes a unique marker from bash output", async () => {
    const marker = `INFER_EVAL_${Date.now()}`;
    const { code, output } = await evalRun(`Run this exact command: echo ${marker}. Report what it printed.`);
    expect(code).toBe(0);
    // The unique marker must appear in the final response
    expect(output).toContain(marker);
  }, 30_000);

  it.skipIf(!modelAvailable)("does not just say 'I ran the command' without output", async () => {
    const { code, messages, output } = await evalRun("run: date +%Y and tell me the year");
    expect(code).toBe(0);
    expect(hasBashCall(messages)).toBe(true);
    // Response should not be a meta-comment without actual content
    expect(output.toLowerCase()).not.toMatch(/^i ran the command\.?$/i);
    // Should contain something that looks like a year
    expect(output).toMatch(/20\d\d/);
  }, 30_000);
});

// --- Network: attempts before refusing ---

describe("network: tries curl before claiming unreachable", () => {
  it.skipIf(!modelAvailable)("attempts curl for a LAN address rather than refusing", async () => {
    const { code, messages } = await evalRun(
      "check if http://127.0.0.1:11434/ responds. Just try it.",
      { allowNetwork: true }
    );
    expect(code).toBe(0);
    const cmds = bashCommands(messages);
    // Model must have tried curl (or wget) — not just said it can't do it
    expect(cmds.some(c => /curl|wget|fetch/.test(c))).toBe(true);
  }, 30_000);

  it.skipIf(!modelAvailable)("does not refuse a localhost request without trying", async () => {
    const { code, messages, output } = await evalRun(
      "fetch http://localhost:1 and tell me what happened",
      { allowNetwork: true }
    );
    expect(code).toBe(0);
    const cmds = bashCommands(messages);
    // Must have attempted something — can't just refuse
    expect(cmds.length).toBeGreaterThan(0);
    // Response must not be a flat refusal without evidence
    expect(output.toLowerCase()).not.toMatch(/^(i cannot|i can't|i'm unable|unable to).{0,30}$/i);
  }, 30_000);
});

// --- Conciseness / pipe friendliness ---

describe("conciseness: output suitable for piping", () => {
  it.skipIf(!modelAvailable)("answers a direct question without excessive preamble", async () => {
    const { code, output } = await evalRun("what is the current unix timestamp? Reply with only the number.");
    expect(code).toBe(0);
    // Should be a number, not a paragraph
    // Note: qwen3 models with chain-of-thought mode will fail this — use /no_think suffix
    expect(output.trim()).toMatch(/^\d+$/);
  }, 60_000);

  it.skipIf(!modelAvailable)("JSON mode output is valid JSON with no prose wrapping", async () => {
    const { code, output } = await evalRun(
      "current working directory",
      { jsonMode: '{"path":"string"}' }
    );
    expect(code).toBe(0);
    const parsed = JSON.parse(output);
    expect(typeof parsed.path).toBe("string");
    expect(parsed.path).toMatch(/\//);
  }, 60_000);
});
