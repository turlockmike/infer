/**
 * E2E tests — hit a real LLM endpoint.
 * Tests are automatically skipped when the configured model is unreachable.
 *
 * Run manually:
 *   bun run test:e2e
 *
 * Configure via env:
 *   INFER_URL   (default: http://localhost:11434/v1)
 *   INFER_MODEL (default: gemma4:latest)
 *   INFER_KEY   (default: ollama)
 */

import { describe, it, expect } from "bun:test";
import { EventEmitter } from "events";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { run, runRepl, loadConfig, loadSession, appendToSession } from "./infer";

const cfg = loadConfig();
const E2E_URL   = process.env.INFER_URL   ?? cfg.url;
const E2E_MODEL = process.env.INFER_MODEL ?? cfg.model;
const E2E_KEY   = process.env.INFER_KEY   ?? cfg.api_key;

async function isModelAvailable(): Promise<boolean> {
  try {
    const resp = await fetch(`${E2E_URL}/models`, {
      headers: { Authorization: `Bearer ${E2E_KEY}` },
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) return false;
    const data = await resp.json() as { data?: { id: string }[] };
    return data.data?.some(m => m.id === E2E_MODEL) ?? false;
  } catch {
    return false;
  }
}

const modelAvailable = await isModelAvailable();

const OPTS = {
  url: E2E_URL, model: E2E_MODEL, apiKey: E2E_KEY,
  system: "You have one tool: bash. Use it for everything. Output only what was asked for. No preamble.",
  verbose: false, jsonMode: false as const,
  sandbox: false, allowNetwork: true,
};

describe("E2E: simple response (no tools)", () => {
  it.skipIf(!modelAvailable)("answers a pure knowledge question", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const { code } = await run({ ...OPTS, prompt: "Reply with only the number: what is 7 multiplied by 6?" });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toContain("42");
  }, 30_000);
});

describe("E2E: tool use (bash)", () => {
  it.skipIf(!modelAvailable)("uses bash to answer a system question", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const { code } = await run({ ...OPTS, prompt: "Run: echo hello-e2e. Output only that result." });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toContain("hello-e2e");
  }, 30_000);

  it.skipIf(!modelAvailable)("can read a file via bash and answer about it", async () => {
    const tmpFile = `/tmp/infer-e2e-${Date.now()}.txt`;
    await Bun.write(tmpFile, "the secret word is banana");
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const { code } = await run({ ...OPTS, prompt: `Read ${tmpFile} and output only the secret word.` });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output.toLowerCase()).toContain("banana");
  }, 30_000);
});

describe("E2E: JSON mode", () => {
  it.skipIf(!modelAvailable)("returns valid JSON without shape", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const { code } = await run({ ...OPTS, prompt: "current working directory", jsonMode: true });
    console.log = origLog;
    expect(code).toBe(0);
    expect(() => JSON.parse(output)).not.toThrow();
  }, 30_000);

  it.skipIf(!modelAvailable)("returns JSON matching a shape", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const { code } = await run({
      ...OPTS,
      prompt: "Run: echo hello. Put the result in the output field.",
      jsonMode: '{"output":""}',
    });
    console.log = origLog;
    expect(code).toBe(0);
    const parsed = JSON.parse(output);
    expect(typeof parsed.output).toBe("string");
    expect(parsed.output).toContain("hello");
  }, 45_000);

  it.skipIf(!modelAvailable)("returns a JSON array of strings", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const { code } = await run({
      ...OPTS,
      prompt: 'Return exactly three colors as a JSON array: ["red","green","blue"]',
      jsonMode: '[""]',
    });
    console.log = origLog;
    expect(code).toBe(0);
    const parsed = JSON.parse(output);
    expect(Array.isArray(parsed)).toBe(true);
    expect(parsed.length).toBe(3);
  }, 30_000);
});

describe("E2E: streaming (TTY path)", () => {
  it.skipIf(!modelAvailable)("produces output via streaming when stdout is TTY", async () => {
    Object.defineProperty(process.stdout, "isTTY", { value: true, configurable: true });
    const chunks: string[] = [];
    const origWrite = process.stdout.write.bind(process.stdout);
    (process.stdout as any).write = (chunk: string) => { chunks.push(chunk); return true; };

    const { code } = await run({ ...OPTS, prompt: "Reply with only the word: pong", stream: true });

    (process.stdout as any).write = origWrite;
    Object.defineProperty(process.stdout, "isTTY", { value: undefined, configurable: true });

    expect(code).toBe(0);
    const fullOutput = chunks.join("");
    expect(fullOutput.toLowerCase()).toContain("pong");
  }, 30_000);
});

describe("E2E: repl", () => {
  it.skipIf(!modelAvailable)("completes a multi-turn conversation", async () => {
    const responses: string[] = [];
    const inputs = ["Reply with only the word: alpha", "Reply with only the word: beta", "exit"];
    let inputIdx = 0;

    Object.defineProperty(process.stdout, "isTTY", { value: true, configurable: true });
    const origWrite = process.stdout.write.bind(process.stdout);
    let currentResponse = "";
    (process.stdout as any).write = (chunk: string) => {
      if (chunk === "\n" && currentResponse) { responses.push(currentResponse.trim()); currentResponse = ""; }
      else if (!chunk.startsWith("infer repl") && chunk !== "> " && chunk !== "\n") currentResponse += chunk;
      return true;
    };

    // Inject fake rl directly — avoids readonly ES module patching issue
    const fakeRl = new EventEmitter() as any;
    fakeRl.question = (_prompt: string, cb: (ans: string) => void) => {
      const answer = inputs[inputIdx++] ?? "exit";
      setTimeout(() => cb(answer), 0);
    };
    fakeRl.close = () => {};
    fakeRl.on = (event: string, cb: () => void) => {
      if (event !== "SIGINT") EventEmitter.prototype.on.call(fakeRl, event, cb);
      return fakeRl;
    };

    await runRepl({ ...OPTS, system: "Output only what was asked. No preamble.", _rl: fakeRl });

    (process.stdout as any).write = origWrite;
    Object.defineProperty(process.stdout, "isTTY", { value: undefined, configurable: true });

    expect(responses.some(r => r.toLowerCase().includes("alpha"))).toBe(true);
    expect(responses.some(r => r.toLowerCase().includes("beta"))).toBe(true);
  }, 60_000);
});

describe("E2E: session round-trip", () => {
  it.skipIf(!modelAvailable)("second call sees context from first call", async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), "infer-e2e-session-"));
    const sessionFile = join(tmpDir, "session.jsonl");

    try {
      // Turn 1: plant a secret word
      const origLog = console.log;
      console.log = () => {};
      const { messages: m1 } = await run({
        ...OPTS,
        prompt: 'Remember this: the secret word is "tangerine". Reply with only: "Noted."',
      });
      console.log = origLog;

      // Save turn to session (skip system message at index 0)
      appendToSession(sessionFile, m1.slice(1));

      // Turn 2: ask what the secret was, with prior context loaded
      let output2 = "";
      const origLog2 = console.log;
      console.log = (s: string) => { output2 = s; };
      const { code } = await run({
        ...OPTS,
        prompt: "What was the secret word I told you? Reply with only the word.",
        initialMessages: loadSession(sessionFile),
      });
      console.log = origLog2;

      expect(code).toBe(0);
      expect(output2.toLowerCase()).toContain("tangerine");
    } finally {
      rmSync(tmpDir, { recursive: true });
    }
  }, 60_000);
});
