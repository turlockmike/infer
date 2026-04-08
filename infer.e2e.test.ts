/**
 * E2E tests — hit a real LLM endpoint.
 * Not included in CI. Run manually:
 *
 *   bun test:e2e
 *   bun test infer.e2e.test.ts
 *
 * Requires a running OpenAI-compatible server. Configure via env:
 *   INFER_URL   (default: http://localhost:11434/v1)
 *   INFER_MODEL (default: gemma4:latest)
 *   INFER_KEY   (default: ollama)
 */

import { describe, it, expect } from "bun:test";
import { run, runRepl } from "./infer";

const E2E_URL   = process.env.INFER_URL   ?? "http://localhost:11434/v1";
const E2E_MODEL = process.env.INFER_MODEL ?? "gemma4:latest";
const E2E_KEY   = process.env.INFER_KEY   ?? "ollama";

const OPTS = {
  url: E2E_URL, model: E2E_MODEL, apiKey: E2E_KEY,
  system: "You have one tool: bash. Use it for everything. Output only what was asked for. No preamble.",
  verbose: false, jsonMode: false as const,
  sandbox: false, allowNetwork: false,
};

describe("E2E: simple response (no tools)", () => {
  it("answers a pure knowledge question", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const code = await run({ ...OPTS, prompt: "Reply with only the number: what is 7 multiplied by 6?" });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toContain("42");
  }, 30_000);
});

describe("E2E: tool use (bash)", () => {
  it("uses bash to answer a system question", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const code = await run({ ...OPTS, prompt: "Run: echo hello-e2e. Output only that result." });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toContain("hello-e2e");
  }, 30_000);

  it("can read a file via bash and answer about it", async () => {
    const tmpFile = `/tmp/infer-e2e-${Date.now()}.txt`;
    await Bun.write(tmpFile, "the secret word is banana");
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const code = await run({ ...OPTS, prompt: `Read ${tmpFile} and output only the secret word.` });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output.toLowerCase()).toContain("banana");
  }, 30_000);
});

describe("E2E: JSON mode", () => {
  it("returns valid JSON without shape", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const code = await run({ ...OPTS, prompt: "current working directory", jsonMode: true });
    console.log = origLog;
    expect(code).toBe(0);
    expect(() => JSON.parse(output)).not.toThrow();
  }, 30_000);

  it("returns JSON matching a shape", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const code = await run({
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

  it("returns a JSON array of strings", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    const code = await run({
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
  it("produces output via streaming when stdout is TTY", async () => {
    Object.defineProperty(process.stdout, "isTTY", { value: true, configurable: true });
    const chunks: string[] = [];
    const origWrite = process.stdout.write.bind(process.stdout);
    (process.stdout as any).write = (chunk: string) => { chunks.push(chunk); return true; };

    const code = await run({ ...OPTS, prompt: "Reply with only the word: pong" });

    (process.stdout as any).write = origWrite;
    Object.defineProperty(process.stdout, "isTTY", { value: undefined, configurable: true });

    expect(code).toBe(0);
    const fullOutput = chunks.join("");
    expect(fullOutput.toLowerCase()).toContain("pong");
  }, 30_000);
});

describe("E2E: repl", () => {
  it("completes a multi-turn conversation", async () => {
    const responses: string[] = [];
    const inputs = ["Reply with only the word: alpha", "Reply with only the word: beta", "exit"];
    let inputIdx = 0;

    // Mock readline
    const { createInterface } = await import("readline");
    const origCreate = createInterface;
    (globalThis as any).__rl_mock_inputs = inputs;

    Object.defineProperty(process.stdout, "isTTY", { value: true, configurable: true });
    const origWrite = process.stdout.write.bind(process.stdout);
    let currentResponse = "";
    (process.stdout as any).write = (chunk: string) => {
      if (chunk === "\n" && currentResponse) { responses.push(currentResponse.trim()); currentResponse = ""; }
      else if (!chunk.startsWith("infer repl") && chunk !== "> " && chunk !== "\n") currentResponse += chunk;
      return true;
    };

    // Stub createInterface to feed scripted inputs
    const { EventEmitter } = await import("events");
    const fakeRl = new EventEmitter() as any;
    fakeRl.question = (_prompt: string, cb: (ans: string) => void) => {
      const answer = inputs[inputIdx++] ?? "exit";
      setTimeout(() => cb(answer), 0);
    };
    fakeRl.close = () => {};
    fakeRl.on = (event: string, cb: () => void) => { if (event !== "SIGINT") EventEmitter.prototype.on.call(fakeRl, event, cb); return fakeRl; };

    const rlModule = await import("readline");
    const origCI = rlModule.createInterface;
    (rlModule as any).createInterface = () => fakeRl;

    await runRepl({ ...OPTS, system: "Output only what was asked. No preamble." });

    (rlModule as any).createInterface = origCI;
    (process.stdout as any).write = origWrite;
    Object.defineProperty(process.stdout, "isTTY", { value: undefined, configurable: true });

    expect(responses.some(r => r.toLowerCase().includes("alpha"))).toBe(true);
    expect(responses.some(r => r.toLowerCase().includes("beta"))).toBe(true);
  }, 60_000);
});
