/**
 * Error handling tests — verify infer fails gracefully under bad conditions.
 * Uses mocked OpenAI client (same pattern as infer.test.ts).
 */

import { describe, it, expect, mock, beforeEach, afterEach } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { run, loadSession, appendToSession } from "./infer";

// --- Mocks ---

const mockCreate = mock(() => Promise.resolve({
  choices: [{ message: { content: "ok", tool_calls: null } }],
  usage: { completion_tokens: 2, prompt_tokens: 10 },
}));

mock.module("openai", () => ({
  default: class {
    chat = { completions: { create: mockCreate } };
  },
}));

mock.module("readline", () => ({
  createInterface: () => ({
    question: (_p: string, cb: (s: string) => void) => setTimeout(() => cb("exit"), 0),
    close: () => {}, once: () => {}, on: () => {}, removeListener: () => {},
  }),
}));

const BASE_OPTS = {
  url: "http://x", model: "m", apiKey: "x", system: "s",
  verbose: false, jsonMode: false as const, sandbox: false, allowNetwork: true,
};

// --- Provider errors ---

describe("provider errors", () => {
  beforeEach(() => mockCreate.mockClear());

  it("surfaces 401 auth error without crashing", async () => {
    mockCreate.mockRejectedValueOnce(Object.assign(new Error("Unauthorized"), { status: 401 }));
    await expect(run({ ...BASE_OPTS, prompt: "test" })).rejects.toThrow();
  });

  it("surfaces 429 rate-limit error without crashing", async () => {
    mockCreate.mockRejectedValueOnce(Object.assign(new Error("Rate limit exceeded"), { status: 429 }));
    await expect(run({ ...BASE_OPTS, prompt: "test" })).rejects.toThrow();
  });

  it("surfaces 500 server error without crashing", async () => {
    mockCreate.mockRejectedValueOnce(Object.assign(new Error("Internal server error"), { status: 500 }));
    await expect(run({ ...BASE_OPTS, prompt: "test" })).rejects.toThrow();
  });

  it("surfaces network error (ECONNREFUSED) without crashing", async () => {
    const err = Object.assign(new Error("connect ECONNREFUSED"), { code: "ECONNREFUSED" });
    mockCreate.mockRejectedValueOnce(err);
    await expect(run({ ...BASE_OPTS, prompt: "test" })).rejects.toThrow(/ECONNREFUSED/);
  });

  it("exits with clear message when model does not support tools", async () => {
    const err = new Error("registry.ollama.ai/library/gemma3:latest does not support tools");
    mockCreate.mockRejectedValueOnce(err);
    const stderrChunks: string[] = [];
    const origStderr = process.stderr.write.bind(process.stderr);
    (process.stderr as any).write = (s: string) => { stderrChunks.push(s); return true; };
    const origExit = process.exit;
    let exitCode: number | undefined;
    (process as any).exit = (code: number) => { exitCode = code; throw new Error("process.exit"); };
    try {
      await run({ ...BASE_OPTS, prompt: "test" });
    } catch (e: any) {
      if (e.message !== "process.exit") throw e;
    } finally {
      (process.stderr as any).write = origStderr;
      (process as any).exit = origExit;
    }
    expect(exitCode).toBe(1);
    const stderr = stderrChunks.join("");
    expect(stderr).toContain("does not support tool use");
    expect(stderr).toContain("ollama list");
  });
});

// --- Session file errors ---

describe("session file: malformed input", () => {
  let tmpDir: string;

  beforeEach(() => { tmpDir = mkdtempSync(join(tmpdir(), "infer-errors-test-")); });
  afterEach(() => { rmSync(tmpDir, { recursive: true }); });

  it("loadSession returns empty array for missing file", () => {
    expect(loadSession(join(tmpDir, "nonexistent.jsonl"))).toEqual([]);
  });

  it("loadSession skips blank lines without throwing", () => {
    const f = join(tmpDir, "blanks.jsonl");
    writeFileSync(f, '\n{"role":"user","content":"hello"}\n\n{"role":"assistant","content":"hi"}\n');
    const msgs = loadSession(f);
    expect(msgs).toHaveLength(2);
  });

  it("loadSession throws on a line that is not valid JSON", () => {
    const f = join(tmpDir, "bad.jsonl");
    writeFileSync(f, '{"role":"user","content":"ok"}\nnot-json\n');
    expect(() => loadSession(f)).toThrow();
  });

  it("appendToSession creates file if it does not exist", () => {
    const f = join(tmpDir, "new.jsonl");
    appendToSession(f, [{ role: "user", content: "hi" }]);
    const msgs = loadSession(f);
    expect(msgs).toHaveLength(1);
  });
});

// --- JSON mode edge cases ---

describe("JSON mode: edge cases", () => {
  beforeEach(() => mockCreate.mockClear());

  it("retries up to max steps on persistent JSON failure then returns code 1", async () => {
    // Model keeps returning invalid JSON
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: "definitely not json", tool_calls: null } }],
      usage: { completion_tokens: 4, prompt_tokens: 10 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "get data", jsonMode: true, maxSteps: 3 });
    expect(code).toBe(1);
    // 1 real attempt + 1 retry per correction = at most maxSteps calls
    expect(mockCreate).toHaveBeenCalledTimes(3);
  });

  it("empty string is not valid JSON — retries", async () => {
    mockCreate
      .mockResolvedValueOnce({
        choices: [{ message: { content: "", tool_calls: null } }],
        usage: { completion_tokens: 0, prompt_tokens: 10 },
      })
      .mockResolvedValueOnce({
        choices: [{ message: { content: '{"ok":true}', tool_calls: null } }],
        usage: { completion_tokens: 3, prompt_tokens: 20 },
      });
    const { code } = await run({ ...BASE_OPTS, prompt: "get data", jsonMode: true });
    expect(code).toBe(0);
  });
});

// --- Prompt edge cases ---

describe("run(): prompt edge cases", () => {
  beforeEach(() => mockCreate.mockClear());

  it("handles a very long prompt without error", async () => {
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "ok", tool_calls: null } }],
      usage: { completion_tokens: 2, prompt_tokens: 9999 },
    });
    const longPrompt = "a".repeat(100_000);
    const { code } = await run({ ...BASE_OPTS, prompt: longPrompt });
    expect(code).toBe(0);
  });

  it("handles model returning null content gracefully", async () => {
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: null, tool_calls: null } }],
      usage: { completion_tokens: 0, prompt_tokens: 10 },
    });
    // null content with no tool calls — should trigger the empty-response re-prompt path
    // which then loops once more; give it a real answer on retry
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "recovered", tool_calls: null } }],
      usage: { completion_tokens: 2, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "test", maxSteps: 3 });
    expect(code).toBe(0);
  });
});
