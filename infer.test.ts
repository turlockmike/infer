import { describe, it, expect, mock, beforeEach } from "bun:test";
import { validateShape, run } from "./infer";

// --- validateShape ---
describe("validateShape", () => {
  it("string match", () => expect(validateShape("hello", "")).toBeNull());
  it("string mismatch", () => expect(validateShape(123, "")).not.toBeNull());
  it("number match", () => expect(validateShape(42, 0)).toBeNull());
  it("number mismatch", () => expect(validateShape("42", 0)).not.toBeNull());
  it("boolean match", () => expect(validateShape(true, false)).toBeNull());
  it("boolean mismatch", () => expect(validateShape("true", false)).not.toBeNull());

  it("object match", () => expect(validateShape({ name: "alice", age: 30 }, { name: "", age: 0 })).toBeNull());
  it("object missing key", () => expect(validateShape({ name: "alice" }, { name: "", age: 0 })).toContain("age"));
  it("object wrong type", () => expect(validateShape({ name: 123 }, { name: "" })).toContain("name"));

  it("array match", () => expect(validateShape(["a", "b"], [""])).toBeNull());
  it("array mismatch", () => expect(validateShape(["a", 2], [""])).not.toBeNull());
  it("array not array", () => expect(validateShape("nope", [""])).toContain("expected array"));

  it("nested object", () => {
    const shape = { user: { name: "", score: 0 } };
    expect(validateShape({ user: { name: "bob", score: 99 } }, shape)).toBeNull();
    expect(validateShape({ user: { name: "bob" } }, shape)).toContain("score");
  });
});

// --- run() ---
const mockCreate = mock(() => Promise.resolve({
  choices: [{ message: { content: "Paris", tool_calls: null } }],
  usage: { completion_tokens: 5, prompt_tokens: 20 },
}));

mock.module("openai", () => ({
  default: class {
    chat = { completions: { create: mockCreate } };
  },
}));

mock.module("just-bash", () => ({
  Bash: class { exec = async () => ({ stdout: "mocked", stderr: "" }); },
  MountableFs: class { mount = () => {}; },
  InMemoryFs: class {},
  ReadWriteFs: class {},
}));


const BASE_OPTS = {
  url: "http://x", model: "m", apiKey: "x", system: "s",
  verbose: false, jsonMode: false as const, sandbox: false, allowNetwork: false,
};

describe("run()", () => {
  beforeEach(() => mockCreate.mockClear());

  it("returns 0 on simple response", async () => {
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "Paris", tool_calls: null } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const code = await run({ ...BASE_OPTS, prompt: "capital of France?" });
    expect(code).toBe(0);
  });

  it("handles tool call then answer", async () => {
    mockCreate
      .mockResolvedValueOnce({
        choices: [{ message: { content: "", tool_calls: [{ id: "c1", function: { name: "bash", arguments: '{"command":"date"}' } }] } }],
        usage: { completion_tokens: 5, prompt_tokens: 20 },
      })
      .mockResolvedValueOnce({
        choices: [{ message: { content: "today", tool_calls: null } }],
        usage: { completion_tokens: 3, prompt_tokens: 30 },
      });
    const code = await run({ ...BASE_OPTS, prompt: "what day is it?", sandbox: true });
    expect(code).toBe(0);
    expect(mockCreate).toHaveBeenCalledTimes(2);
  });

  it("retries on invalid JSON", async () => {
    mockCreate
      .mockResolvedValueOnce({
        choices: [{ message: { content: "not json", tool_calls: null } }],
        usage: { completion_tokens: 2, prompt_tokens: 10 },
      })
      .mockResolvedValueOnce({
        choices: [{ message: { content: '{"ok":true}', tool_calls: null } }],
        usage: { completion_tokens: 5, prompt_tokens: 20 },
      });
    const code = await run({ ...BASE_OPTS, prompt: "get data", jsonMode: true });
    expect(code).toBe(0);
    expect(mockCreate).toHaveBeenCalledTimes(2);
  });

  it("retries on shape mismatch", async () => {
    mockCreate
      .mockResolvedValueOnce({
        choices: [{ message: { content: '{"name":123}', tool_calls: null } }],
        usage: { completion_tokens: 5, prompt_tokens: 20 },
      })
      .mockResolvedValueOnce({
        choices: [{ message: { content: '{"name":"alice"}', tool_calls: null } }],
        usage: { completion_tokens: 5, prompt_tokens: 20 },
      });
    const code = await run({ ...BASE_OPTS, prompt: "get user", jsonMode: '{"name":""}' });
    expect(code).toBe(0);
    expect(mockCreate).toHaveBeenCalledTimes(2);
  });

  it("returns 1 when max steps reached", async () => {
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: "", tool_calls: [{ id: "c1", function: { name: "bash", arguments: '{"command":"echo hi"}' } }] } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const code = await run({ ...BASE_OPTS, prompt: "loop", maxSteps: 3, sandbox: true });
    expect(code).toBe(1);
    expect(mockCreate).toHaveBeenCalledTimes(3);
  });
});
