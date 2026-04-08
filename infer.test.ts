import { describe, it, expect, mock, beforeEach, afterEach, spyOn } from "bun:test";
import { validateShape, run, runRepl, loadConfig } from "./infer";

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

  it("empty object shape accepts any object", () => expect(validateShape({ anything: 1 }, {})).toBeNull());
  it("empty array shape accepts any array", () => expect(validateShape([1, 2, 3], [])).toBeNull());
  it("null shape returns null", () => expect(validateShape(null, null)).toBeNull());
  it("array of objects", () => {
    const shape = [{ name: "", score: 0 }];
    expect(validateShape([{ name: "a", score: 1 }], shape)).toBeNull();
    expect(validateShape([{ name: "a" }], shape)).toContain("[0]");
  });
});

// --- loadConfig env vars ---
describe("loadConfig env vars", () => {
  const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
  afterEach(() => {
    process.env.INFER_URL     = saved.url     as string;
    process.env.INFER_MODEL   = saved.model   as string;
    process.env.INFER_API_KEY = saved.key     as string;
  });

  it("picks up INFER_URL", () => {
    process.env.INFER_URL = "http://custom:1234/v1";
    expect(loadConfig().url).toBe("http://custom:1234/v1");
  });

  it("picks up INFER_MODEL", () => {
    process.env.INFER_MODEL = "gpt-4o";
    expect(loadConfig().model).toBe("gpt-4o");
  });

  it("picks up INFER_API_KEY", () => {
    process.env.INFER_API_KEY = "sk-test-123";
    expect(loadConfig().api_key).toBe("sk-test-123");
  });

  it("env vars override config file defaults", () => {
    process.env.INFER_MODEL = "override-model";
    const cfg = loadConfig();
    expect(cfg.model).toBe("override-model");
  });
});

// --- Mocks ---

async function* makeStream(chunks: object[]) {
  for (const chunk of chunks) yield chunk;
}

const mockCreate = mock(() => Promise.resolve({
  choices: [{ message: { content: "Paris", tool_calls: null } }],
  usage: { completion_tokens: 5, prompt_tokens: 20 },
}));

mock.module("openai", () => ({
  default: class {
    chat = { completions: { create: mockCreate } };
  },
}));

// readline mock — controlled via replInputs queue
let replInputs: (string | null)[] = [];
mock.module("readline", () => ({
  createInterface: () => ({
    question: (_p: string, cb: (s: string | null) => void) => {
      const next = replInputs.shift() ?? null;
      setTimeout(() => cb(next), 0);
    },
    close: () => {},
    once: () => {},
    on: (_e: string, cb: () => void) => {},
  }),
}));

const BASE_OPTS = {
  url: "http://x", model: "m", apiKey: "x", system: "s",
  verbose: false, jsonMode: false as const, sandbox: false, allowNetwork: false,
};

// --- run() non-streaming ---
describe("run() non-streaming", () => {
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
        choices: [{ message: { content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"date"}' } }] } }],
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

  it("pushes assistant message once for multiple tool calls", async () => {
    const capturedMessages: any[] = [];
    mockCreate
      .mockImplementationOnce(async ({ messages }: any) => {
        return {
          choices: [{ message: { content: "", tool_calls: [
            { id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"echo a"}' } },
            { id: "c2", type: "function", function: { name: "bash", arguments: '{"command":"echo b"}' } },
          ] } }],
          usage: { completion_tokens: 10, prompt_tokens: 20 },
        };
      })
      .mockImplementationOnce(async ({ messages }: any) => {
        capturedMessages.push(...messages);
        return {
          choices: [{ message: { content: "done", tool_calls: null } }],
          usage: { completion_tokens: 3, prompt_tokens: 50 },
        };
      });
    const code = await run({ ...BASE_OPTS, prompt: "run two commands" });
    expect(code).toBe(0);
    // assistant message should appear exactly once
    const assistantMsgs = capturedMessages.filter((m: any) => m.role === "assistant");
    expect(assistantMsgs.length).toBe(1);
    // both tool results should be present
    const toolMsgs = capturedMessages.filter((m: any) => m.role === "tool");
    expect(toolMsgs.length).toBe(2);
    expect(toolMsgs[0].tool_call_id).toBe("c1");
    expect(toolMsgs[1].tool_call_id).toBe("c2");
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
      choices: [{ message: { content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"echo hi"}' } }] } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const code = await run({ ...BASE_OPTS, prompt: "loop", maxSteps: 3, sandbox: true });
    expect(code).toBe(1);
    expect(mockCreate).toHaveBeenCalledTimes(3);
  });

  it("defaults to 10 max steps", async () => {
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"echo hi"}' } }] } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const code = await run({ ...BASE_OPTS, prompt: "loop" });
    expect(code).toBe(1);
    expect(mockCreate).toHaveBeenCalledTimes(10);
  });

  it("does not stream by default", async () => {
    let capturedOpts: any;
    mockCreate.mockImplementationOnce(async (opts: any) => {
      capturedOpts = opts;
      return { choices: [{ message: { content: "ok", tool_calls: null } }], usage: { completion_tokens: 2, prompt_tokens: 10 } };
    });
    await run({ ...BASE_OPTS, prompt: "test" });
    expect(capturedOpts.stream).toBeFalsy();
  });

  it("passes stream:false when jsonMode is set even with stream:true", async () => {
    let capturedOpts: any;
    mockCreate.mockImplementationOnce(async (opts: any) => {
      capturedOpts = opts;
      return { choices: [{ message: { content: '{"x":1}', tool_calls: null } }], usage: { completion_tokens: 5, prompt_tokens: 10 } };
    });
    await run({ ...BASE_OPTS, prompt: "test", jsonMode: true, stream: true });
    expect(capturedOpts.stream).toBeFalsy();
  });
});

// --- run() streaming ---
describe("run() streaming", () => {
  let writeSpy: any;

  beforeEach(() => {
    mockCreate.mockClear();
    writeSpy = spyOn(process.stdout, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    writeSpy.mockRestore();
  });

  it("uses stream:true when stream option is set", async () => {
    let capturedOpts: any;
    mockCreate.mockImplementationOnce(async (opts: any) => {
      capturedOpts = opts;
      return makeStream([
        { choices: [{ delta: { content: "hello" } }] },
        { choices: [{ delta: { content: " world" } }] },
      ]);
    });
    const code = await run({ ...BASE_OPTS, prompt: "test", stream: true });
    expect(code).toBe(0);
    expect(capturedOpts.stream).toBe(true);
  });

  it("writes chunks to stdout as they arrive", async () => {
    mockCreate.mockImplementationOnce(async () =>
      makeStream([
        { choices: [{ delta: { content: "chunk1" } }] },
        { choices: [{ delta: { content: "chunk2" } }] },
      ])
    );
    await run({ ...BASE_OPTS, prompt: "test", stream: true });
    const written = writeSpy.mock.calls.map((c: any) => c[0]).join("");
    expect(written).toContain("chunk1");
    expect(written).toContain("chunk2");
  });

  it("accumulates streaming tool_call deltas and executes", async () => {
    mockCreate
      .mockImplementationOnce(async () =>
        makeStream([
          { choices: [{ delta: { content: "", tool_calls: [{ index: 0, id: "c", type: "function", function: { name: "ba", arguments: "" } }] } }] },
          { choices: [{ delta: { content: "", tool_calls: [{ index: 0, id: "", type: "function", function: { name: "sh", arguments: '{"command' } }] } }] },
          { choices: [{ delta: { content: "", tool_calls: [{ index: 0, id: "", type: "function", function: { name: "", arguments: '":"pwd"}' } }] } }] },
        ])
      )
      .mockImplementationOnce(async () =>
        makeStream([{ choices: [{ delta: { content: "/home" } }] }])
      );
    const code = await run({ ...BASE_OPTS, prompt: "where am i?", sandbox: true, stream: true });
    expect(code).toBe(0);
    expect(mockCreate).toHaveBeenCalledTimes(2);
    const secondCallMessages = (mockCreate.mock.calls[1] as any)[0].messages;
    const toolMsg = secondCallMessages.find((m: any) => m.role === "tool");
    expect(toolMsg).toBeDefined();
    expect(toolMsg.tool_call_id).toBe("c");
  });

  it("streaming tool call: pushes assistant message exactly once", async () => {
    const capturedMessages: any[] = [];
    mockCreate
      .mockImplementationOnce(async () =>
        makeStream([
          { choices: [{ delta: { tool_calls: [{ index: 0, id: "t1", type: "function", function: { name: "bash", arguments: '{"command":"ls"}' } }] } }] },
          { choices: [{ delta: { tool_calls: [{ index: 1, id: "t2", type: "function", function: { name: "bash", arguments: '{"command":"pwd"}' } }] } }] },
        ])
      )
      .mockImplementationOnce(async ({ messages }: any) => {
        capturedMessages.push(...messages);
        return makeStream([{ choices: [{ delta: { content: "done" } }] }]);
      });
    await run({ ...BASE_OPTS, prompt: "two tools", stream: true });
    const assistantMsgs = capturedMessages.filter((m: any) => m.role === "assistant");
    expect(assistantMsgs.length).toBe(1);
    const toolMsgs = capturedMessages.filter((m: any) => m.role === "tool");
    expect(toolMsgs.length).toBe(2);
  });

  it("disables streaming when jsonMode is set even with stream:true", async () => {
    let capturedOpts: any;
    mockCreate.mockImplementationOnce(async (opts: any) => {
      capturedOpts = opts;
      return { choices: [{ message: { content: '{"x":1}', tool_calls: null } }], usage: { completion_tokens: 5, prompt_tokens: 10 } };
    });
    await run({ ...BASE_OPTS, prompt: "test", jsonMode: true, stream: true });
    expect(capturedOpts.stream).toBeFalsy();
  });
});

// --- runRepl() ---
const REPL_OPTS = {
  url: "http://x", model: "m", apiKey: "x",
  system: "s", verbose: false, sandbox: false, allowNetwork: false,
};

describe("runRepl()", () => {
  let writeSpy: any;

  beforeEach(() => {
    mockCreate.mockClear();
    replInputs = [];
    writeSpy = spyOn(process.stdout, "write").mockImplementation(() => true);
    Object.defineProperty(process.stdout, "isTTY", { value: true, configurable: true });
  });

  afterEach(() => {
    writeSpy.mockRestore();
    Object.defineProperty(process.stdout, "isTTY", { value: undefined, configurable: true });
  });

  it("exits immediately on 'exit' input", async () => {
    replInputs = ["exit"];
    await runRepl(REPL_OPTS);
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("exits immediately on null (EOF)", async () => {
    replInputs = [null];
    await runRepl(REPL_OPTS);
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("skips empty input lines without calling LLM", async () => {
    replInputs = ["", "  ", "exit"];
    await runRepl(REPL_OPTS);
    expect(mockCreate).not.toHaveBeenCalled();
  });

  it("calls LLM with user input and prints response", async () => {
    replInputs = ["hello", "exit"];
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "hi there", tool_calls: null } }],
      usage: { completion_tokens: 3, prompt_tokens: 10 },
    });
    await runRepl(REPL_OPTS);
    expect(mockCreate).toHaveBeenCalledTimes(1);
    const written = writeSpy.mock.calls.map((c: any) => c[0]).join("");
    expect(written).toContain("hi there");
  });

  it("handles tool call then answer in repl turn", async () => {
    replInputs = ["what dir?", "exit"];
    mockCreate
      .mockResolvedValueOnce({
        choices: [{ message: { content: "", tool_calls: [{ id: "r1", type: "function", function: { name: "bash", arguments: '{"command":"pwd"}' } }] } }],
        usage: { completion_tokens: 5, prompt_tokens: 20 },
      })
      .mockResolvedValueOnce({
        choices: [{ message: { content: "/home", tool_calls: null } }],
        usage: { completion_tokens: 3, prompt_tokens: 30 },
      });
    await runRepl(REPL_OPTS);
    expect(mockCreate).toHaveBeenCalledTimes(2);
    const secondMessages = (mockCreate.mock.calls[1] as any)[0].messages;
    expect(secondMessages.some((m: any) => m.role === "tool")).toBe(true);
  });

  it("maintains conversation history across turns", async () => {
    replInputs = ["turn one", "turn two", "exit"];
    mockCreate
      .mockResolvedValueOnce({ choices: [{ message: { content: "reply one", tool_calls: null } }], usage: { completion_tokens: 3, prompt_tokens: 10 } })
      .mockImplementationOnce(async ({ messages }: any) => {
        const userMsgs = messages.filter((m: any) => m.role === "user");
        expect(userMsgs.length).toBe(2);
        return { choices: [{ message: { content: "reply two", tool_calls: null } }], usage: { completion_tokens: 3, prompt_tokens: 20 } };
      });
    await runRepl(REPL_OPTS);
    expect(mockCreate).toHaveBeenCalledTimes(2);
  });

  it("does not stream by default", async () => {
    replInputs = ["test", "exit"];
    let capturedOpts: any;
    mockCreate.mockImplementationOnce(async (opts: any) => {
      capturedOpts = opts;
      return { choices: [{ message: { content: "ok", tool_calls: null } }], usage: { completion_tokens: 2, prompt_tokens: 10 } };
    });
    await runRepl(REPL_OPTS);
    expect(capturedOpts.stream).toBeFalsy();
  });

  it("uses stream:true when stream option is set", async () => {
    replInputs = ["test", "exit"];
    let capturedOpts: any;
    mockCreate.mockImplementationOnce(async (opts: any) => {
      capturedOpts = opts;
      return makeStream([{ choices: [{ delta: { content: "ok" } }] }]);
    });
    await runRepl({ ...REPL_OPTS, stream: true });
    expect(capturedOpts.stream).toBe(true);
  });
});
