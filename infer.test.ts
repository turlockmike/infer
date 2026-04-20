import { describe, it, expect, mock, beforeEach, afterEach, spyOn } from "bun:test";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { validateShape, run, runRepl, loadConfig, loadSession, appendToSession, runConfigCmd, _setConfigDir, encodeImagePart } from "./infer";

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
    on: () => {},
    removeListener: () => {},
  }),
}));

const BASE_OPTS = {
  url: "http://x", model: "m", apiKey: "x", system: "s",
  verbose: false, jsonMode: false as const, sandbox: false, allowNetwork: false,
};

// --- thinking block stripping ---
describe("thinking block stripping", () => {
  beforeEach(() => mockCreate.mockClear());

  it("strips <think>...</think> blocks from response (qwen3/DeepSeek style)", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "<think>\nsome chain of thought\n</think>\nParis", tool_calls: null } }],
      usage: { completion_tokens: 10, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "capital of France?" });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toBe("Paris");
    expect(output).not.toContain("<think>");
  });

  it("strips <thinking>...</thinking> blocks (GLM/nemotron style)", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "<thinking>\nlong reasoning here\n</thinking>\n42", tool_calls: null } }],
      usage: { completion_tokens: 10, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "7 * 6?" });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toBe("42");
    expect(output).not.toContain("<thinking>");
  });

  it("passes through responses with no thinking blocks unchanged", async () => {
    let output = "";
    const origLog = console.log;
    console.log = (s: string) => { output = s; };
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "plain answer", tool_calls: null } }],
      usage: { completion_tokens: 2, prompt_tokens: 10 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "test" });
    console.log = origLog;
    expect(code).toBe(0);
    expect(output).toBe("plain answer");
  });
});

// --- run() non-streaming ---
describe("run() non-streaming", () => {
  beforeEach(() => mockCreate.mockClear());

  it("returns 0 on simple response", async () => {
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "Paris", tool_calls: null } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "capital of France?" });
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
    const { code } = await run({ ...BASE_OPTS, prompt: "what day is it?", sandbox: true });
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
    const { code } = await run({ ...BASE_OPTS, prompt: "run two commands" });
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
    const { code } = await run({ ...BASE_OPTS, prompt: "get data", jsonMode: true });
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
    const { code } = await run({ ...BASE_OPTS, prompt: "get user", jsonMode: '{"name":""}' });
    expect(code).toBe(0);
    expect(mockCreate).toHaveBeenCalledTimes(2);
  });

  it("returns 1 when max steps reached", async () => {
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"echo hi"}' } }] } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "loop", maxSteps: 3, sandbox: true });
    expect(code).toBe(1);
    expect(mockCreate).toHaveBeenCalledTimes(3);
  });

  it("defaults to 10 max steps", async () => {
    mockCreate.mockResolvedValue({
      choices: [{ message: { content: "", tool_calls: [{ id: "c1", type: "function", function: { name: "bash", arguments: '{"command":"echo hi"}' } }] } }],
      usage: { completion_tokens: 5, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "loop" });
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
    const { code } = await run({ ...BASE_OPTS, prompt: "test", stream: true });
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
    const { code } = await run({ ...BASE_OPTS, prompt: "where am i?", sandbox: true, stream: true });
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

// --- Session (JSONL) ---
describe("session JSONL", () => {
  let tmpDir: string;
  let sessionFile: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), "infer-test-"));
    sessionFile = join(tmpDir, "session.jsonl");
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true });
  });

  it("loadSession returns empty array for missing file", () => {
    expect(loadSession(sessionFile)).toEqual([]);
  });

  it("appendToSession + loadSession round-trips messages", () => {
    const msgs = [
      { role: "user" as const, content: "hello" },
      { role: "assistant" as const, content: "hi" },
    ];
    appendToSession(sessionFile, msgs);
    expect(loadSession(sessionFile)).toEqual(msgs);
  });

  it("loadSession filters out system messages", () => {
    const msgs = [
      { role: "system" as const, content: "you are helpful" },
      { role: "user" as const, content: "hello" },
      { role: "assistant" as const, content: "hi" },
    ];
    appendToSession(sessionFile, msgs);
    const loaded = loadSession(sessionFile);
    expect(loaded).toHaveLength(2);
    expect(loaded.every(m => m.role !== "system")).toBe(true);
  });

  it("appends across multiple calls (conversation grows)", () => {
    appendToSession(sessionFile, [{ role: "user" as const, content: "turn1" }]);
    appendToSession(sessionFile, [{ role: "assistant" as const, content: "reply1" }]);
    appendToSession(sessionFile, [{ role: "user" as const, content: "turn2" }]);
    const loaded = loadSession(sessionFile);
    expect(loaded).toHaveLength(3);
    expect((loaded[2] as any).content).toBe("turn2");
  });

  it("run() returns messages including new turn", async () => {
    mockCreate.mockClear();
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "answer", tool_calls: null } }],
      usage: { completion_tokens: 2, prompt_tokens: 10 },
    });
    const { code, messages } = await run({ ...BASE_OPTS, prompt: "question" });
    expect(code).toBe(0);
    expect(messages.some(m => m.role === "user" && (m as any).content === "question")).toBe(true);
    expect(messages.some(m => m.role === "assistant" && (m as any).content === "answer")).toBe(true);
  });

  it("run() with initialMessages continues from prior session", async () => {
    mockCreate.mockClear();
    const prior = [
      { role: "user" as const, content: "first question" },
      { role: "assistant" as const, content: "first answer" },
    ];
    let capturedMessages: any[] = [];
    mockCreate.mockImplementationOnce(async ({ messages }: any) => {
      capturedMessages = [...messages]; // snapshot before run() mutates the array
      return { choices: [{ message: { content: "second answer", tool_calls: null } }], usage: { completion_tokens: 3, prompt_tokens: 20 } };
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "second question", initialMessages: prior });
    expect(code).toBe(0);
    // System + 2 prior + new user = 4 messages sent to LLM
    expect(capturedMessages).toHaveLength(4);
    expect(capturedMessages[1].content).toBe("first question");
    expect(capturedMessages[3].content).toBe("second question");
  });
});

// --- image input ---
describe("encodeImagePart", () => {
  let tmp: string;
  beforeEach(() => { tmp = mkdtempSync(join(tmpdir(), "infer-img-")); });
  afterEach(() => rmSync(tmp, { recursive: true, force: true }));

  it("encodes a PNG file as a data URL", () => {
    const path = join(tmp, "x.png");
    writeFileSync(path, Buffer.from([0x89, 0x50, 0x4e, 0x47])); // PNG magic
    const part = encodeImagePart(path);
    expect(part.type).toBe("image_url");
    expect(part.image_url.url.startsWith("data:image/png;base64,")).toBe(true);
    // base64("\x89PNG") = "iVBORw==" — first 4 bytes of a PNG signature
    expect(part.image_url.url).toContain("iVBORw==");
  });

  it("detects jpeg extension as image/jpeg", () => {
    const path = join(tmp, "x.jpeg");
    writeFileSync(path, Buffer.from([0xff, 0xd8, 0xff])); // JPEG magic
    expect(encodeImagePart(path).image_url.url.startsWith("data:image/jpeg;base64,")).toBe(true);
  });

  it("detects .jpg as image/jpeg", () => {
    const path = join(tmp, "x.jpg");
    writeFileSync(path, Buffer.from([0xff]));
    expect(encodeImagePart(path).image_url.url.startsWith("data:image/jpeg;base64,")).toBe(true);
  });

  it("detects webp", () => {
    const path = join(tmp, "x.webp");
    writeFileSync(path, Buffer.from([0]));
    expect(encodeImagePart(path).image_url.url.startsWith("data:image/webp;base64,")).toBe(true);
  });

  it("throws on missing file", () => {
    expect(() => encodeImagePart(join(tmp, "nope.png"))).toThrow(/not found/);
  });

  it("throws on unsupported extension", () => {
    const path = join(tmp, "x.bmp");
    writeFileSync(path, Buffer.from([0]));
    expect(() => encodeImagePart(path)).toThrow(/unsupported/);
  });
});

describe("run() with images", () => {
  let tmp: string;
  beforeEach(() => { mockCreate.mockClear(); tmp = mkdtempSync(join(tmpdir(), "infer-img-run-")); });
  afterEach(() => rmSync(tmp, { recursive: true, force: true }));

  it("builds multimodal user content when images are attached", async () => {
    const imgPath = join(tmp, "frame.png");
    writeFileSync(imgPath, Buffer.from([0x89, 0x50, 0x4e, 0x47]));

    let capturedMessages: any[] = [];
    mockCreate.mockImplementationOnce(async ({ messages }: any) => {
      capturedMessages = messages;
      return { choices: [{ message: { content: "a PNG", tool_calls: null } }], usage: { completion_tokens: 3, prompt_tokens: 20 } };
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "describe this", images: [imgPath] });
    expect(code).toBe(0);
    const userMsg = capturedMessages.find((m: any) => m.role === "user");
    expect(Array.isArray(userMsg.content)).toBe(true);
    expect(userMsg.content[0]).toEqual({ type: "text", text: "describe this" });
    expect(userMsg.content[1].type).toBe("image_url");
    expect(userMsg.content[1].image_url.url.startsWith("data:image/png;base64,")).toBe(true);
  });

  it("supports multiple images in one call", async () => {
    const a = join(tmp, "a.png"); writeFileSync(a, Buffer.from([0x89]));
    const b = join(tmp, "b.jpg"); writeFileSync(b, Buffer.from([0xff, 0xd8]));

    let capturedMessages: any[] = [];
    mockCreate.mockImplementationOnce(async ({ messages }: any) => {
      capturedMessages = messages;
      return { choices: [{ message: { content: "ok", tool_calls: null } }], usage: { completion_tokens: 1, prompt_tokens: 20 } };
    });
    await run({ ...BASE_OPTS, prompt: "compare", images: [a, b] });
    const userMsg = capturedMessages.find((m: any) => m.role === "user");
    expect(userMsg.content).toHaveLength(3); // text + 2 images
    expect(userMsg.content[1].image_url.url).toContain("image/png");
    expect(userMsg.content[2].image_url.url).toContain("image/jpeg");
  });

  it("uses plain string content when no images are attached", async () => {
    let capturedMessages: any[] = [];
    mockCreate.mockImplementationOnce(async ({ messages }: any) => {
      capturedMessages = messages;
      return { choices: [{ message: { content: "ok", tool_calls: null } }], usage: { completion_tokens: 1, prompt_tokens: 20 } };
    });
    await run({ ...BASE_OPTS, prompt: "hi" });
    const userMsg = capturedMessages.find((m: any) => m.role === "user");
    expect(typeof userMsg.content).toBe("string");
    expect(userMsg.content).toBe("hi");
  });

  it("omits the text part when prompt is empty but images are attached", async () => {
    const img = join(tmp, "img.png"); writeFileSync(img, Buffer.from([0x89]));
    let capturedMessages: any[] = [];
    mockCreate.mockImplementationOnce(async ({ messages }: any) => {
      capturedMessages = messages;
      return { choices: [{ message: { content: "ok", tool_calls: null } }], usage: { completion_tokens: 1, prompt_tokens: 20 } };
    });
    await run({ ...BASE_OPTS, prompt: "", images: [img] });
    const userMsg = capturedMessages.find((m: any) => m.role === "user");
    expect(Array.isArray(userMsg.content)).toBe(true);
    expect(userMsg.content).toHaveLength(1);
    expect(userMsg.content[0].type).toBe("image_url");
  });

  it("image-only invocation still completes successfully (no prompt required)", async () => {
    const img = join(tmp, "img.png"); writeFileSync(img, Buffer.from([0x89]));
    mockCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "looks like a PNG", tool_calls: null } }],
      usage: { completion_tokens: 3, prompt_tokens: 20 },
    });
    const { code } = await run({ ...BASE_OPTS, prompt: "", images: [img] });
    expect(code).toBe(0);
  });
});

// --- Helpers for profile tests ---
// Isolate each test by overriding the config dir via the exported _configDirOverride.
// (Bun's homedir() ignores process.env.HOME — uses getpwuid — so we use a module-level override instead.)
function withTmpConfig<T>(fn: (configDir: string) => T): T {
  const configDir = mkdtempSync(join(tmpdir(), "infer-cfg-"));
  _setConfigDir(configDir);
  try {
    return fn(configDir);
  } finally {
    _setConfigDir(undefined);
    rmSync(configDir, { recursive: true });
  }
}

function writeConfig(configDir: string, data: object) {
  mkdirSync(configDir, { recursive: true });
  writeFileSync(join(configDir, "config.json"), JSON.stringify(data, null, 2));
}

function writeProfile(configDir: string, name: string, data: object) {
  mkdirSync(join(configDir, "profiles"), { recursive: true });
  writeFileSync(join(configDir, "profiles", `${name}.json`), JSON.stringify(data, null, 2));
}

function captureOutput(fn: () => void): { stdout: string; stderr: string } {
  const logs: string[] = [];
  const errs: string[] = [];
  const origLog   = console.log;
  const origError = console.error;
  console.log   = (...args: any[]) => logs.push(args.join(" "));
  console.error = (...args: any[]) => errs.push(args.join(" "));
  fn();
  console.log   = origLog;
  console.error = origError;
  return { stdout: logs.join("\n"), stderr: errs.join("\n") };
}

// --- loadConfig — profile precedence ---
describe("loadConfig — profile precedence", () => {
  it("returns defaults when no config or profile", () => {
    withTmpConfig(() => {
      // clear env vars to avoid interference
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL;
      delete process.env.INFER_MODEL;
      delete process.env.INFER_API_KEY;
      const cfg = loadConfig();
      process.env.INFER_URL     = saved.url;
      process.env.INFER_MODEL   = saved.model;
      process.env.INFER_API_KEY = saved.key;
      expect(cfg.url).toBe("http://localhost:11434/v1");
      expect(cfg.model).toBe("gemma4:latest");
      expect(cfg.api_key).toBe("ollama");
      expect(cfg.profile).toBeUndefined();
    });
  });

  it("profile-only: profile values used when no global override", () => {
    withTmpConfig((configDir) => {
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL; delete process.env.INFER_MODEL; delete process.env.INFER_API_KEY;
      writeConfig(configDir, { profile: "tower" });
      writeProfile(configDir, "tower", { url: "http://192.168.4.30:11434/v1", model: "llama3:latest" });
      const cfg = loadConfig();
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      expect(cfg.url).toBe("http://192.168.4.30:11434/v1");
      expect(cfg.model).toBe("llama3:latest");
      expect(cfg.profile).toBe("tower");
    });
  });

  it("global config url/model override profile values", () => {
    withTmpConfig((configDir) => {
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL; delete process.env.INFER_MODEL; delete process.env.INFER_API_KEY;
      writeConfig(configDir, { profile: "tower", url: "http://override:9000/v1" });
      writeProfile(configDir, "tower", { url: "http://192.168.4.30:11434/v1", model: "llama3:latest" });
      const cfg = loadConfig();
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      // global url wins over profile url
      expect(cfg.url).toBe("http://override:9000/v1");
      // profile model still applies (no global model override)
      expect(cfg.model).toBe("llama3:latest");
    });
  });

  it("env var overrides both profile and global config", () => {
    withTmpConfig((configDir) => {
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      writeConfig(configDir, { profile: "tower", url: "http://override:9000/v1" });
      writeProfile(configDir, "tower", { url: "http://192.168.4.30:11434/v1", model: "llama3:latest" });
      process.env.INFER_URL = "http://env-wins:1234/v1";
      const cfg = loadConfig();
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      expect(cfg.url).toBe("http://env-wins:1234/v1");
    });
  });

  it("missing profile file is silently skipped", () => {
    withTmpConfig((configDir) => {
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL; delete process.env.INFER_MODEL; delete process.env.INFER_API_KEY;
      writeConfig(configDir, { profile: "ghost" }); // no profile file
      const cfg = loadConfig();
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      // falls back to defaults
      expect(cfg.url).toBe("http://localhost:11434/v1");
      expect(cfg.profile).toBe("ghost");
    });
  });
});

// --- runConfigCmd — use ---
describe("runConfigCmd use", () => {
  it("activates an existing profile", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, {});
      writeProfile(configDir, "tower", { url: "http://tower/v1" });
      const out = captureOutput(() => runConfigCmd(["use", "tower"]));
      expect(out.stdout).toContain("tower");
      // config should now have profile=tower
      const cfg = loadConfig();
      expect(cfg.profile).toBe("tower");
    });
  });

  it("errors if profile does not exist", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, {});
      let exited = false;
      const origExit = process.exit;
      (process as any).exit = () => { exited = true; throw new Error("exit"); };
      try {
        captureOutput(() => runConfigCmd(["use", "ghost"]));
      } catch {}
      (process as any).exit = origExit;
      expect(exited).toBe(true);
    });
  });

  it("clears active profile with --none", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, { profile: "tower" });
      writeProfile(configDir, "tower", { url: "http://tower/v1" });
      captureOutput(() => runConfigCmd(["use", "--none"]));
      const cfg = loadConfig();
      expect(cfg.profile).toBeUndefined();
    });
  });
});

// --- runConfigCmd — profile list ---
describe("runConfigCmd profile list", () => {
  it("lists no profiles when dir doesn't exist", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, {});
      const out = captureOutput(() => runConfigCmd(["profile", "list"]));
      expect(out.stdout).toBe("");
    });
  });

  it("lists profiles, marking the active one", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, { profile: "tower" });
      writeProfile(configDir, "tower", { url: "http://tower/v1" });
      writeProfile(configDir, "local", { url: "http://localhost/v1" });
      const out = captureOutput(() => runConfigCmd(["profile", "list"]));
      expect(out.stdout).toContain("tower (active)");
      expect(out.stdout).toContain("local");
      expect(out.stdout).not.toContain("local (active)");
    });
  });
});

// --- runConfigCmd — profile show ---
describe("runConfigCmd profile show", () => {
  it("prints key=value lines for the profile", () => {
    withTmpConfig((configDir) => {
      writeProfile(configDir, "tower", { url: "http://tower/v1", model: "gemma4:latest" });
      const out = captureOutput(() => runConfigCmd(["profile", "show", "tower"]));
      expect(out.stdout).toContain("url=http://tower/v1");
      expect(out.stdout).toContain("model=gemma4:latest");
    });
  });

  it("errors if profile not found", () => {
    withTmpConfig(() => {
      let exited = false;
      const origExit = process.exit;
      (process as any).exit = () => { exited = true; throw new Error("exit"); };
      try { captureOutput(() => runConfigCmd(["profile", "show", "ghost"])); } catch {}
      (process as any).exit = origExit;
      expect(exited).toBe(true);
    });
  });
});

// --- runConfigCmd — profile add ---
describe("runConfigCmd profile add", () => {
  it("creates a profile file with given fields", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, {});
      captureOutput(() => runConfigCmd(["profile", "add", "tower", "--url", "http://tower/v1", "--model", "llama3"]));
      const cfg = loadConfig();
      // activate and check
      captureOutput(() => runConfigCmd(["use", "tower"]));
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL; delete process.env.INFER_MODEL; delete process.env.INFER_API_KEY;
      const resolved = loadConfig();
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      expect(resolved.url).toBe("http://tower/v1");
      expect(resolved.model).toBe("llama3");
    });
  });

  it("errors if no fields provided", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, {});
      let exited = false;
      const origExit = process.exit;
      (process as any).exit = () => { exited = true; throw new Error("exit"); };
      try { captureOutput(() => runConfigCmd(["profile", "add", "empty"])); } catch {}
      (process as any).exit = origExit;
      expect(exited).toBe(true);
    });
  });

  it("errors if profile already exists", () => {
    withTmpConfig((configDir) => {
      writeProfile(configDir, "tower", { url: "http://tower/v1" });
      let exited = false;
      const origExit = process.exit;
      (process as any).exit = () => { exited = true; throw new Error("exit"); };
      try { captureOutput(() => runConfigCmd(["profile", "add", "tower", "--url", "http://other/v1"])); } catch {}
      (process as any).exit = origExit;
      expect(exited).toBe(true);
    });
  });
});

// --- runConfigCmd — profile edit ---
describe("runConfigCmd profile edit", () => {
  it("merges new fields into an existing profile", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, { profile: "tower" });
      writeProfile(configDir, "tower", { url: "http://tower/v1", model: "old-model" });
      captureOutput(() => runConfigCmd(["profile", "edit", "tower", "--model", "new-model"]));
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL; delete process.env.INFER_MODEL; delete process.env.INFER_API_KEY;
      const resolved = loadConfig();
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      expect(resolved.url).toBe("http://tower/v1");   // unchanged
      expect(resolved.model).toBe("new-model");        // updated
    });
  });

  it("errors if profile not found", () => {
    withTmpConfig(() => {
      let exited = false;
      const origExit = process.exit;
      (process as any).exit = () => { exited = true; throw new Error("exit"); };
      try { captureOutput(() => runConfigCmd(["profile", "edit", "ghost", "--url", "x"])); } catch {}
      (process as any).exit = origExit;
      expect(exited).toBe(true);
    });
  });
});

// --- runConfigCmd — profile rm ---
describe("runConfigCmd profile rm", () => {
  it("removes a profile file", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, {});
      writeProfile(configDir, "tower", { url: "http://tower/v1" });
      captureOutput(() => runConfigCmd(["profile", "rm", "tower"]));
      const out = captureOutput(() => runConfigCmd(["profile", "list"]));
      expect(out.stdout).not.toContain("tower");
    });
  });

  it("clears active profile when removing the active one", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, { profile: "tower" });
      writeProfile(configDir, "tower", { url: "http://tower/v1" });
      captureOutput(() => runConfigCmd(["profile", "rm", "tower"]));
      const cfg = loadConfig();
      expect(cfg.profile).toBeUndefined();
    });
  });

  it("errors if profile not found", () => {
    withTmpConfig(() => {
      let exited = false;
      const origExit = process.exit;
      (process as any).exit = () => { exited = true; throw new Error("exit"); };
      try { captureOutput(() => runConfigCmd(["profile", "rm", "ghost"])); } catch {}
      (process as any).exit = origExit;
      expect(exited).toBe(true);
    });
  });
});

// --- runConfigCmd — config show with profile ---
describe("runConfigCmd show with profile", () => {
  it("shows profile=(none) when no profile active", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, { url: "http://x/v1", model: "m", api_key: "k" });
      const out = captureOutput(() => runConfigCmd(["show"]));
      expect(out.stdout).toContain("profile=(none)");
    });
  });

  it("shows active profile name in config show", () => {
    withTmpConfig((configDir) => {
      writeConfig(configDir, { profile: "tower" });
      writeProfile(configDir, "tower", { url: "http://tower/v1", model: "llama3" });
      const saved = { url: process.env.INFER_URL, model: process.env.INFER_MODEL, key: process.env.INFER_API_KEY };
      delete process.env.INFER_URL; delete process.env.INFER_MODEL; delete process.env.INFER_API_KEY;
      const out = captureOutput(() => runConfigCmd(["show"]));
      process.env.INFER_URL = saved.url; process.env.INFER_MODEL = saved.model; process.env.INFER_API_KEY = saved.key;
      expect(out.stdout).toContain("profile=tower");
      expect(out.stdout).toContain("url=http://tower/v1");
      expect(out.stdout).toContain("model=llama3");
    });
  });
});
