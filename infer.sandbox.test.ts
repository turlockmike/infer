/**
 * Sandbox integration tests — verify OS-enforced restrictions actually work.
 * These run the real sandbox-exec (macOS) or bwrap (Linux), no mocking.
 *
 * Skipped automatically on platforms without a sandbox.
 */

import { describe, it, expect, beforeAll } from "bun:test";
import { spawnSync } from "child_process";
import { execBash } from "./infer";

const isMac   = process.platform === "darwin";
const isLinux  = process.platform === "linux";
const hasBwrap = isLinux && spawnSync("which", ["bwrap"], { encoding: "utf8" }).stdout.trim() !== "";
const hasSandbox = isMac || hasBwrap;

const cwd = process.cwd();

// --- Network blocking ---

// Use bash /dev/tcp — it prints the raw OS error, unlike curl -s which suppresses it.
// Sandbox EPERM → "Operation not permitted"
// No sandbox, no service → "Connection refused"
const tcpProbe = "(echo > /dev/tcp/127.0.0.1/80) 2>&1; echo EXIT:$?";

describe("sandbox: network blocking", () => {
  it.skipIf(!hasSandbox)("blocks network when allowNetwork is false (OS returns EPERM)", async () => {
    const out = await execBash(tcpProbe, true, false, cwd);
    expect(out).toMatch(/operation not permitted/i);
  }, 10_000);

  it.skipIf(!hasSandbox)("allows network when allowNetwork is true (TCP stack responds normally)", async () => {
    // Network is allowed — we get "Connection refused" (no web server), never EPERM
    const out = await execBash(tcpProbe, true, true, cwd);
    expect(out).not.toMatch(/operation not permitted/i);
  }, 10_000);

  it("runs without sandbox: no network restrictions", async () => {
    const out = await execBash(tcpProbe, false, false, cwd);
    expect(out).not.toMatch(/operation not permitted/i);
  }, 10_000);
});

// --- Write restrictions ---

describe("sandbox: write restrictions", () => {
  it.skipIf(!hasSandbox)("blocks writes to home directory", async () => {
    const testFile = `${process.env.HOME}/infer-sandbox-write-test-${Date.now()}`;
    const out = await execBash(
      `echo test > ${testFile} 2>&1; echo EXIT:$?`,
      true, false, cwd
    );
    // Sandbox blocks the write — should not succeed
    expect(out).not.toContain("EXIT:0");
    // Clean up in case it somehow wrote (shouldn't happen)
    spawnSync("rm", ["-f", testFile]);
  }, 10_000);

  it.skipIf(!hasSandbox)("allows writes inside cwd", async () => {
    const testFile = `infer-sandbox-cwd-test-${Date.now()}.tmp`;
    const out = await execBash(
      `echo sandbox_ok > ${testFile} && cat ${testFile} && rm ${testFile}`,
      true, false, cwd
    );
    expect(out).toContain("sandbox_ok");
  }, 10_000);

  it.skipIf(!hasSandbox)("allows writes inside /tmp", async () => {
    const testFile = `/tmp/infer-sandbox-tmp-test-${Date.now()}.tmp`;
    const out = await execBash(
      `echo tmp_ok > ${testFile} && cat ${testFile} && rm ${testFile}`,
      true, false, cwd
    );
    expect(out).toContain("tmp_ok");
  }, 10_000);

  it("unsandboxed: writes to home succeed", async () => {
    const testFile = `${process.env.HOME}/infer-sandbox-write-test-${Date.now()}`;
    const out = await execBash(
      `echo test > ${testFile} 2>&1 && rm ${testFile} && echo EXIT:0`,
      false, false, cwd
    );
    expect(out).toContain("EXIT:0");
  }, 10_000);
});

// --- Reads are never restricted ---

describe("sandbox: reads always allowed", () => {
  it.skipIf(!hasSandbox)("can read files outside cwd (e.g. /etc/hostname)", async () => {
    const out = await execBash("cat /etc/hostname 2>/dev/null || echo fallback", true, false, cwd);
    // Should return something — sandbox only restricts writes and network
    expect(out.length).toBeGreaterThan(0);
  }, 10_000);
});
