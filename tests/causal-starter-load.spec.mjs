import {test,expect} from "@playwright/test";

test.setTimeout(180000);

test("published causal starter auto-loads and its validated replay can run",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html?starter=auto",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toContainText("CHECKPOINT READY",{timeout:120000});
  await expect(page.locator("#policyChip")).toContainText("ready to play");
  await page.locator("#inspectTabBtn").click();
  await expect(page.locator("#checkpointStatus")).toContainText("causal policy");
  await expect(page.locator("#checkpointStatus")).toContainText("32 kills");
  await expect(page.locator("#publishedReplayStatus")).toContainText("32 kills");
  await expect(page.locator("#publishedReplaySelect option")).toHaveCount(16);
  await page.selectOption("#publishedReplaySelect","0");
  await page.locator("#publishedReplayBtn").click();
  await expect(page.locator("#publishedReplayStatus")).toContainText("Replay complete",{timeout:30000});
  await expect(page.locator("#publishedReplayStatus")).toContainText("2 kills");
  await expect(page.locator("#publishedReplayStatus")).toContainText("60 damage");
});
