import {test,expect} from "@playwright/test";

test.setTimeout(180000);

test("published causal starter auto-loads and its validated replay can run",async({page})=>{
  await page.goto("http://127.0.0.1:8000/doom.html?starter=auto",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toContainText("CHECKPOINT READY",{timeout:120000});
  await expect(page.locator("#policyChip")).toContainText("ready to play");
  await page.locator("#inspectTabBtn").click();

  const checkpoint=page.locator("#checkpointStatus"),replayStatus=page.locator("#publishedReplayStatus"),options=page.locator("#publishedReplaySelect option");
  await expect(checkpoint).toContainText("causal policy");
  await expect(replayStatus).toContainText("Published replay bundle");
  await expect(options).toHaveCount(16);

  const checkpointText=await checkpoint.textContent(),bundleText=await replayStatus.textContent();
  const checkpointKills=Number(checkpointText?.match(/([0-9]+) kills/)?.[1]||0);
  const bundleKills=Number(bundleText?.match(/([0-9]+) kills/)?.[1]||0);
  expect(checkpointKills).toBeGreaterThanOrEqual(20);
  expect(bundleKills).toBe(checkpointKills);

  await page.selectOption("#publishedReplaySelect","0");
  const optionText=await page.locator("#publishedReplaySelect option:checked").textContent();
  const expectedKills=Number(optionText?.match(/([0-9]+) kills/)?.[1]||0);
  const expectedDamage=Number(optionText?.match(/([0-9]+) dmg/)?.[1]||0);

  await page.locator("#publishedReplayBtn").click();
  await expect(replayStatus).toContainText("Replay complete",{timeout:30000});
  const completed=await replayStatus.textContent();
  const actual=completed?.match(/run 1 · ([0-9]+) kills · ([0-9]+) damage · expected ([0-9]+) \/ ([0-9]+)/);
  expect(actual).not.toBeNull();
  expect(Number(actual[1])).toBe(expectedKills);
  expect(Number(actual[2])).toBe(expectedDamage);
  expect(Number(actual[3])).toBe(expectedKills);
  expect(Number(actual[4])).toBe(expectedDamage);
});
