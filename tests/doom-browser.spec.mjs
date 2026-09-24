import {test,expect} from "@playwright/test";

test.setTimeout(120000);

test("real DOOM runtime boots and accepts one primitive policy step",async({page})=>{
  const errors=[];
  page.on("pageerror",error=>errors.push(String(error)));
  page.on("console",message=>{if(message.type()==="error")errors.push(message.text())});

  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();
  await expect(page.locator("#runtimeStatus")).toHaveText("READY",{timeout:90000});
  await expect(page.locator("#stateTable .state-row")).toHaveCount(14);
  await expect(page.locator("#backboneName")).toContainText("HashSemanticAdapter");

  await page.locator("#stepBtn").click();
  await expect(page.locator("#steps")).toHaveText("1",{timeout:15000});
  await expect(page.locator("#chosenAction")).not.toHaveText("—");
  await expect(page.locator("#eventLog")).toContainText("s0001");

  if(errors.length)throw new Error("Browser errors: "+errors.join(" | "));
});
