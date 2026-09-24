import {test,expect} from "@playwright/test";

test.setTimeout(120000);

test("real DOOM runtime boots and accepts one primitive policy step",async({page})=>{
  const consoleErrors=[];
  page.on("pageerror",error=>consoleErrors.push("pageerror: "+String(error)));
  page.on("console",message=>{if(message.type()==="error")consoleErrors.push("console: "+message.text())});

  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});
  await page.locator("#bootBtn").click();

  await page.waitForFunction(()=>{
    const text=document.querySelector("#runtimeStatus")?.textContent;
    return text==="READY"||text==="BOOT FAILED";
  },null,{timeout:90000});

  const runtimeStatus=(await page.locator("#runtimeStatus").textContent())||"";
  if(runtimeStatus!=="READY"){
    const bootStatus=(await page.locator("#bootStatus").textContent())||"";
    const log=(await page.locator("#eventLog").textContent())||"";
    throw new Error("DOOM boot failed. bootStatus="+bootStatus+" log="+log+" browser="+consoleErrors.join(" | "));
  }

  await expect(page.locator("#stateTable .state-row")).toHaveCount(14);
  await expect(page.locator("#backboneName")).toContainText("HashSemanticAdapter");

  await page.locator("#stepBtn").click();
  await expect(page.locator("#steps")).toHaveText("1",{timeout:15000});
  await expect(page.locator("#chosenAction")).not.toHaveText("—");
  await expect(page.locator("#eventLog")).toContainText("s0001");

  if(consoleErrors.length)throw new Error("Browser errors after successful boot: "+consoleErrors.join(" | "));
});
