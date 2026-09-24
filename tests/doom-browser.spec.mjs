import {test,expect} from "@playwright/test";

test.setTimeout(300000);

test("real DOOM boots, primes MobileBERT, then keeps the control tick neural-fast",async({page})=>{
  const consoleErrors=[];page.on("pageerror",error=>consoleErrors.push("pageerror: "+String(error)));page.on("console",message=>{if(message.type()==="error")consoleErrors.push("console: "+message.text())});
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#runtimeStatus")?.textContent;return text==="READY"||text==="BOOT FAILED"},null,{timeout:90000});
  const runtimeStatus=(await page.locator("#runtimeStatus").textContent())||"";
  if(runtimeStatus!=="READY")throw new Error("DOOM boot failed. "+(await page.locator("#bootStatus").textContent())+" "+(await page.locator("#eventLog").textContent()));
  await expect(page.locator("#stateTable .state-row")).toHaveCount(14);await expect(page.locator("#backboneName")).toContainText("SchemaHashAttentionSetNet");

  await page.locator("#stepBtn").click();await expect(page.locator("#steps")).toHaveText("1",{timeout:5000});await expect(page.locator("#eventLog")).toContainText("s0001");await expect(page.locator("#attentionList .attention-row").first()).toBeVisible();

  await page.locator("#modelSelect").selectOption("mobilebert");await page.locator("#loadModelBtn").click();
  await page.waitForFunction(()=>{
    const text=document.querySelector("#modelStatus")?.textContent||"";
    return text.includes("teacher ready")||text.includes("load/bootstrap failed");
  },null,{timeout:220000});
  const modelStatus=(await page.locator("#modelStatus").textContent())||"";
  if(modelStatus.includes("failed"))throw new Error("MobileBERT activation failed. status="+modelStatus+" browser="+consoleErrors.join(" | "));

  await expect(page.locator("#backboneName")).toContainText("MobileBERT-MNLI");await expect(page.locator("#steps")).toHaveText("0");
  const teacherBefore=Number(await page.locator("#teacherCalls").textContent());expect(teacherBefore).toBeGreaterThanOrEqual(1);
  const start=Date.now();await page.locator("#stepBtn").click();await expect(page.locator("#steps")).toHaveText("1",{timeout:5000});
  const wallMs=Date.now()-start;expect(wallMs).toBeLessThan(5000);
  await expect(page.locator("#chosenAction")).not.toHaveText("—");await expect(page.locator("#eventLog")).toContainText("s0001");

  if(consoleErrors.length)throw new Error("Browser errors after successful learned-model step: "+consoleErrors.join(" | "));
});
