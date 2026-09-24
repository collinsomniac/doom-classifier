import {test,expect} from "@playwright/test";

test.setTimeout(420000);

test("real DOOM verifies firing, prepares semantics, and runs the neural fast path",async({page})=>{
  const consoleErrors=[];page.on("pageerror",error=>consoleErrors.push("pageerror: "+String(error)));page.on("console",message=>{if(message.type()==="error")consoleErrors.push("console: "+message.text())});
  await page.goto("http://127.0.0.1:8000/doom.html",{waitUntil:"domcontentloaded"});await page.locator("#bootBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#runtimeStatus")?.textContent;return text==="ENGINE READY"||text==="BOOT FAILED"},null,{timeout:90000});
  if((await page.locator("#runtimeStatus").textContent())!=="ENGINE READY")throw new Error("DOOM boot failed. "+(await page.locator("#bootStatus").textContent()));
  await expect(page.locator("#stateTable .state-row")).toHaveCount(17);
  await expect(page.locator("#weaponState")).toContainText("pistol");
  await expect(page.locator("#manualActionSelect")).toHaveValue("fire");

  const fire=await page.evaluate(async()=>{
    const lab=window.__doomLab,env=lab.env;env.setActionMs(300);
    const before=env.observe(),ammoBefore=before.bullets+before.shells+before.rockets+before.cells;
    const step=await env.step("fire"),after=step.observation,ammoAfter=after.bullets+after.shells+after.rockets+after.cells;
    await env.reset();env.setActionMs(110);
    return{ammoBefore,ammoAfter,damage:step.info?.outcome?.damageDealt||0,mask:env.actionMasks.fire,combo:env.actionMasks.strafe_left_fire};
  });
  expect(fire.mask).toBe(64);expect(fire.combo).toBe(80);expect(fire.ammoAfter<fire.ammoBefore||fire.damage>0).toBeTruthy();

  await page.locator("#prepareBtn").click();
  await page.waitForFunction(()=>{const text=document.querySelector("#prepareStatus")?.textContent||"";return text.includes("READY TO PLAY")||text.includes("Preparation failed")},null,{timeout:300000});
  const prepare=(await page.locator("#prepareStatus").textContent())||"";
  if(prepare.includes("failed"))throw new Error("Preparation failed: "+prepare+" browser="+consoleErrors.join(" | "));
  await expect(page.locator("#schemaChip")).toContainText("schema compiled");await expect(page.locator("#teacherChip")).toContainText("teacher loaded");await expect(page.locator("#policyChip")).toContainText("ready to play");
  await expect(page.locator("#backboneName")).toContainText("params");

  await page.locator("#stepBtn").click();await expect(page.locator("#steps")).toHaveText("1",{timeout:8000});
  await expect(page.locator("#chosenAction")).not.toHaveText("—");await expect(page.locator("#eventLog")).toContainText("s0001");
  await expect(page.locator("#attentionList .attention-row").first()).toBeVisible();

  if(consoleErrors.length)throw new Error("Browser errors after successful prepared-model step: "+consoleErrors.join(" | "));
});
