const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({
    headless: false,
    args: [
      '--lang=en-US',
      '--disable-features=TranslateUI,Translate',
      '--disable-translate',
      '--disable-infobars',
      '--no-default-browser-check',
      '--disable-component-update'
    ]
  });
  const context = await browser.newContext({
    viewport: { width: 1200, height: 800 },
    locale: 'en-US'
  });
  const page = await context.newPage();

  // 1. Open page — wait 8s for manual translate popup dismiss
  await page.goto('http://localhost:8080/');
  await page.waitForTimeout(8000);

  // 2. Select Local CLI -> claude
  await page.click('text=Local CLI');
  await page.waitForTimeout(1500);
  // CLI options are dynamically generated divs — click the one containing 'claude'
  await page.click('#cli-options div:has-text("claude")');
  await page.waitForTimeout(1500);

  // 3. Wait for input
  await page.waitForSelector('input[type="text"], textarea', { timeout: 30000 });
  await page.waitForTimeout(1000);

  // 4. Type prompt slowly
  const input = await page.$('input[type="text"], textarea');
  await input.click();
  await input.type('Simulate a congested 8-lane urban arterial during evening rush hour', { delay: 50 });
  await page.waitForTimeout(2000);

  // 5. Submit
  await page.keyboard.press('Enter');

  // 6. Wait for any action button to appear (Adjust, Correction, Error Correction, etc.)
  await page.waitForFunction(() => {
    const buttons = document.querySelectorAll('button');
    return Array.from(buttons).some(b =>
      b.textContent.includes('Adjust') ||
      b.textContent.includes('Correction') ||
      b.textContent.includes('Modify') ||
      b.textContent.includes('Error')
    );
  }, null, { timeout: 300000 });

  // Scroll to bottom
  await page.evaluate(() => {
    const msgs = document.getElementById('messages');
    if (msgs) msgs.scrollTop = msgs.scrollHeight;
  });
  await page.waitForTimeout(3000);

  // 7. Click the correction/error correction button (try multiple selectors)
  const corrBtn = await page.$('button:has-text("Error Correction")')
    || await page.$('button:has-text("Correction")')
    || await page.$('button:has-text("Modify")');
  if (corrBtn) {
    await corrBtn.click();
    await page.waitForTimeout(1500);
  }

  // If Modify was clicked, now click Error Correction submenu
  const errCorrBtn = await page.$('button:has-text("Error Correction")');
  if (errCorrBtn) {
    await errCorrBtn.click();
    await page.waitForTimeout(1500);
  }

  // 8. Type correction
  await page.waitForSelector('input[type="text"], textarea', { timeout: 10000 });
  const modInput = await page.$('input[type="text"], textarea');
  await modInput.click();
  await modInput.type('Speed is too high for rush hour congestion, should be under 30 km/h', { delay: 50 });
  await page.waitForTimeout(2000);

  // 9. Submit correction
  await page.keyboard.press('Enter');

  // 10. Wait for final result
  await page.waitForFunction(() => {
    const buttons = document.querySelectorAll('button');
    return Array.from(buttons).some(b =>
      b.textContent.includes('Adjust') ||
      b.textContent.includes('Correction') ||
      b.textContent.includes('Modify')
    );
  }, null, { timeout: 300000 });

  await page.evaluate(() => {
    const msgs = document.getElementById('messages');
    if (msgs) msgs.scrollTop = msgs.scrollHeight;
  });
  await page.waitForTimeout(5000);

  await browser.close();
})();
