const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  
  await page.goto('http://localhost:8080/');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'docs/ui-main.png', fullPage: false });
  console.log('ui-main.png done');
  
  await page.goto('http://localhost:8080/admin');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'docs/admin.png', fullPage: false });
  console.log('admin.png done');

  await page.goto('http://localhost:8080/about');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'docs/about.png', fullPage: false });
  console.log('about.png done');
  
  await browser.close();
})();
