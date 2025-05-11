chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
    const url = details.url;
    console.log("Navigating to:", details.url);

    // Skip internal Chrome pages
    if (!url.startsWith("http://") && !url.startsWith("https://")) return;
  
    try {
      const response = await fetch("http://localhost:5000/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      });
  
      const result = await response.json();
      if (result.is_malicious) {
        const warningUrl = chrome.runtime.getURL(`warning.html?url=${encodeURIComponent(url)}`);
  
        // Redirect the entire tab instead of injecting script
        chrome.tabs.update(details.tabId, { url: warningUrl });
      }
    } catch (err) {
      console.error("Error checking URL:", err);
    }
  }, {
    url: [{ schemes: ["http", "https"] }]
  });
  