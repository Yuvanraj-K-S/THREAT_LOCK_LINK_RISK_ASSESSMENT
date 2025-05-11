document.getElementById("continue").addEventListener("click", () => {
    const urlParams = new URLSearchParams(window.location.search);
    const targetUrl = urlParams.get("url");
    if (targetUrl) {
      window.location.href = targetUrl;
    }
  });
  