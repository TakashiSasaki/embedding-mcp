const copyButton = document.querySelector("#copy-endpoint");
const endpoint = document.querySelector("#mcp-url");
const feedback = document.querySelector("#copy-feedback");

copyButton.hidden = false;
copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(endpoint.value);
    feedback.textContent = "URL をコピーしました。";
  } catch {
    endpoint.focus();
    endpoint.select();
    feedback.textContent = "選択した URL をコピーしてください。";
  }
});
