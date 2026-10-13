// Colour theme: follows the system until the visitor picks one with the button, then remembers it.
const root = document.documentElement;
try {
  const t = localStorage.getItem("tinybot.theme");
  if (t === "light" || t === "dark") root.dataset.theme = t;
} catch {}
export function wireTheme(button) {
  button?.addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem("tinybot.theme", root.dataset.theme);
    } catch {}
  });
}
