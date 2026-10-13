// Small inline icons (24x24, stroke style) so nothing depends on an icon font or an external request.
const P = {
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>',
  send: '<path d="M22 2 11 13"/><path d="m22 2-7 20-4-9-9-4z"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/>',
  bellOff: '<path d="M18 8a6 6 0 0 0-9.3-5M6.3 6.3C6.1 6.9 6 7.4 6 8c0 7-3 9-3 9h13.5M13.7 21a2 2 0 0 1-3.4 0"/><path d="m3 3 18 18"/>',
  timer: '<circle cx="12" cy="13" r="8"/><path d="M12 9v4l2.5 2.5M9 2h6"/>',
};
export function icon(name, cls = "ico") {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", cls);
  s.setAttribute("aria-hidden", "true");
  s.innerHTML = P[name] || "";
  return s;
}
