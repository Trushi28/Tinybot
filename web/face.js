// TinyBot's face: eyes and mouth are swapped per emotion; blinking and talking are CSS.
const dotEyes = (ry = 9, dy = 0) =>
  `<ellipse class="eye fl" cx="68" cy="${70 + dy}" rx="9" ry="${ry}"/><ellipse class="eye r fl" cx="132" cy="${70 + dy}" rx="9" ry="${ry}"/>`;
const arcEyes = `<path class="st" d="M54 76q14-22 28 0"/><path class="st" d="M118 76q14-22 28 0"/>`;
const heart = (x) => `<path class="fl" transform="translate(${x} 66)" d="M0 -4c-4-9-18-4-12 6l12 12 12-12c6-10-8-15-12-6z"/>`;
const EYES = {
  neutral: dotEyes(),
  joy: arcEyes, ecstatic: arcEyes, proud: arcEyes,
  laugh: `<path class="st" d="M54 70l24 8-24 8"/><path class="st" d="M146 70l-24 8 24 8"/>`,
  love: heart(68) + heart(132),
  anger: dotEyes(8, 4) + `<path class="st" d="M50 50l34 12"/><path class="st" d="M150 50l-34 12"/>`,
  grumpy: dotEyes(6, 4) + `<path class="st" d="M50 52l34 8"/><path class="st" d="M150 52l-34 8"/>`,
  sad: dotEyes(9, 4) + `<path class="st" d="M52 56l30-8"/><path class="st" d="M148 56l-30-8"/>`,
  concern: dotEyes(9, 4) + `<path class="st" d="M52 58l30-10"/><path class="st" d="M148 58l-30-10"/>`,
  wow: `<circle class="st eye" cx="68" cy="68" r="12"/><circle class="st eye r" cx="132" cy="68" r="12"/>`,
  confused: `<ellipse class="eye fl" cx="68" cy="70" rx="6" ry="6"/><ellipse class="eye r fl" cx="132" cy="68" rx="11" ry="11"/>`,
  curious: dotEyes() + `<path class="st" d="M116 50q16-8 32 0"/>`,
  think: `<ellipse class="eye fl" cx="72" cy="62" rx="9" ry="9"/><ellipse class="eye r fl" cx="136" cy="62" rx="9" ry="9"/>`,
  sleepy: `<path class="st" d="M54 74h28"/><path class="st" d="M118 74h28"/>`,
  asleep: `<path class="st" d="M54 74q14 10 28 0"/><path class="st" d="M118 74q14 10 28 0"/>`,
  wink: `<ellipse class="eye fl" cx="68" cy="70" rx="9" ry="9"/><path class="st" d="M118 76q14-22 28 0"/>`,
  sulk: dotEyes(5, 4) + `<path class="st" d="M50 56l34 4"/><path class="st" d="M150 56l-34 4"/>`,
};
const smile = `<path class="st" d="M72 108q28 24 56 0"/>`;
const MOUTH = {
  neutral: `<path class="st" d="M78 114q22 8 44 0"/>`, joy: smile, proud: smile, wink: smile, love: smile,
  ecstatic: `<path class="fl" d="M68 104q32 44 64 0z"/>`, laugh: `<path class="fl" d="M66 102q34 50 68 0z"/>`,
  anger: `<path class="st" d="M76 124q24-18 48 0"/>`, grumpy: `<path class="st" d="M78 120q22-8 44 0"/>`, sulk: `<path class="st" d="M80 122q20-10 40 0"/>`,
  sad: `<path class="st" d="M76 124q24-20 48 0"/>`, concern: `<path class="st" d="M80 120q20-8 40 0"/>`,
  wow: `<ellipse class="st" cx="100" cy="118" rx="10" ry="13"/>`, confused: `<path class="st" d="M72 118q9-9 18 0t18 0t18 0"/>`,
  curious: `<path class="st" d="M82 116h30"/>`, think: `<path class="st" d="M84 116h24"/>`,
  sleepy: `<path class="st" d="M84 116h32"/>`, asleep: `<ellipse class="st" cx="100" cy="116" rx="8" ry="5"/>`,
};
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

/** @param {SVGElement} svg an <svg class="face"> containing <g class="eyes"> and <g class="mouth">; moodEl is optional */
export function createFace(svg, moodEl) {
  const eyes = svg.querySelector(".eyes"), mouth = svg.querySelector(".mouth");
  let calm = 0;
  function set(emo, hold = 5000) {
    const e = EYES[emo] ? emo : "neutral";
    eyes.innerHTML = EYES[e];
    mouth.innerHTML = MOUTH[e] || MOUTH.neutral;
    svg.dataset.emo = e;
    if (moodEl) moodEl.textContent = e;
    if (!reduceMotion) {
      svg.classList.remove("swap");
      void svg.getBoundingClientRect();
      svg.classList.add("swap");
    }
    clearTimeout(calm);
    if (e !== "neutral") calm = setTimeout(() => set("neutral"), hold);
  }
  set("neutral");
  return { set, talking: (on) => svg.classList.toggle("talking", on) };
}
