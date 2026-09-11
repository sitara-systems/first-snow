// Shared crystal-growth engine for both demo/index.html (the puck) and
// demo/sliders.html (named sliders). Extracted from the original single-file
// demo so the validated WebGL2 shaders, step-budget table, and pacing logic
// exist in exactly one place -- this code has already been through several
// rounds of real bug fixes (a display-transform shear bug, a step-overflow
// bug, a jagged-edge fix), and duplicating it across two files would mean
// re-fixing each bug twice, or worse, fixing it in one copy and not the
// other. See PLAN.md for the full history.
//
// Public API: `createCrystalEngine({canvas, wrap, getParams, onComplete})`.
// `getParams()` must return `{gamma, beta, sigma}` (sigma optional, defaults
// to DEFAULT_SIGMA) every time it's called -- it is read fresh every
// simulation step, so a caller whose params change live (dragging a puck,
// moving a slider) gets growth that responds while it's happening, per the
// Phase 4 acceptance bar. `onComplete(beta)` is called once when growth
// finishes; it should return the label text to show (see `morphologyLabel`
// below for the puck's mapping, or supply a different one).

const GAMMA_RANGE = [0.40, 0.55];
const BETA_RANGE = [1.10, 2.60];
const BASE = {alpha:0.08, theta:0.025, kappa:0.005, mu:0.06, upsilon:0.0001};
const DEFAULT_SIGMA = 0.0006; // small, deliberate imperfection -- see docs/performance.md's symmetry comparison
function clamp01(x){ return Math.max(0, Math.min(1, x)); }
function lerp(a,b,t){ return a+(b-a)*t; }

// The fern/stellar-dendrite band (beta 1.10-1.60) is a third of the
// validated beta range but is where nearly all the visible branching
// variety lives; the sectored-plate/simple-plate band (1.60-2.60) is the
// other two-thirds and, rendered, looks like minor variations on a plain
// hexagon regardless of exactly where beta sits in it. A straight linear
// mapping therefore under-represents the interesting end. BETA_CURVE_POWER
// compresses the "looks-the-same" end and expands the varied end -- see
// PLAN.md's sixth 2026-09-11 finding for the full account and the puck
// screen-space measurement that motivated it.
const BETA_CURVE_POWER = 1.6;
function puckToParams(px, py){
  // px, py in [0,1], puck-area local coordinates.
  // py: supersaturation axis (bottom=low, top=high) -> gamma
  // px: temperature axis (left=cold, right=warm) -> beta
  const gamma = lerp(GAMMA_RANGE[0], GAMMA_RANGE[1], clamp01(1-py));
  const beta = BETA_RANGE[0] + (BETA_RANGE[1]-BETA_RANGE[0]) * Math.pow(clamp01(px), BETA_CURVE_POWER);
  return {gamma, beta};
}
function morphologyLabel(beta){
  if (beta <= 1.30) return "Fern";
  if (beta <= 1.60) return "Stellar Dendrite";
  if (beta <= 2.00) return "Sectored Plate";
  return "Simple Plate";
}

const VS_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }
`;

const FS_DIFFUSE_SRC = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform vec2 uTexel;
in vec2 vUv;
out vec4 outColor;
vec4 samp(vec2 off){ return texture(uState, vUv+off*uTexel); }
void main(){
  vec4 self = samp(vec2(0.0));
  if (self.r > 0.5) { outColor = self; return; }
  vec2 offs[6] = vec2[6](vec2(0,1),vec2(0,-1),vec2(-1,0),vec2(1,0),vec2(-1,-1),vec2(1,1));
  float sum = self.a;
  for (int i=0;i<6;i++){
    vec4 n = samp(offs[i]);
    sum += (n.r > 0.5) ? self.a : n.a;
  }
  outColor = vec4(self.r, self.g, self.b, sum/7.0);
}
`;

const FS_UPDATE_SRC = `#version 300 es
precision highp float;
// Freeze -> decide-attach -> melt (always) -> commit-attach, matching
// docs/gg-model.md's synchronous equations exactly (Phase 3-validated
// ordering; see scripts/webgl_port/index.html, the parent of this file).
uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uBeta, uTheta, uAlpha, uKappa, uMu, uUpsilon, uSigma;
uniform float uStep, uSeed;
in vec2 vUv;
out vec4 outColor;
vec4 samp(vec2 off){ return texture(uState, vUv+off*uTexel); }
float hash(vec2 p, float step, float seed){
  vec3 p3 = vec3(p.x,p.y, step+seed*977.0);
  p3 = fract(p3*0.1031);
  p3 += dot(p3, p3.yzx+33.33);
  return fract((p3.x+p3.y)*p3.z);
}
void main(){
  vec4 self = samp(vec2(0.0));
  float a=self.r, b=self.g, c=self.b, d=self.a;
  if (a > 0.5) { outColor = self; return; }
  vec2 offs[6] = vec2[6](vec2(0,1),vec2(0,-1),vec2(-1,0),vec2(1,0),vec2(-1,-1),vec2(1,1));
  int attachedCount = 0;
  float neighborDSum = d;
  for (int i=0;i<6;i++){
    vec4 n = samp(offs[i]);
    if (n.r > 0.5) attachedCount++;
    neighborDSum += n.a;
  }
  bool boundary = attachedCount > 0;
  if (boundary) {
    b = b + (1.0-uKappa)*d;
    c = c + uKappa*d;
    d = 0.0;
    bool attach;
    if (attachedCount <= 2) attach = b >= uBeta;
    else if (attachedCount == 3) attach = (b >= 1.0) || (neighborDSum < uTheta && b >= uAlpha);
    else attach = true;
    float bM = (1.0-uMu)*b;
    float cM = (1.0-uUpsilon)*c;
    float dM = d + uMu*b + uUpsilon*c;
    b = bM; c = cM; d = dM;
    if (attach) { c = b + c; b = 0.0; a = 1.0; }
  } else {
    float r = hash(gl_FragCoord.xy, uStep, uSeed);
    d = (r < 0.5) ? (1.0-uSigma)*d : (1.0+uSigma)*d;
  }
  outColor = vec4(a,b,c,d);
}
`;

const FS_VIEW_SRC = `#version 300 es
precision highp float;
// The six neighbor offsets used by the update shader -- (0,1),(0,-1),
// (1,0),(-1,0),(1,1),(-1,-1) -- are the integer coordinates of a
// triangular lattice's nearest neighbors, stored the way the reference
// stores them (engine.py's get_neighbors), NOT literal Euclidean unit
// vectors. Embedding them correctly into real (equal-angle, equal-length)
// space requires a SHEAR, not a rotation: physical = (x - 0.5*y, y*sqrt(3)/2).
// Do not replace this with a rotation without redoing that numeric check --
// see PLAN.md's 2026-09-11 correction entries for how many wrong attempts
// that took and why a numeric pixel measurement, not a visual comparison,
// is what actually confirmed it.
uniform sampler2D uState;
uniform vec2 uPixel; // one screen pixel, in vUv (0..1) units
in vec2 vUv;
out vec4 outColor;
// 4x rotated-grid supersampling in screen space -- see PLAN.md's eighth
// 2026-09-11 finding: a single hard sample per pixel against a 384x384
// lattice magnified onto a much larger canvas produced a visible staircase
// on every edge. Jittering before the shear reflects actual on-screen pixel
// coverage; does not touch the simulation's own textures or filtering.
float sampleV(vec2 uv){
  vec2 centered = (uv - 0.5) * 2.0;
  vec2 simSpace = vec2(centered.x + centered.y/1.7320508, centered.y*1.1547005);
  vec2 simUv = simSpace*0.5 + 0.5;
  if (simUv.x < 0.0 || simUv.x > 1.0 || simUv.y < 0.0 || simUv.y > 1.0) return 0.0;
  vec4 s = texture(uState, simUv);
  return s.r > 0.5 ? clamp(s.b,0.0,1.0) : 0.0;
}
void main(){
  vec2 o1 = uPixel * vec2( 0.125,  0.375);
  vec2 o2 = uPixel * vec2( 0.375, -0.125);
  vec2 o3 = uPixel * vec2(-0.125, -0.375);
  vec2 o4 = uPixel * vec2(-0.375,  0.125);
  float v = (sampleV(vUv+o1) + sampleV(vUv+o2) + sampleV(vUv+o3) + sampleV(vUv+o4)) * 0.25;
  vec3 crystalColor = mix(vec3(0.02,0.027,0.039), vec3(0.93,0.96,1.0), v);
  outColor = vec4(crystalColor, 1.0);
}
`;

// Tried raising this to 512 (docs/performance.md's own upper end of its
// "256 or 512, not 1024" recommendation) for finer branch detail, and
// reverted after measuring the actual cost at high beta: step count to
// reach 72% of the boundary does NOT scale linearly with SIZE the way
// overflow-margin reasoning assumed. At beta=2.05 it went from 30,300
// steps (SIZE=384) to 98,000 (SIZE=512) -- more than 3x, not the ~1.33x
// a linear radius-vs-lattice relationship predicts. At beta=2.20, 512
// didn't even reach 72% within 160,000 steps (was 38,400 at 384). At the
// true beta=2.60 extreme, 512 reached only 52% of the boundary after
// 250,000 steps (was a clean 70,800 steps to 72% at 384). Whatever is
// happening -- some interaction between the reflecting boundary, the
// larger domain, and the high-beta regime's already-slow kinetics -- it
// makes high-beta growth qualitatively worse at 512, not just slower to
// finish, and was not something the 256-vs-512-vs-1024 throughput
// recommendation (based on raw steps/sec, not this radius-vs-steps
// relationship) anticipated. Left at 384, the value this project's own
// prior testing actually validated at this level of detail. See PLAN.md's
// tenth 2026-09-11 finding -- worth understanding properly before trying
// this again, not re-attempting on a hunch that more headroom helps.
const SIZE = 384;

// Steps needed to reach a "developed but safe" crystal (72% of the way to
// the lattice's reflecting boundary, measured as the raw sim-texture's
// bounding-box radius, transform-independent) is NOT linear in beta --
// measured directly (not extrapolated) at gamma=0.55 (the fastest-growing
// gamma in the live range, so every other gamma at the same beta is at
// least this safe). See PLAN.md's fifth 2026-09-11 finding for the
// measurement methodology; re-measure this table if BETA_RANGE,
// GAMMA_RANGE, or the base kappa/mu/alpha/theta values ever change.
const BETA_STEP_TABLE = [
  [1.10, 1050], [1.15, 1200], [1.20, 1450], [1.25, 1650],
  [1.30, 2100], [1.35, 2600], [1.40, 3200], [1.50, 5700],
  [1.60, 9750], [1.75, 17850], [1.90, 26400], [2.05, 30300],
  [2.20, 38400], [2.35, 47700], [2.50, 60300], [2.60, 70800],
];
function targetStepsFor(beta){
  const b = Math.max(BETA_STEP_TABLE[0][0], Math.min(BETA_STEP_TABLE[BETA_STEP_TABLE.length-1][0], beta));
  for (let i=0;i<BETA_STEP_TABLE.length-1;i++){
    const [b0,s0] = BETA_STEP_TABLE[i], [b1,s1] = BETA_STEP_TABLE[i+1];
    if (b <= b1) return Math.round(lerp(s0, s1, clamp01((b-b0)/(b1-b0))));
  }
  return BETA_STEP_TABLE[BETA_STEP_TABLE.length-1][1];
}
// Every crystal grows for the same fixed real-world duration, regardless
// of its step budget -- a deliberate product decision (every visitor gets
// the same-length moment, and a progress bar showing "how much is left"
// only means the same thing for everyone if that "everything" is a fixed
// duration). This was previously duration-scaled-to-step-budget instead
// (see PLAN.md's fifth 2026-09-11 finding) to avoid the fastest corner's
// small step budget looking sparse when stretched across a long fixed
// window -- that trade-off is now accepted deliberately rather than
// avoided: a Fern's ~1,050 steps over GROWTH_DURATION_MS is a real step
// roughly every 1.4 rendered frames rather than every frame, visibly
// sparser than a Simple Plate's ~70,800 steps over the same window, but
// still incremental and still holds for the reflecting-boundary safety
// margin the step table was built for -- it never overflows, it just
// updates less often for shapes that mature faster.
const GROWTH_DURATION_MS = 30000;

function createCrystalEngine({canvas, wrap, getParams, onComplete, onProgress}){
  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  function compile(type, src){
    const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }
  function link(vsSrc, fsSrc){
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  // gl.getUniformLocation/getAttribLocation are driver round-trips -- cache
  // once at link time instead of querying fresh every simulation step. See
  // PLAN.md's seventh 2026-09-11 finding.
  function programInfo(vs, fs, uniformNames){
    const prog = link(vs, fs);
    const u = {};
    for (const name of uniformNames) u[name] = gl.getUniformLocation(prog, name);
    return {prog, aPos: gl.getAttribLocation(prog, 'aPos'), u};
  }
  const piDiffuse = programInfo(VS_SRC, FS_DIFFUSE_SRC, ['uState','uTexel']);
  const piUpdate = programInfo(VS_SRC, FS_UPDATE_SRC,
    ['uState','uTexel','uBeta','uTheta','uAlpha','uKappa','uMu','uUpsilon','uSigma','uStep','uSeed']);
  const piView = programInfo(VS_SRC, FS_VIEW_SRC, ['uState','uPixel']);
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);

  function makeState(size){
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,size,size,0,gl.RGBA,gl.FLOAT,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return {tex, fbo};
  }
  function initialData(size, rho){
    const data = new Float32Array(size*size*4);
    for (let i=0;i<size*size;i++){ data[i*4+3]=rho; }
    const c = Math.floor(size/2);
    const ci = (c*size+c)*4;
    data[ci+0]=1; data[ci+1]=0; data[ci+2]=1; data[ci+3]=0;
    return data;
  }
  function drawQuad(pi, w, h, setUniforms){
    gl.viewport(0,0,w,h);
    gl.useProgram(pi.prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(pi.aPos);
    gl.vertexAttribPointer(pi.aPos,2,gl.FLOAT,false,0,0);
    if (setUniforms) setUniforms(pi.u);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  }

  let bufs, cur, stepCount, seed, running, startTime, currentGamma, currentBeta, currentSigma;

  function resizeCanvas(){
    // Square canvas: the reference's own final thumbnails are always
    // square (graphics.py's save_image(resize=N) forces NxN regardless of
    // the rotated/squished intermediate image's aspect ratio).
    const side = Math.round(Math.min(wrap.clientWidth, wrap.clientHeight) * 0.92);
    canvas.style.width = side + "px";
    canvas.style.height = side + "px";
    // The canvas's backing store (canvas.width/height) is a display-only
    // resolution -- independent of SIZE, the simulation lattice above. An
    // earlier version left this at the CSS pixel size directly, which
    // under-renders on a high-DPI display (a Surface Pro's screen is
    // notably dense): the browser then upscales the WebGL output like any
    // other undersized image, going soft exactly where a dense screen
    // would otherwise show the 4x supersampled edges above at full
    // sharpness. Matching devicePixelRatio costs nothing simulation-side
    // (view shader cost scales with canvas pixel count, but that shader is
    // cheap -- 4 texture taps -- and runs once per rendered frame, not per
    // simulation step, unlike SIZE).
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(side * dpr);
    canvas.height = Math.round(side * dpr);
  }

  function readParams(){
    const p = getParams();
    return {gamma: p.gamma, beta: p.beta, sigma: p.sigma == null ? DEFAULT_SIGMA : p.sigma};
  }

  function resetCrystal(){
    if (bufs) { gl.deleteTexture(bufs[0].tex); gl.deleteTexture(bufs[1].tex);
                gl.deleteFramebuffer(bufs[0].fbo); gl.deleteFramebuffer(bufs[1].fbo); }
    const params = readParams();
    bufs = [makeState(SIZE), makeState(SIZE)];
    cur = 0;
    gl.bindTexture(gl.TEXTURE_2D, bufs[0].tex);
    gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,SIZE,SIZE,gl.RGBA,gl.FLOAT, initialData(SIZE, params.gamma));
    stepCount = 0;
    seed = Date.now() % 100000; // deterministic given this value -- hard rule 3: seeded, not OS-entropy-unseeded
    running = true;
    startTime = performance.now();
    renderView();
  }

  function stepSimulation(n){
    const texel = [1/SIZE, 1/SIZE];
    const p = readParams();
    currentGamma = p.gamma; currentBeta = p.beta; currentSigma = p.sigma;
    for (let i=0;i<n;i++){
      const srcBuf = bufs[cur], tmpBuf = bufs[1-cur];
      gl.bindFramebuffer(gl.FRAMEBUFFER, tmpBuf.fbo);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcBuf.tex);
      drawQuad(piDiffuse, SIZE, SIZE, u=>{
        gl.uniform1i(u.uState,0);
        gl.uniform2fv(u.uTexel,texel);
      });
      gl.bindFramebuffer(gl.FRAMEBUFFER, srcBuf.fbo);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tmpBuf.tex);
      drawQuad(piUpdate, SIZE, SIZE, u=>{
        gl.uniform1i(u.uState,0);
        gl.uniform2fv(u.uTexel,texel);
        gl.uniform1f(u.uBeta, currentBeta);
        gl.uniform1f(u.uTheta, BASE.theta);
        gl.uniform1f(u.uAlpha, BASE.alpha);
        gl.uniform1f(u.uKappa, BASE.kappa);
        gl.uniform1f(u.uMu, BASE.mu);
        gl.uniform1f(u.uUpsilon, BASE.upsilon);
        gl.uniform1f(u.uSigma, currentSigma);
        gl.uniform1f(u.uStep, stepCount+i);
        gl.uniform1f(u.uSeed, seed);
      });
    }
    stepCount += n;
  }

  function renderView(){
    const finalBuf = bufs[cur];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, finalBuf.tex);
    drawQuad(piView, canvas.width, canvas.height, u=>{
      gl.uniform1i(u.uState,0);
      gl.uniform2f(u.uPixel, 1/canvas.width, 1/canvas.height);
    });
  }

  function tick(){
    if (!running) return;
    const peek = readParams();
    const totalTarget = targetStepsFor(peek.beta);
    const elapsed = performance.now() - startTime;
    const targetProgress = clamp01(elapsed / GROWTH_DURATION_MS);
    const targetSteps = Math.floor(targetProgress * totalTarget);
    const toRun = Math.max(0, Math.min(targetSteps - stepCount, 400)); // cap per-frame batch so a slow device degrades gracefully instead of jank-freezing
    if (toRun > 0) stepSimulation(toRun);
    renderView();
    if (onProgress) onProgress(targetProgress, Math.max(0, GROWTH_DURATION_MS - elapsed));
    if (targetProgress >= 1) {
      running = false;
      if (onComplete) onComplete(currentBeta, currentGamma, currentSigma);
    }
  }
  function frame(){ requestAnimationFrame(frame); tick(); }

  resizeCanvas();
  resetCrystal();
  requestAnimationFrame(frame);

  return {
    reset: resetCrystal,
    resize: resizeCanvas,
    isRunning: () => running,
    // Calibration/testing hook only -- lets a measurement script (see
    // PLAN.md for the BETA_STEP_TABLE methodology) drive raw simulation
    // steps and read back the sim texture directly, bypassing the
    // time-paced frame() loop. Not used by either front end's own UI.
    _debug: {
      gl,
      fastForward(){ startTime = performance.now() - GROWTH_DURATION_MS - 1; },
      tick,
      stepSimulation,
      readAttachedBoundingRadiusFrac(){
        const buf = bufs[cur];
        gl.bindFramebuffer(gl.FRAMEBUFFER, buf.fbo);
        const px = new Float32Array(SIZE*SIZE*4);
        gl.readPixels(0,0,SIZE,SIZE,gl.RGBA,gl.FLOAT,px);
        const c = SIZE/2;
        let maxR = 0;
        for (let y=0;y<SIZE;y++) for (let x=0;x<SIZE;x++){
          if (px[(y*SIZE+x)*4] > 0.5){
            const dx=x-c, dy=y-c;
            const r = Math.sqrt(dx*dx+dy*dy);
            if (r>maxR) maxR = r;
          }
        }
        return maxR/c;
      },
    },
  };
}
