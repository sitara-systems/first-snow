// column-engine.js — PROTOTYPE. Real snow crystal *habit* (column vs.
// plate vs. needle, the vertical axis of the classic Nakaya diagram) comes
// from ice's prism facets having different attachment kinetics than its
// basal facet -- a genuinely 3D phenomenon. The Gravner-Griffeath model
// index.html/sliders.html run (see docs/gg-model.md) simulates only the
// basal facet in 2D; it has no representation of the prism axis at all,
// so it cannot produce a column or needle by itself.
//
// This file does NOT add a second physics simulation for the prism facet
// (that's a research-grade 3D problem -- see PLAN.md's 2026-09-12 "3D
// habit prototype" finding for what was actually surveyed). Instead it
// runs the SAME real, validated 2D CA (shaders copied verbatim from
// engine.js) and extrudes its output along z using a heuristic: whenever
// the 2D crystal's growth stalls, treat that stall as "the growth that
// would otherwise be happening on the prism facet instead" and add
// height proportionally. This is a plausible-looking design choice, not
// a derived physical result -- flagged deliberately, per this project's
// hard rules, rather than presented as a real 3D simulation.
//
// The full beta range (fern through simple plate) is accepted here, not
// just the high-beta "hexagon" end -- see hexagonCompatibility() below.
// Real snow crystals never come as extruded dendrites (a live report's
// own framing), so height is tapered toward zero as beta moves into
// branchy (fern/stellar-dendrite) territory, using the same beta bands
// engine.js's morphologyLabel() already documents. The boundary trace
// itself (traceBoundary, further down) is exact marching squares, not an
// angle-sampled approximation -- it correctly resolves a fern's concave
// notches and side-branch structure, not just a solid hexagon's outline.
// An earlier version used a 180-ray radial march instead, which could
// only ever be right for star-shaped, non-branching blobs; see PLAN.md's
// 2026-09-12 entries for both that limitation and this fix.
//
// Kept as a separate file from engine.js on purpose: this is a spike, and
// duplicating the ~30 lines of shader/stepping plumbing it actually needs
// is cheaper than risking a regression in the two shipped pages that
// depend on engine.js.

const COL_SIZE = 384;
const COL_BASE = {alpha:0.08, theta:0.025, kappa:0.005, mu:0.06, upsilon:0.0001};

const COL_VS_SRC = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.0,1.0); }
`;

// Identical to engine.js's FS_DIFFUSE_SRC (open boundary -- see its own
// comment there for why) and FS_UPDATE_SRC (the validated GG update rule
// plus the fp32-safe noise hash) -- copied verbatim, not reimplemented,
// so this prototype's 2D growth is exactly as real as the shipped pages'.
const COL_FS_DIFFUSE_SRC = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uGamma;
in vec2 vUv;
out vec4 outColor;
vec4 samp(vec2 off){ return texture(uState, vUv+off*uTexel); }
void main(){
  if (vUv.x < uTexel.x || vUv.y < uTexel.y || vUv.x > 1.0-uTexel.x || vUv.y > 1.0-uTexel.y) {
    vec4 edgeSelf = samp(vec2(0.0));
    outColor = vec4(edgeSelf.r, edgeSelf.g, edgeSelf.b, uGamma);
    return;
  }
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

const COL_FS_UPDATE_SRC = `#version 300 es
precision highp float;
uniform sampler2D uState;
uniform vec2 uTexel;
uniform float uBeta, uTheta, uAlpha, uKappa, uMu, uUpsilon, uSigma;
uniform float uStep, uSeed;
in vec2 vUv;
out vec4 outColor;
vec4 samp(vec2 off){ return texture(uState, vUv+off*uTexel); }
float hash(vec2 p, float step, float seed){
  vec3 p3 = vec3(p.x,p.y, step + fract(seed*0.7137253)*1000.0);
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

// Runs the 2D CA headlessly to TARGET_RADIUS_FRAC (matching engine.js's
// same-final-size fix -- see PLAN.md), capturing a simplified marching-
// squares boundary trace (NOT the full mask -- see traceBoundary below)
// at intervals, then in a second pass assigns each captured frame a
// height via the stall heuristic described at the top of this file.
//
// aspectRatio: total final height AT FULL hexagonCompatibility, as a
// multiple of the crystal's own target xy radius (both in the same
// lattice-pixel units -- see traceBoundary below) -- 0.15 reads as a
// squat plate, ~4 as a slender needle. The ACTUAL total height is this
// scaled by hexagonCompatibility(beta) (see below), so a low (fern) beta
// ends up nearly flat regardless of aspectRatio. Untuned -- see PLAN.md's
// prototype finding. The *distribution* of that budget across frames
// (not its total) is what the stall heuristic controls: intervals where
// xy growth stalled get a bigger share of it.
async function growColumnCrystal({beta = 2.2, gamma = 0.50, sigma = 0.0006, aspectRatio = 0.8,
                                   numFrames = 90, chunkSteps = 2000, targetRadiusFrac = 0.72,
                                   onProgress} = {}){
  const canvas = document.createElement('canvas');
  canvas.width = COL_SIZE; canvas.height = COL_SIZE;
  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  function compile(type, src){
    const sh = gl.createShader(type); gl.shaderSource(sh, src); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  }
  function link(vs, fs){
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }
  function programInfo(vs, fs, names){
    const prog = link(vs, fs);
    const u = {};
    for (const n of names) u[n] = gl.getUniformLocation(prog, n);
    return {prog, aPos: gl.getAttribLocation(prog, 'aPos'), u};
  }
  const piDiffuse = programInfo(COL_VS_SRC, COL_FS_DIFFUSE_SRC, ['uState','uTexel','uGamma']);
  const piUpdate = programInfo(COL_VS_SRC, COL_FS_UPDATE_SRC,
    ['uState','uTexel','uBeta','uTheta','uAlpha','uKappa','uMu','uUpsilon','uSigma','uStep','uSeed']);
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
  function drawQuad(pi, setUniforms){
    gl.viewport(0,0,COL_SIZE,COL_SIZE);
    gl.useProgram(pi.prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(pi.aPos);
    gl.vertexAttribPointer(pi.aPos,2,gl.FLOAT,false,0,0);
    if (setUniforms) setUniforms(pi.u);
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  }
  function makeState(){
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,COL_SIZE,COL_SIZE,0,gl.RGBA,gl.FLOAT,null);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    return {tex, fbo};
  }

  let bufs = [makeState(), makeState()];
  let cur = 0;
  const data = new Float32Array(COL_SIZE*COL_SIZE*4);
  for (let i=0;i<COL_SIZE*COL_SIZE;i++) data[i*4+3] = gamma;
  const c0 = Math.floor(COL_SIZE/2), ci = (c0*COL_SIZE+c0)*4;
  data[ci]=1; data[ci+1]=0; data[ci+2]=1; data[ci+3]=0;
  gl.bindTexture(gl.TEXTURE_2D, bufs[0].tex);
  gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,COL_SIZE,COL_SIZE,gl.RGBA,gl.FLOAT,data);

  const texel = [1/COL_SIZE, 1/COL_SIZE];
  const seed = Date.now() % 100000; // hard rule 3: seeded, not unseeded entropy
  let stepCount = 0;
  function stepFixed(n){
    for (let i=0;i<n;i++){
      const src = bufs[cur], tmp = bufs[1-cur];
      gl.bindFramebuffer(gl.FRAMEBUFFER, tmp.fbo);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, src.tex);
      drawQuad(piDiffuse, u=>{
        gl.uniform1i(u.uState,0); gl.uniform2fv(u.uTexel,texel); gl.uniform1f(u.uGamma, gamma);
      });
      gl.bindFramebuffer(gl.FRAMEBUFFER, src.fbo);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tmp.tex);
      drawQuad(piUpdate, u=>{
        gl.uniform1i(u.uState,0); gl.uniform2fv(u.uTexel,texel);
        gl.uniform1f(u.uBeta, beta); gl.uniform1f(u.uTheta, COL_BASE.theta);
        gl.uniform1f(u.uAlpha, COL_BASE.alpha); gl.uniform1f(u.uKappa, COL_BASE.kappa);
        gl.uniform1f(u.uMu, COL_BASE.mu); gl.uniform1f(u.uUpsilon, COL_BASE.upsilon);
        gl.uniform1f(u.uSigma, sigma); gl.uniform1f(u.uStep, stepCount+i); gl.uniform1f(u.uSeed, seed);
      });
    }
    stepCount += n;
  }

  const pixels = new Float32Array(COL_SIZE*COL_SIZE*4);
  function readback(){
    gl.bindFramebuffer(gl.FRAMEBUFFER, bufs[cur].fbo);
    gl.readPixels(0,0,COL_SIZE,COL_SIZE,gl.RGBA,gl.FLOAT,pixels);
  }
  function boundingRadiusFrac(){
    const c = COL_SIZE/2;
    let maxR2 = 0;
    for (let y=0;y<COL_SIZE;y++) for (let x=0;x<COL_SIZE;x++){
      if (pixels[(y*COL_SIZE+x)*4] > 0.5){
        const dx=x-c, dy=y-c, r2=dx*dx+dy*dy;
        if (r2>maxR2) maxR2 = r2;
      }
    }
    return Math.sqrt(maxR2)/c;
  }
  // Exact pixel-precision boundary via marching squares, replacing an
  // earlier 180-ray radial march that only worked for solid, star-shaped
  // hexagons -- it silently threw away any concave detail (a fern's
  // notches between arms, side-branch structure) *before* tracing, since
  // it could only record one radius per angle. Marching squares finds
  // every crossing of the binary attached/unattached field, so a
  // dendrite's actual fingered outline comes through, not an approximate
  // envelope of it. See this file's top comment and PLAN.md's
  // 2026-09-12 entry.
  //
  // The attached region is a single connected blob with no interior
  // holes by construction (FS_UPDATE_SRC: any cell with >=4 attached
  // neighbors always attaches on its very next step, so a fully-enclosed
  // unattached pocket can't persist to the frames this captures) -- so
  // this always expects exactly one closed loop, but defensively returns
  // the largest if more than one ever turns up.
  function marchingSquaresBoundary(){
    const isIn = (x,y) => pixels[(y*COL_SIZE+x)*4] > 0.5;
    const midpoint = new Map(); // canonical edge id -> [x,y] grid-space midpoint
    const segs = []; // pairs of edge ids, one per boundary crossing
    function hId(gx,gy){ const id = 'h,'+gx+','+gy; if(!midpoint.has(id)) midpoint.set(id,[gx+0.5,gy]); return id; }
    function vId(gx,gy){ const id = 'v,'+gx+','+gy; if(!midpoint.has(id)) midpoint.set(id,[gx,gy+0.5]); return id; }
    for (let y=0;y<COL_SIZE-1;y++){
      for (let x=0;x<COL_SIZE-1;x++){
        const tl=isIn(x,y), tr=isIn(x+1,y), br=isIn(x+1,y+1), bl=isIn(x,y+1);
        const c = (tl?1:0)|(tr?2:0)|(br?4:0)|(bl?8:0);
        if (c===0 || c===15) continue;
        const top=hId(x,y), bottom=hId(x,y+1), left=vId(x,y), right=vId(x+1,y);
        // Standard marching-squares case table (TL=1,TR=2,BR=4,BL=8).
        // Cases 5 and 10 are the ambiguous "diagonal corners" saddle --
        // resolved here by always treating the two corners as separate
        // (not diagonally connected), a fixed, arbitrary tie-break with
        // no universally "correct" answer; see PLAN.md.
        switch(c){
          case 1: segs.push([top,left]); break;
          case 2: segs.push([top,right]); break;
          case 3: segs.push([left,right]); break;
          case 4: segs.push([right,bottom]); break;
          case 5: segs.push([top,left]); segs.push([right,bottom]); break;
          case 6: segs.push([top,bottom]); break;
          case 7: segs.push([left,bottom]); break;
          case 8: segs.push([bottom,left]); break;
          case 9: segs.push([top,bottom]); break;
          case 10: segs.push([top,right]); segs.push([bottom,left]); break;
          case 11: segs.push([right,bottom]); break;
          case 12: segs.push([left,right]); break;
          case 13: segs.push([top,right]); break;
          case 14: segs.push([top,left]); break;
        }
      }
    }
    if (segs.length === 0) return [];
    const adj = new Map();
    for (const [a,b] of segs){
      if (!adj.has(a)) adj.set(a,[]);
      if (!adj.has(b)) adj.set(b,[]);
      adj.get(a).push(b);
      adj.get(b).push(a);
    }
    const visited = new Set();
    let best = [];
    for (const startId of adj.keys()){
      if (visited.has(startId)) continue;
      const loop = [];
      let prev = null, curr = startId;
      while (curr != null && !visited.has(curr)){
        visited.add(curr);
        loop.push(curr);
        const neighbors = adj.get(curr) || [];
        const next = neighbors.find(n => n !== prev);
        prev = curr;
        curr = next === undefined ? neighbors[0] : next;
        if (curr === startId) break;
      }
      if (loop.length > best.length) best = loop;
    }
    return best.map(id => midpoint.get(id));
  }
  // Ramer-Douglas-Peucker simplification of a CLOSED polygon: rotate to
  // start at an extreme (stable, far-apart) point so the open-path RDP
  // below has two well-separated anchors, run it, then drop the
  // duplicated closing point. Pixel-precise marching squares on a 384x384
  // grid can return thousands of points on a long run's most-developed
  // frame; this brings that down to a size ExtrudeGeometry can rebuild
  // every rendered frame during playback without becoming the bottleneck,
  // while keeping real concave detail an angle-sampled radial trace
  // would have discarded outright.
  // Splits the loop at two well-separated points (the min-x point and its
  // opposite by array index) into two OPEN arcs, each simplified by
  // standard RDP with those two points as fixed endpoints, then rejoined.
  // A first version instead rotated the loop to start/end at one single
  // anchor and ran RDP on that as one "open" path -- broken, because RDP
  // measures distance from the chord between the array's first and last
  // point, and a closed loop's first and last point are the same point,
  // making that chord's length zero and its "distance" to every other
  // point zero too, collapsing the entire polygon to one point on the
  // very first call. Caught by inspecting rawLen vs. simplifiedLen
  // directly rather than just eyeballing the render.
  function simplifyClosedPolygon(points, epsilon){
    if (points.length < 6) return points;
    let a = 0;
    for (let i=1;i<points.length;i++){
      if (points[i][0] < points[a][0] || (points[i][0] === points[a][0] && points[i][1] < points[a][1])) a = i;
    }
    const b = (a + Math.floor(points.length/2)) % points.length;
    function arc(from, to){
      const out = [];
      let i = from;
      while (true){ out.push(points[i]); if (i === to) break; i = (i+1) % points.length; }
      return out;
    }
    const s1 = rdp(arc(a,b), epsilon); // a..b, endpoints preserved
    const s2 = rdp(arc(b,a), epsilon); // b..a, endpoints preserved
    // s1 ends at b (s2's first point) and s2 ends at a (s1's first point,
    // i.e. where the caller's implicit "close the loop" will land) --
    // drop both to avoid duplicate vertices at the seams.
    return s1.concat(s2.slice(1, -1));
  }
  function rdp(points, epsilon){
    if (points.length < 3) return points.slice();
    let maxDist = 0, idx = 0;
    const [x1,y1] = points[0], [x2,y2] = points[points.length-1];
    const dx = x2-x1, dy = y2-y1;
    const len = Math.hypot(dx,dy) || 1;
    for (let i=1;i<points.length-1;i++){
      const [px,py] = points[i];
      const d = Math.abs(dy*px - dx*py + x2*y1 - y2*x1) / len;
      if (d > maxDist) { maxDist = d; idx = i; }
    }
    if (maxDist > epsilon){
      const left = rdp(points.slice(0, idx+1), epsilon);
      const right = rdp(points.slice(idx), epsilon);
      return left.slice(0,-1).concat(right);
    }
    return [points[0], points[points.length-1]];
  }
  const RDP_EPSILON = 1.2; // pixel units in the unsheared lattice grid
  function traceBoundary(){
    const c = COL_SIZE/2;
    const raw = marchingSquaresBoundary();
    const simplified = simplifyClosedPolygon(raw, RDP_EPSILON);
    // Forward shear into equal-angle physical space -- see engine.js's
    // FS_VIEW_SRC comment for the derivation of this exact formula.
    return simplified.map(([x,y]) => {
      const lx = x-c, ly = y-c;
      return [lx - 0.5*ly, ly*Math.sqrt(3)/2];
    });
  }

  // numFrames only sizes chunkSteps (how often we capture) against the
  // table's initial estimate -- a real completion can run past that
  // estimate (see the extension logic below), so the actual frame count
  // is whatever falls out of stepCap / chunkSteps, not exactly numFrames.
  chunkSteps = Math.max(200, Math.round(targetRadiusFor(beta) / numFrames));
  const frames = []; // {points, attached}
  readback();
  frames.push({points: traceBoundary(), attached: 1});
  let stepCap = targetRadiusFor(beta);
  const HARD_CAP = stepCap * 3;
  while (true){
    const thisChunk = Math.min(chunkSteps, stepCap - stepCount);
    if (thisChunk > 0) stepFixed(thisChunk);
    readback();
    let attached = 0;
    for (let i=0;i<COL_SIZE*COL_SIZE;i++) if (pixels[i*4]>0.5) attached++;
    frames.push({points: traceBoundary(), attached});
    if (onProgress) onProgress(Math.min(1, stepCount/stepCap));
    if (stepCount >= stepCap) {
      const radiusFrac = boundingRadiusFrac();
      if (radiusFrac < targetRadiusFrac && stepCap < HARD_CAP) {
        stepCap = Math.min(HARD_CAP, Math.round(stepCap*1.5));
        continue;
      }
      break;
    }
    await new Promise(r => setTimeout(r, 0)); // yield between chunks
  }

  // Second pass: distribute a FIXED total height budget (aspectRatio *
  // the crystal's own target radius, so it scales with the crystal
  // itself rather than with step count or frame count) across intervals,
  // weighted by how stalled xy growth was that interval -- using the
  // GLOBAL max xy-growth-rate seen across the whole run (not a running
  // max, which would under-credit early frames before the true peak rate
  // had been observed yet). See this file's top comment.
  let maxDelta = 1;
  for (let i=1;i<frames.length;i++) maxDelta = Math.max(maxDelta, frames[i].attached - frames[i-1].attached);
  const stallWeights = [0];
  let totalWeight = 0;
  for (let i=1;i<frames.length;i++){
    const delta = Math.max(0, frames[i].attached - frames[i-1].attached);
    const xyActivity = delta / maxDelta; // 0 = fully stalled this interval, 1 = fastest xy growth seen
    const w = 1 - xyActivity;
    stallWeights.push(w);
    totalWeight += w;
  }
  const targetRadiusLatticeUnits = targetRadiusFrac * (COL_SIZE/2);
  const totalHeightBudget = aspectRatio * targetRadiusLatticeUnits * hexagonCompatibility(beta);
  let height = 0;
  frames[0].height = 0;
  for (let i=1;i<frames.length;i++){
    height += totalWeight > 0 ? (stallWeights[i]/totalWeight) * totalHeightBudget : 0;
    frames[i].height = height;
  }

  gl.deleteTexture(bufs[0].tex); gl.deleteTexture(bufs[1].tex);
  gl.deleteFramebuffer(bufs[0].fbo); gl.deleteFramebuffer(bufs[1].fbo);
  return frames.map(f => ({points: f.points, height: f.height}));
}

// Real snow crystals never come as extruded dendrites -- see this file's
// top comment. Tapers the total height budget to 0 through the fern/
// stellar-dendrite bands (beta <= 1.60, matching engine.js's own
// morphologyLabel thresholds exactly) and to full strength by the start
// of the simple-plate band (beta >= 2.00); ramps through the sectored-
// plate band in between, where real crystals do start showing more
// hexagonal, less-fingered structure. Smoothstep, not linear, so the
// transition doesn't have a visible kink at either edge. The two edges
// are chosen to line up with morphologyLabel's own bands, not
// independently measured -- same "designed, not derived" caveat as the
// rest of this file.
function hexagonCompatibility(beta){
  const e0 = 1.60, e1 = 2.00;
  const t = Math.max(0, Math.min(1, (beta-e0)/(e1-e0)));
  return t*t*(3-2*t);
}

// Local copy of engine.js's targetStepsFor/BETA_STEP_TABLE (paces the
// capture interval and initial step estimate only -- see growColumnCrystal
// above for the actual TARGET_RADIUS_FRAC-driven completion check).
const COL_BETA_STEP_TABLE = [
  [1.10, 1000], [1.15, 1300], [1.20, 1500], [1.25, 1700],
  [1.30, 2200], [1.35, 2800], [1.40, 3600], [1.50, 7200],
  [1.60, 12900], [1.75, 24300], [1.90, 34800], [2.05, 46000],
  [2.20, 58000], [2.35, 71000], [2.50, 85000], [2.60, 94000],
];
function targetRadiusFor(beta){
  const t = COL_BETA_STEP_TABLE;
  const b = Math.max(t[0][0], Math.min(t[t.length-1][0], beta));
  for (let i=0;i<t.length-1;i++){
    const [b0,s0]=t[i], [b1,s1]=t[i+1];
    if (b<=b1) return Math.round(s0 + (s1-s0)*Math.max(0,Math.min(1,(b-b0)/(b1-b0))));
  }
  return t[t.length-1][1];
}
