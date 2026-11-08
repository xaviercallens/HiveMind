# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Self-contained WebGL renderer for the HUD's Geo-Ring topology panel (PR #24).

The panel is plain HTML + inline JS: raw WebGL (WebGL2 when available, WebGL1
otherwise), no CDN, no external requests. It draws a lat/long wireframe globe,
one glowing point per peer from ``/api/georing-topology`` (colored by continent),
and arcs between peers and their continental bridge and between bridges.

Season 5 (`HM-S5-04`) adds the **live animated Guild Raid Arcs** of the
cross-guild Co-Op tournament: the very same module now also exposes the
server-side geometry generator (:func:`build_guild_raid_arc_geometry`) that
produces the exact vertex buffers the WebGL renderer uploads for the
``GUILD-01`` / ``GUILD-02`` / ``GUILD-03`` raid arcs served by
``/api/guild-tournament``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

GEORING_PANEL_HTML = r"""
    <div class="card" id="webgl-georing-map" style="grid-column: 1 / -1;" data-nodes="0" data-arcs="0" data-frames="0" data-errors="0" data-webgl="pending" data-yaw="0" data-raid-arcs="0">
      <h2>SEASON 3 INTERACTIVE WEBGL 3D GEO-RING TOPOLOGY MAP &amp; MISSIONS (9-12)</h2>
      <div id="georing-stage" style="position:relative;">
        <canvas id="webgl-georing-canvas" width="920" height="360" style="display:block; width:100%; height:360px; background:#050811; border:1px solid #00f0ff; border-radius:6px; cursor:grab; touch-action:none;"></canvas>
        <div id="georing-label" style="display:none; position:absolute; pointer-events:none; background:rgba(5,8,17,0.92); border:1px solid #00f0ff; border-radius:4px; padding:4px 8px; font-size:0.78rem; color:#e2e8f0; white-space:pre; z-index:2;"></div>
        <div id="georing-fallback" style="display:none; padding:40px 16px; text-align:center; border:1px solid #ff3860; border-radius:6px; color:#ff9eb1; background:#050811;"></div>
      </div>
      <div class="terminal" id="georing-terminal">[WEBGL 3D GEO-RING] Loading /api/georing-topology ... Drag to orbit, wheel to zoom, hover a peer for details.</div>
      <div style="font-size:0.75rem; margin-top:4px;">
        <span style="color:#38bdf8;">&#9679; EU</span> &nbsp; <span style="color:#fb923c;">&#9679; US</span> &nbsp; <span style="color:#4ade80;">&#9679; APAC</span>
      </div>
    </div>
    <script>
""" + r"""
(function () {
  'use strict';
  var panel = document.getElementById('webgl-georing-map');
  var canvas = document.getElementById('webgl-georing-canvas');
  var term = document.getElementById('georing-terminal');
  var labelEl = document.getElementById('georing-label');
  var fallback = document.getElementById('georing-fallback');
  if (!panel || !canvas) return;
  if (/[?&]georing=solo\b/.test(location.search)) {
    for (var node = panel; node && node !== document.body; node = node.parentElement) {
      Array.prototype.forEach.call(node.parentElement.children, function (sib) {
        if (sib !== node && sib.tagName !== 'SCRIPT') sib.style.display = 'none';
      });
    }
  }
  var stats = { nodes: 0, arcs: 0, frames: 0, errors: 0, webgl: 'pending', raidArcs: 0 };
  var yaw = 0.6, pitch = 0.35, dist = 3.4;
  window.__georingStats = stats;
  var qy = /[?&]georing_yaw=(-?[0-9.]+)/.exec(location.search);
  if (qy) yaw = parseFloat(qy[1]);
  function publish() {
    panel.setAttribute('data-nodes', stats.nodes);
    panel.setAttribute('data-arcs', stats.arcs);
    panel.setAttribute('data-frames', stats.frames);
    panel.setAttribute('data-errors', stats.errors);
    panel.setAttribute('data-webgl', stats.webgl);
    panel.setAttribute('data-yaw', yaw.toFixed(4));
    panel.setAttribute('data-raid-arcs', stats.raidArcs);
    var r = canvas.getBoundingClientRect();
    panel.setAttribute('data-canvas-rect', [r.left + window.scrollX, r.top + window.scrollY, r.width, r.height].map(Math.round).join(','));
  }
  function fail(msg) {
    stats.errors++;
    term.textContent = '[GEORING ERROR] ' + msg;
    publish();
  }

  var CONT = {
    EU: { rgb: [0.22, 0.74, 0.97] },
    US: { rgb: [0.98, 0.57, 0.24] },
    APAC: { rgb: [0.29, 0.87, 0.50] }
  };
  function continentOf(n) {
    var r = String(n.region || n.node_id || '').toUpperCase();
    if (r.indexOf('EU') === 0) return 'EU';
    if (r.indexOf('US') === 0 || r.indexOf('NA') === 0) return 'US';
    if (r.indexOf('APAC') === 0 || r.indexOf('AP-') === 0) return 'APAC';
    var lon = Number(n.lon);
    if (lon < -30) return 'US';
    if (lon > 60) return 'APAC';
    return 'EU';
  }

  var m4 = {
    ident: function () { return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]; },
    mul: function (a, b) {
      var o = new Array(16);
      for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
        var s = 0;
        for (var k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
        o[c * 4 + r] = s;
      }
      return o;
    },
    persp: function (fovy, asp, n, f) {
      var t = 1 / Math.tan(fovy / 2);
      return [t / asp,0,0,0, 0,t,0,0, 0,0,(f + n) / (n - f),-1, 0,0,2 * f * n / (n - f),0];
    },
    trans: function (x, y, z) { var m = m4.ident(); m[12] = x; m[13] = y; m[14] = z; return m; },
    rotY: function (a) { var c = Math.cos(a), s = Math.sin(a); return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]; },
    rotX: function (a) { var c = Math.cos(a), s = Math.sin(a); return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]; },
    apply: function (m, v) {
      var x = v[0], y = v[1], z = v[2];
      return [
        m[0]*x + m[4]*y + m[8]*z + m[12],
        m[1]*x + m[5]*y + m[9]*z + m[13],
        m[2]*x + m[6]*y + m[10]*z + m[14],
        m[3]*x + m[7]*y + m[11]*z + m[15]
      ];
    }
  };

  function llToXyz(lat, lon, r) {
    var la = lat * Math.PI / 180, lo = lon * Math.PI / 180;
    return [r * Math.cos(la) * Math.sin(lo), r * Math.sin(la), r * Math.cos(la) * Math.cos(lo)];
  }
  function arcPoints(a, b, lift, steps) {
    var da = Math.hypot(a[0], a[1], a[2]), db = Math.hypot(b[0], b[1], b[2]);
    var ua = [a[0]/da, a[1]/da, a[2]/da], ub = [b[0]/db, b[1]/db, b[2]/db];
    var dot = Math.max(-1, Math.min(1, ua[0]*ub[0] + ua[1]*ub[1] + ua[2]*ub[2]));
    var om = Math.acos(dot), so = Math.sin(om) || 1e-6, out = [];
    for (var i = 0; i <= steps; i++) {
      var t = i / steps, w1 = Math.sin((1 - t) * om) / so, w2 = Math.sin(t * om) / so;
      var rr = 1.0 + lift * Math.sin(Math.PI * t) * Math.min(1, om / 1.5);
      out.push([(ua[0]*w1 + ub[0]*w2) * rr, (ua[1]*w1 + ub[1]*w2) * rr, (ua[2]*w1 + ub[2]*w2) * rr]);
    }
    return out;
  }

  var gl = null, isGL2 = false, prog = null, loc = {};
  var dragging = false, moved = false, lastX = 0, lastY = 0;
  var hover = -1, pinned = -1, mouse = null, peers = [], arcs = [];
  var batches = { grid: null, arcs: null, pts: null };
  var lastT = 0, telemetry = null;

  // ---- Season 5 (HM-S5-04): live animated cross-guild raid arcs ----
  var GUILD_COLORS = {
    'GUILD-01': [1.00, 0.30, 0.52],
    'GUILD-02': [0.30, 0.85, 1.00],
    'GUILD-03': [0.45, 1.00, 0.45]
  };
  var raidArcs = [], raidNodes = [], tournament = null;
  var raidBatch = null, raidHeadBatch = null, raidNodeBatch = null;

  function guildColor(gid) {
    if (GUILD_COLORS[gid]) return GUILD_COLORS[gid];
    var h = 0, s = String(gid || '');
    for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
    return [
      0.55 + 0.45 * Math.abs(Math.sin(h * Math.PI / 180)),
      0.55 + 0.45 * Math.abs(Math.sin((h + 120) * Math.PI / 180)),
      0.55 + 0.45 * Math.abs(Math.sin((h + 240) * Math.PI / 180))
    ];
  }

  function rebuildRaidArcs(t) {
    raidArcs = [];
    raidNodes = [];
    if (!t) { stats.raidArcs = 0; return; }
    (t.guilds || []).forEach(function (g) {
      if (!isFinite(Number(g.lat)) || !isFinite(Number(g.lon))) return;
      raidNodes.push({
        id: g.guild_id, guild_id: g.guild_id, rank: g.rank,
        pos: llToXyz(Number(g.lat), Number(g.lon), 1.0), color: guildColor(g.guild_id)
      });
    });
    (t.raid_arcs || []).forEach(function (a) {
      var pts = null;
      if (a.points && a.points.length > 1) {
        pts = a.points.map(function (p) { return [Number(p[0]), Number(p[1]), Number(p[2])]; });
      } else if (isFinite(Number(a.source_lat)) && isFinite(Number(a.target_lat))) {
        pts = arcPoints(
          llToXyz(Number(a.source_lat), Number(a.source_lon), 1.0),
          llToXyz(Number(a.target_lat), Number(a.target_lon), 1.0),
          isFinite(Number(a.lift)) ? Number(a.lift) : 0.32,
          isFinite(Number(a.steps)) ? Number(a.steps) : 40);
      }
      if (!pts) return;
      raidArcs.push({
        id: a.arc_id, pts: pts, color: a.color || guildColor(a.source_guild),
        intensity: isFinite(Number(a.intensity)) ? Number(a.intensity) : 0.5,
        phase: isFinite(Number(a.phase)) ? Number(a.phase) : 0,
        source: a.source_guild, target: a.target_guild
      });
    });
    stats.raidArcs = raidArcs.length;
    buildRaidNodeBatch();
  }

  function buildRaidNodeBatch() {
    if (!gl) return;
    var d = [];
    raidNodes.forEach(function (n) { pushV(d, n.pos, n.color, 0.9, 40); });
    if (!raidNodeBatch) raidNodeBatch = { buf: gl.createBuffer(), count: 0 };
    gl.bindBuffer(gl.ARRAY_BUFFER, raidNodeBatch.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d), gl.STATIC_DRAW);
    raidNodeBatch.count = d.length / 8;
  }

  function updateRaidBatch(now) {
    if (!gl) return;
    var d = [], ranges = [], heads = [];
    raidArcs.forEach(function (a) {
      var pulse = 0.35 + 0.65 * Math.abs(Math.sin(now * 1.7 + a.phase));
      var alpha = (0.30 + 0.70 * a.intensity) * pulse;
      var start = d.length / 8;
      a.pts.forEach(function (p) { pushV(d, p, a.color, alpha, 1); });
      ranges.push([start, a.pts.length]);
      var u = (now * 0.22 + a.phase * 0.37) % 1;
      var idx = u * (a.pts.length - 1);
      var i0 = Math.floor(idx), i1 = Math.min(a.pts.length - 1, i0 + 1), f = idx - i0;
      var p0 = a.pts[i0], p1 = a.pts[i1];
      var hp = [p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f, p0[2] + (p1[2] - p0[2]) * f];
      pushV(heads, hp, [1, 1, 1], 0.95, 11);
      pushV(heads, hp, a.color, 0.60, 20);
    });
    if (!raidBatch) raidBatch = { buf: gl.createBuffer(), ranges: [] };
    gl.bindBuffer(gl.ARRAY_BUFFER, raidBatch.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(d), gl.DYNAMIC_DRAW);
    raidBatch.ranges = ranges;
    if (!raidHeadBatch) raidHeadBatch = { buf: gl.createBuffer(), count: 0 };
    gl.bindBuffer(gl.ARRAY_BUFFER, raidHeadBatch.buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(heads), gl.DYNAMIC_DRAW);
    raidHeadBatch.count = heads.length / 8;
  }

  function initGL() {
    var opts = { antialias: true, alpha: false };
    gl = canvas.getContext('webgl2', opts);
    isGL2 = !!gl;
    if (!gl) gl = canvas.getContext('webgl', opts) || canvas.getContext('experimental-webgl', opts);
    if (!gl) return false;
    stats.webgl = isGL2 ? 'webgl2' : 'webgl1';
    var vs = 'attribute vec3 aPos; attribute vec4 aCol; attribute float aSize;' +
      'uniform mat4 uMVP; uniform mat4 uModel; varying vec4 vCol;' +
      'void main(){ vec4 w = uModel * vec4(aPos,1.0); float front = w.z > -0.15 ? 1.0 : 0.22;' +
      'vCol = vec4(aCol.rgb, aCol.a * front); gl_Position = uMVP * vec4(aPos,1.0); gl_PointSize = aSize * front; }';
    // uRing > 0 draws a hollow ring, so a marker sharing a peer's position leaves its continent color visible.
    var fs = 'precision mediump float; varying vec4 vCol; uniform float uRound; uniform float uRing;' +
      'void main(){ if (uRound > 0.5) { float d = distance(gl_PointCoord, vec2(0.5)); if (d > 0.5 || d < uRing) discard;' +
      'float a = uRing > 0.0 ? 1.0 - smoothstep(0.44, 0.5, d) : smoothstep(0.5, 0.05, d);' +
      'gl_FragColor = vec4(vCol.rgb, vCol.a * a); } else gl_FragColor = vCol; }';
    function sh(type, src) {
      var s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('shader: ' + gl.getShaderInfoLog(s));
      return s;
    }
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
    loc.aPos = gl.getAttribLocation(prog, 'aPos');
    loc.aCol = gl.getAttribLocation(prog, 'aCol');
    loc.aSize = gl.getAttribLocation(prog, 'aSize');
    loc.uMVP = gl.getUniformLocation(prog, 'uMVP');
    loc.uModel = gl.getUniformLocation(prog, 'uModel');
    loc.uRound = gl.getUniformLocation(prog, 'uRound');
    loc.uRing = gl.getUniformLocation(prog, 'uRing');
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    return true;
  }

  function makeBatch(data, ranges, old) {
    if (old && old.buf) gl.deleteBuffer(old.buf);
    var buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
    return { buf: buf, ranges: ranges };
  }
  function pushV(d, p, c, a, s) { d.push(p[0], p[1], p[2], c[0], c[1], c[2], a, s); }

  function buildGrid() {
    var d = [], ranges = [], col = [0.10, 0.45, 0.62], i, j, start;
    for (var lat = -60; lat <= 60; lat += 30) {
      start = d.length / 8;
      for (i = 0; i <= 72; i++) pushV(d, llToXyz(lat, i * 5 - 180, 1), lat === 0 ? [0.2, 0.7, 0.9] : col, 0.55, 1);
      ranges.push([start, 73]);
    }
    for (var lon = -180; lon < 180; lon += 30) {
      start = d.length / 8;
      for (j = 0; j <= 36; j++) pushV(d, llToXyz(j * 5 - 90, lon, 1), col, 0.55, 1);
      ranges.push([start, 37]);
    }
    return makeBatch(d, ranges, null);
  }

  function rebuild(topo) {
    var nodes = (topo && (topo.nodes || topo.peers)) || [];
    peers = nodes.filter(function (n) { return isFinite(Number(n.lat)) && isFinite(Number(n.lon)); }).map(function (n) {
      var cont = continentOf(n);
      return { id: n.node_id || n.id || n.region, region: n.region, cont: cont, lat: Number(n.lat), lon: Number(n.lon),
               rtt: n.rtt_ms, nat: n.nat_mode, pos: llToXyz(Number(n.lat), Number(n.lon), 1.0), bridge: false };
    });
    var bridgeOf = {};
    peers.forEach(function (p) {
      if (!bridgeOf[p.cont] || p.rtt < bridgeOf[p.cont].rtt) bridgeOf[p.cont] = p;
    });
    Object.keys(bridgeOf).forEach(function (k) { bridgeOf[k].bridge = true; });
    arcs = [];
    peers.forEach(function (p) {
      var b = bridgeOf[p.cont];
      if (b && b !== p) arcs.push({ a: p, b: b, kind: 'peer' });
    });
    var bs = Object.keys(bridgeOf).map(function (k) { return bridgeOf[k]; });
    for (var i = 0; i < bs.length; i++) for (var j = i + 1; j < bs.length; j++) arcs.push({ a: bs[i], b: bs[j], kind: 'bridge' });
    stats.nodes = peers.length; stats.arcs = arcs.length;
    if (!gl) return;
    var d = [], ranges = [];
    arcs.forEach(function (e) {
      var c = e.kind === 'bridge' ? [1.0, 0.85, 0.35] : CONT[e.a.cont].rgb;
      var pts = arcPoints(e.a.pos, e.b.pos, e.kind === 'bridge' ? 0.32 : 0.14, 40);
      var start = d.length / 8;
      pts.forEach(function (p) { pushV(d, p, c, e.kind === 'bridge' ? 0.95 : 0.7, 1); });
      ranges.push([start, pts.length]);
    });
    batches.arcs = makeBatch(d, ranges, batches.arcs);
    var pd = [];
    peers.forEach(function (p) {
      var c = CONT[p.cont].rgb;
      pushV(pd, p.pos, c, 0.25, p.bridge ? 34 : 26);
      pushV(pd, p.pos, c, 1.0, p.bridge ? 16 : 11);
      // Keep the hot core in the continent hue so the US/EU/APAC colors survive
      // compositing; a white core washed every marker out to white.
      pushV(pd, p.pos, c, 0.9, 5);
    });
    batches.pts = makeBatch(pd, [[0, peers.length * 3]], batches.pts);
  }

  function bind(batch) {
    gl.bindBuffer(gl.ARRAY_BUFFER, batch.buf);
    gl.enableVertexAttribArray(loc.aPos); gl.vertexAttribPointer(loc.aPos, 3, gl.FLOAT, false, 32, 0);
    gl.enableVertexAttribArray(loc.aCol); gl.vertexAttribPointer(loc.aCol, 4, gl.FLOAT, false, 32, 12);
    gl.enableVertexAttribArray(loc.aSize); gl.vertexAttribPointer(loc.aSize, 1, gl.FLOAT, false, 32, 28);
  }

  var cssW = 1, cssH = 1;
  function resize() {
    var dpr = window.devicePixelRatio || 1;
    var r = canvas.getBoundingClientRect();
    cssW = Math.max(1, r.width); cssH = Math.max(1, r.height);
    var w = Math.max(1, Math.round(cssW * dpr)), h = Math.max(1, Math.round(cssH * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function matrices() {
    var model = m4.mul(m4.rotX(pitch), m4.rotY(yaw));
    var view = m4.trans(0, 0, -dist);
    var proj = m4.persp(0.8, canvas.width / canvas.height, 0.1, 20);
    return { model: model, mvp: m4.mul(proj, m4.mul(view, model)) };
  }
  function project(p, M) {
    var c = m4.apply(M.mvp, p);
    var w = m4.apply(M.model, p);
    return { x: (c[0] / c[3] * 0.5 + 0.5) * cssW, y: (1 - (c[1] / c[3] * 0.5 + 0.5)) * cssH, front: w[2] > -0.15 };
  }

  function frame(t) {
    var dt = lastT ? Math.min(0.1, (t - lastT) / 1000) : 0;
    lastT = t;
    if (!dragging) yaw += dt * 0.18;
    try {
      resize();
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.02, 0.03, 0.07, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(prog);
      var M = matrices();
      gl.uniformMatrix4fv(loc.uMVP, false, new Float32Array(M.mvp));
      gl.uniformMatrix4fv(loc.uModel, false, new Float32Array(M.model));
      gl.uniform1f(loc.uRound, 0);
      [batches.grid, batches.arcs].forEach(function (b) {
        if (!b) return;
        bind(b);
        b.ranges.forEach(function (r) { gl.drawArrays(gl.LINE_STRIP, r[0], r[1]); });
      });
      if (raidArcs.length) {
        updateRaidBatch(t / 1000);
        if (raidBatch) {
          gl.uniform1f(loc.uRound, 0);
          bind(raidBatch);
          raidBatch.ranges.forEach(function (r) { gl.drawArrays(gl.LINE_STRIP, r[0], r[1]); });
        }
        if (raidNodeBatch && raidNodeBatch.count) {
          gl.uniform1f(loc.uRound, 1);
          gl.uniform1f(loc.uRing, 0.43);
          bind(raidNodeBatch);
          gl.drawArrays(gl.POINTS, 0, raidNodeBatch.count);
          gl.uniform1f(loc.uRing, 0);
        }
        if (raidHeadBatch && raidHeadBatch.count) {
          gl.uniform1f(loc.uRound, 1);
          bind(raidHeadBatch);
          gl.drawArrays(gl.POINTS, 0, raidHeadBatch.count);
        }
      }
      // Peer markers are drawn last so a guild node sharing the same home node
      // cannot paint over the continent color.
      if (batches.pts && peers.length) {
        gl.uniform1f(loc.uRound, 1);
        bind(batches.pts);
        // Standard alpha compositing keeps the continent color stable; additive
        // blending saturated the glow + core layers to white.
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArrays(gl.POINTS, 0, peers.length * 3);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      }
      updateLabel(M);
      stats.frames++;
    } catch (e) { fail('render: ' + e.message); return; }
    publish();
    requestAnimationFrame(frame);
  }

  function pick(M) {
    if (!mouse) return -1;
    var best = -1, bd = 18 * 18;
    peers.forEach(function (p, i) {
      var s = project(p.pos, M);
      if (!s.front) return;
      var d2 = (s.x - mouse.x) * (s.x - mouse.x) + (s.y - mouse.y) * (s.y - mouse.y);
      if (d2 < bd) { bd = d2; best = i; }
    });
    return best;
  }
  function updateLabel(M) {
    hover = dragging ? -1 : pick(M);
    var i = hover >= 0 ? hover : pinned;
    if (i < 0 || !peers[i]) { labelEl.style.display = 'none'; return; }
    var p = peers[i], s = project(p.pos, M);
    labelEl.textContent = p.id + (p.bridge ? '  [bridge]' : '') + '\n' + p.cont + ' | ' + p.region +
      '\nRTT ' + p.rtt + ' ms | NAT ' + p.nat;
    labelEl.style.display = 'block';
    labelEl.style.left = Math.min(s.x + 14, Math.max(0, cssW - 230)) + 'px';
    labelEl.style.top = Math.max(0, s.y - 20) + 'px';
  }

  function events() {
    function rel(e) { var r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }
    canvas.addEventListener('pointerdown', function (e) {
      dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY;
      canvas.style.cursor = 'grabbing';
      try { canvas.setPointerCapture(e.pointerId); } catch (x) {}
    });
    canvas.addEventListener('pointermove', function (e) {
      mouse = rel(e);
      if (!dragging) return;
      var dx = e.clientX - lastX, dy = e.clientY - lastY;
      if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
      yaw += dx * 0.008; pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.008));
      lastX = e.clientX; lastY = e.clientY;
    });
    canvas.addEventListener('pointerup', function (e) {
      if (dragging && !moved) { mouse = rel(e); pinned = pick(matrices()); }
      dragging = false; canvas.style.cursor = 'grab';
    });
    canvas.addEventListener('pointercancel', function () { dragging = false; });
    canvas.addEventListener('pointerleave', function () { mouse = null; });
    canvas.addEventListener('wheel', function (e) {
      e.preventDefault();
      dist = Math.max(1.8, Math.min(8, dist * Math.exp(e.deltaY * 0.0012)));
    }, { passive: false });
    window.addEventListener('resize', resize);
  }

  function status() {
    var s = '[WEBGL 3D GEO-RING ' + stats.webgl.toUpperCase() + '] ' + stats.nodes + ' peers / ' + stats.arcs +
      ' arcs. Drag to orbit, wheel to zoom, hover or click a peer.';
    if (telemetry) s += ' | live telemetry frame #' + telemetry.seq;
    if (stats.raidArcs) s += ' | ' + stats.raidArcs + ' live guild raid arc(s)';
    term.textContent = s;
  }
  var offline = location.protocol === 'file:';
  function loadSnapshot() {
    var el = document.getElementById('georing-topology-snapshot');
    if (!el) return false;
    try { rebuild(JSON.parse(el.textContent)); } catch (e) { return false; }
    status();
    term.textContent += ' | offline snapshot (no /api/georing-topology)';
    publish();
    return true;
  }
  function load() {
    if (offline) {
      if (!loadSnapshot()) fail('offline HUD has no embedded topology snapshot');
      return;
    }
    fetch('/api/georing-topology', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (topo) { rebuild(topo); status(); publish(); })
      .catch(function (e) {
        if (!peers.length) loadSnapshot();
        fail('topology fetch failed: ' + e.message);
      });
  }
  function loadTournament() {
    fetch('/api/guild-tournament', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (t) {
      tournament = t;
      rebuildRaidArcs(t);
      status();
      publish();
    }).catch(function (e) { /* offline HUD: keep the last raid-arc snapshot */ });
  }
  function ws() {
    if (!window.WebSocket || location.protocol === 'file:') return;
    try {
      var s = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/swarm-telemetry');
      s.onmessage = function (m) { try { telemetry = JSON.parse(m.data); } catch (e) {} };
      s.onclose = function () { setTimeout(ws, 5000); };
    } catch (e) {}
  }

  try {
    if (!initGL()) {
      stats.webgl = 'unavailable';
      canvas.style.display = 'none';
      fallback.style.display = 'block';
      fallback.textContent = '[WEBGL UNAVAILABLE] This browser or GPU could not create a WebGL context, so the 3D Geo-Ring globe cannot be drawn. Peer data is still available at /api/georing-topology.';
      term.textContent = fallback.textContent;
      publish();
      return;
    }
    batches.grid = buildGrid();
    events(); resize();
    load(); loadTournament();
    if (!offline) { setInterval(load, 2000); setInterval(loadTournament, 2000); ws(); }
    requestAnimationFrame(frame);
  } catch (e) {
    stats.webgl = 'error';
    canvas.style.display = 'none';
    fallback.style.display = 'block';
    fallback.textContent = '[WEBGL ERROR] ' + e.message;
    fail(e.message);
  }
})();
</script>
"""


# ---------------------------------------------------------------------------
# Season 5 (`HM-S5-04`): server-side generation of the 3D WebGL raid-arc geometry
# ---------------------------------------------------------------------------
# The HUD's WebGL renderer can draw an arc from any two globe points, but the
# cross-guild Co-Op tournament (`/api/guild-tournament`) must publish the very
# same geometry so that the HUD, the REST API and the unit tests all agree on
# the raid-arc vertex buffers.  The helpers below are a faithful Python port of
# the renderer's `llToXyz()` / `arcPoints()` JavaScript functions.

GUILD_RAID_ARC_STEPS = 40
GUILD_RAID_ARC_LIFT = 0.32

GUILD_RAID_ARC_PALETTE: Dict[str, Tuple[float, float, float]] = {
    "GUILD-01": (1.00, 0.30, 0.52),
    "GUILD-02": (0.30, 0.85, 1.00),
    "GUILD-03": (0.45, 1.00, 0.45),
}


def guild_raid_arc_color(guild_id: str) -> List[float]:
    """Return the deterministic RGB color used to draw a guild's raid arcs."""
    if not isinstance(guild_id, str) or not guild_id.strip():
        raise ValueError("guild_id must be a non-empty string")
    if guild_id in GUILD_RAID_ARC_PALETTE:
        return [round(c, 4) for c in GUILD_RAID_ARC_PALETTE[guild_id]]
    h = 0
    for ch in str(guild_id):
        h = (h * 31 + ord(ch)) % 360
    return [
        round(0.55 + 0.45 * abs(math.sin(math.radians(h))), 4),
        round(0.55 + 0.45 * abs(math.sin(math.radians(h + 120))), 4),
        round(0.55 + 0.45 * abs(math.sin(math.radians(h + 240))), 4),
    ]


def lat_lon_to_xyz(lat: float, lon: float, radius: float = 1.0) -> List[float]:
    """Convert lat/lon (degrees) into the WebGL globe's Cartesian coordinates."""
    if (
        isinstance(lat, bool)
        or isinstance(lon, bool)
        or isinstance(radius, bool)
        or not isinstance(lat, (int, float))
        or not isinstance(lon, (int, float))
        or not isinstance(radius, (int, float))
        or not math.isfinite(float(lat))
        or not math.isfinite(float(lon))
        or not math.isfinite(float(radius))
    ):
        raise ValueError("lat, lon, and radius must be finite floats")
    if float(radius) <= 0.0:
        raise ValueError("radius must be > 0")
    if abs(float(lat)) > 90.0 or abs(float(lon)) > 180.0:
        raise ValueError("lat must be in [-90, 90] and lon in [-180, 180]")
    la = math.radians(float(lat))
    lo = math.radians(float(lon))
    return [
        float(radius) * math.cos(la) * math.sin(lo),
        float(radius) * math.sin(la),
        float(radius) * math.cos(la) * math.cos(lo),
    ]


def great_circle_arc_points(
    a: Sequence[float],
    b: Sequence[float],
    lift: float = GUILD_RAID_ARC_LIFT,
    steps: int = GUILD_RAID_ARC_STEPS,
) -> List[List[float]]:
    """Slerp a lifted great-circle arc between two globe points.

    Mirrors the renderer's `arcPoints()` JavaScript helper so that the
    server-published geometry is byte-for-byte the geometry drawn on the HUD.
    """
    if isinstance(lift, bool) or not isinstance(lift, (int, float)) or not math.isfinite(float(lift)) or float(lift) < 0.0:
        raise ValueError("lift must be a finite non-negative float")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("steps must be an integer")
    for pt_name, pt in (("a", a), ("b", b)):
        if isinstance(pt, (str, bytes)) or not isinstance(pt, Sequence) or len(pt) != 3:
            raise ValueError(f"{pt_name} must be a 3-element coordinate sequence")
        if any(isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(float(c)) for c in pt):
            raise ValueError(f"{pt_name} coordinates must be finite floats")
    if steps < 1:
        steps = 1
    da = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) or 1.0
    db = math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2) or 1.0
    ua = [a[0] / da, a[1] / da, a[2] / da]
    ub = [b[0] / db, b[1] / db, b[2] / db]
    dot = max(-1.0, min(1.0, ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2]))
    omega = math.acos(dot)
    so = math.sin(omega) or 1e-6
    out: List[List[float]] = []
    for i in range(steps + 1):
        t = i / steps
        w1 = math.sin((1 - t) * omega) / so
        w2 = math.sin(t * omega) / so
        rr = 1.0 + lift * math.sin(math.pi * t) * min(1.0, omega / 1.5)
        out.append(
            [
                (ua[0] * w1 + ub[0] * w2) * rr,
                (ua[1] * w1 + ub[1] * w2) * rr,
                (ua[2] * w1 + ub[2] * w2) * rr,
            ]
        )
    return out


def build_guild_raid_arc_geometry(
    arcs: Iterable[Dict[str, Any]],
    steps: int = GUILD_RAID_ARC_STEPS,
) -> List[Dict[str, Any]]:
    """Generate the WebGL raid-arc vertex geometry for the 3D Geo-Ring globe.

    Every arc spec must expose `source_lat`/`source_lon` and `target_lat`/
    `target_lon` (a pre-computed `points` list is accepted and re-used as-is).
    The returned dictionaries carry the exact vertex buffer the HUD uploads
    plus the derived metrics (polyline length, peak radius) used by the
    tournament scorer and asserted by the unit tests.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    geometry: List[Dict[str, Any]] = []
    for arc in arcs:
        source_lat = float(arc.get("source_lat", arc.get("lat", 0.0)))
        source_lon = float(arc.get("source_lon", arc.get("lon", 0.0)))
        target_lat = float(arc.get("target_lat", arc.get("lat", 0.0)))
        target_lon = float(arc.get("target_lon", arc.get("lon", 0.0)))
        lift = float(arc.get("lift", GUILD_RAID_ARC_LIFT))
        raw_points = arc.get("points")
        if raw_points:
            points = [[float(p[0]), float(p[1]), float(p[2])] for p in raw_points]
        else:
            points = great_circle_arc_points(
                lat_lon_to_xyz(source_lat, source_lon),
                lat_lon_to_xyz(target_lat, target_lon),
                lift=lift,
                steps=steps,
            )
        arc_length = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            dz = points[i][2] - points[i - 1][2]
            arc_length += math.sqrt(dx * dx + dy * dy + dz * dz)
        max_radius = max(
            (math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in points),
            default=0.0,
        )
        geometry.append(
            {
                "arc_id": str(arc.get("arc_id", "")),
                "source_guild": str(arc.get("source_guild", "")),
                "target_guild": str(arc.get("target_guild", "")),
                "kind": str(arc.get("kind", "guild_raid")),
                "lift": round(lift, 4),
                "steps": len(points) - 1,
                "source_xyz": [round(v, 6) for v in lat_lon_to_xyz(source_lat, source_lon)],
                "target_xyz": [round(v, 6) for v in lat_lon_to_xyz(target_lat, target_lon)],
                "points": [[round(v, 6) for v in p] for p in points],
                "arc_length": round(arc_length, 6),
                "max_radius": round(max_radius, 6),
                "color": arc.get("color") or guild_raid_arc_color(str(arc.get("source_guild", ""))),
                "intensity": round(float(arc.get("intensity", 0.5)), 4),
                "phase": round(float(arc.get("phase", 0.0)), 4),
            }
        )
    return geometry
