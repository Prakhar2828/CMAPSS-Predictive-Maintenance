"use strict";
const el = id => document.getElementById(id);
const format = value => value.toFixed(1);
const engineName = id => "Engine " + id.padStart(3, "0");
let fleet, ordered, selected, timer;

function stopReplay() {
  clearInterval(timer);
  timer = null;
  el("play").textContent = "Replay";
}

// Use the full observation window for the time axis so it stays fixed during replay.
function chart(target, rows, visible, series, range, band, title) {
  const width = Math.max(285, Math.round(el(target).getBoundingClientRect().width)), height = band ? 178 : 130;
  const left = 48, right = width - 12, bottom = height - 23;
  const x = i => left + i / Math.max(1, rows.length - 1) * (right - left);
  const y = value => bottom - (value - range[0]) / (range[1] - range[0] || 1) * (bottom - 12);
  const points = key => visible.map((row, i) => x(i).toFixed(1) + "," + y(row[key]).toFixed(1));
  const markup = [];
  const divisions = width < 480 ? 4 : 8;
  for (let tick = 0; tick <= divisions; tick++) {
    const i = Math.round(tick * (rows.length - 1) / divisions);
    markup.push('<path stroke="#39454d" stroke-dasharray="2 4" d="M'+x(i)+' 12V'+bottom+'"/>');
    if (tick > 0 && tick < divisions) markup.push('<text x="'+x(i)+'" y="'+(height-3)+'" text-anchor="middle">'+rows[i].cycle+'</text>');
  }
  for (const value of [range[0], (range[0] + range[1]) / 2, range[1]]) {
    markup.push('<path stroke="#45515a" d="M'+left+' '+y(value)+'H'+right+'"/>',
      '<text x="40" y="'+(y(value)+4)+'" text-anchor="end">'+Number(value.toFixed(1))+'</text>');
  }
  if (band) markup.push('<polygon fill="#614b3b" points="'+points("upper_RUL").concat(points("lower_RUL").reverse()).join(" ")+'"/>');
  for (const [key, color, dash] of series) {
    const coords = points(key).join(" ");
    markup.push('<polyline fill="none" stroke="'+color+'" stroke-width="2.5" stroke-dasharray="'+dash+'" points="'+coords+'"/>');
    markup.push('<circle fill="'+color+'" cx="'+x(visible.length-1)+'" cy="'+y(visible.at(-1)[key])+'" r="3"/>');
  }
  markup.push('<text x="'+left+'" y="'+(height-3)+'">1</text><text text-anchor="end" x="'+right+'" y="'+(height-3)+'">'+rows.at(-1).cycle+'</text>');
  el(target).innerHTML='<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="'+title+'" font-family="monospace" font-size="12" fill="#ced7dc">'+markup.join("")+'</svg>';
}

function render() {
  const rows = fleet[selected];
  const visible = rows.slice(0, Number(el("cycle").value));
  const row = visible.at(-1);
  el("name").textContent = engineName(selected);
  el("current").textContent = row.cycle + " / " + rows.length;
  el("priority").textContent = row.risk_level + " priority";
  el("priority").dataset.risk = row.risk_level;
  el("estimate").textContent = format(row.predicted_RUL);
  el("bounds").textContent = format(row.lower_RUL) + "–" + format(row.upper_RUL);
  el("actual").textContent = row.RUL;
  chart("rul", rows, visible, [["RUL","#c4d0d9","6 4"],["predicted_RUL","#ffaa63","none"]], [0,125], true, "Predicted and actual remaining life with calibrated interval");
  const key = el("sensor").value;
  const values = rows.map(item => item[key]);
  chart("trace", rows, visible, [[key,"#87c7f3","none"]], [Math.min(...values),Math.max(...values)], false, "Selected sensor readings through cycle " + row.cycle);
  el("flag").textContent = key.toUpperCase() + " reading " + row[key] + " · " +
    (row.anomaly_reason === "normal" ? "No sensor flag this cycle" : "Sensor flag · " + row.anomaly_reason);
}

function selectEngine(id) {
  stopReplay();
  selected = id;
  el("cycle").max = fleet[id].length;
  el("cycle").value = fleet[id].length;
  for (const button of el("roster").children) button.setAttribute("aria-pressed", String(button.value === id));
  render();
}

function updateRoster() {
  const ids = ordered.filter(id => !el("filter").value || fleet[id].at(-1).risk_level === el("filter").value);
  const buttons = ids.map(id => {
    const button = document.createElement("button");
    button.value = id;
    button.textContent = engineName(id);
    const priority = document.createElement("small");
    priority.textContent = fleet[id].at(-1).risk_level;
    const life = document.createElement("span");
    life.className = "queue-life";
    life.textContent = format(fleet[id].at(-1).predicted_RUL);
    priority.dataset.risk = fleet[id].at(-1).risk_level;
    button.append(life, priority);
    button.onclick = () => selectEngine(id);
    return button;
  });
  el("roster").replaceChildren(...buttons);
  selectEngine(ids.includes(selected) ? selected : ids[0]);
}

async function start() {
  const response = await fetch("./fleet.json");
  if (!response.ok) throw Error(response.status);
  const data = await response.json();
  fleet = Object.fromEntries(Object.entries(data.engines).map(([id, rows]) => [id,
    rows.map(row => Object.fromEntries(data.columns.map((name,i) => [name,row[i]])))]));
  ordered = Object.keys(fleet).sort((a,b) => fleet[a].at(-1).lower_RUL - fleet[b].at(-1).lower_RUL || Number(a)-Number(b));
  const counts = ["High","Medium","Low"].map(level => ordered.filter(id => fleet[id].at(-1).risk_level === level).length);
  el("stats").innerHTML = [["Test engines",ordered.length],["High priority",counts[0]],["Medium priority",counts[1]],["Low priority",counts[2]]]
    .map(([label,count]) => '<div><label>'+label+'</label><strong>'+count+'</strong></div>').join("");
  for (const [i, count] of counts.entries()) el("filter").options[i+1].disabled = count === 0;
  el("evaluation").textContent = "At the final observed cycle per engine, mean absolute error was " + format(data.mae) + " cycles and interval coverage was " + format(data.coverage) + "%.";
  el("filter").onchange = updateRoster;
  el("sensor").onchange = render;
  el("cycle").oninput = () => { stopReplay(); render(); };
  el("play").onclick = () => {
    if (timer) return stopReplay();
    const slider = el("cycle");
    if (slider.value === slider.max) slider.value = 1;
    render();
    el("play").textContent = "Pause";
    timer = setInterval(() => {
      slider.stepUp();
      render();
      if (slider.value === slider.max) stopReplay();
    }, 140);
  };
  el("status").hidden = true;
  el("monitor").hidden = false;
  updateRoster();
  new ResizeObserver(render).observe(el("rul"));
}
start().catch(() => { el("status").textContent = "Engine data could not load. Refresh the page to try again."; });
