"use strict";

const data = window.CHRONOVPR_DATA;
const byId = (id) => document.getElementById(id);
const accuracy = (value) => value.toFixed(2) + "%";
const points = (value) => "+" + value.toFixed(2) + " pp";
const colors = {
  full: "var(--accent)", naive: "var(--warm)", single: "#a3b4be",
  oracle: "var(--blue)", gate: "var(--pink)",
};
const families = {
  early: ["Early CNN / VLAD", "square", "var(--warm)"],
  cnn: ["Large-scale CNN", "triangle", "var(--accent)"],
  vfm: ["Foundation models", "", "var(--blue)"],
  clip: ["CLIP", "diamond", "var(--pink)"],
};
const taskNames = {
  occlusion_recovery: "Occlusion recovery",
  out_of_fov: "Out-of-FoV reasoning",
  temporal_evolution: "Temporal evolution",
};
const views = {
  accuracy: ["VQA accuracy by retriever", "Each vertical pair shows the naïve and full-system VQA accuracy for one frozen retriever."],
  gate: ["Conflict Gate transfer", "Each point is one frozen retriever. Gate AUROC measures aliasing detection, and the dashed line shows the fit for the selected family or families."],
  ablation: ["Component ablation · CosPlace", "Accuracy for each configuration on the 8,047-question test set."],
  noise: ["Accuracy under injected aliasing noise", "Compare the two protocols across five noise levels in the separately reported stress test."],
};
const ablationNames = ["Naïve baseline", "CTSE only", "+ Pose and time", "+ Conflict Gate", "+ LoRA"];
const heldOut = new Set(["Nordland", "SPED"]);
const sortedDatasets = Object.entries(data.datasets).sort((a, b) => sum(b[1]) - sum(a[1]));
let plotMode = "accuracy";
let resizeObserver;
let sampleIndex = 0;

function sum(values) {
  return Object.values(values).reduce((total, value) => total + value, 0);
}

function element(tag, className, label) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (label !== undefined) node.textContent = label;
  return node;
}

function pressed(group, selected) {
  group.forEach((button) => button.setAttribute("aria-pressed", String(button === selected)));
}

document.querySelectorAll("[data-view]").forEach((button, _, group) => {
  button.addEventListener("click", () => {
    pressed(group, button);
    const isMemory = button.dataset.view === "memory";
    byId("scene-frame").dataset.frame = button.dataset.view;
    byId("scene-frame").setAttribute("aria-label", isMemory ? "Illustrative verified historical street view" : "Illustrative masked query street view");
    byId("scene-label").textContent = isMemory ? "VERIFIED MEMORY" : "QUERY VIEW";
  });
});

function selectDetail(id, detail) {
  byId("plot-detail").textContent = detail;
  byId("chart").querySelectorAll(".point").forEach((button) => {
    button.classList.toggle("active", button.dataset.key === id);
  });
}

function setTable(headers, rows) {
  const table = byId("results-table");
  table.replaceChildren();
  const tr = table.createTHead().insertRow();
  headers.forEach((header) => {
    const th = element("th", "", header);
    th.scope = "col";
    tr.append(th);
  });
  const body = table.createTBody();
  rows.forEach(({ id, cells, detail }) => {
    const row = body.insertRow();
    const label = element("th");
    label.scope = "row";
    const button = element("button", "table-button", cells[0]);
    button.type = "button";
    button.addEventListener("click", () => selectDetail(id, detail));
    label.append(button);
    row.append(label);
    cells.slice(1).forEach((value) => { row.insertCell().textContent = value; });
  });
}

function legend(entries) {
  const host = byId("plot-legend");
  host.replaceChildren();
  entries.forEach(([label, shape, color]) => {
    const item = element("span", "legend-item");
    const mark = element("span", "legend-mark " + shape);
    mark.setAttribute("aria-hidden", "true");
    mark.style.setProperty("--series", color);
    item.append(mark, label);
    host.append(item);
  });
}

function scatter({ xTicks, yTicks, xRange, yRange, xLabel, yFormat, marks, lines }) {
  const plot = element("div", "plot");
  const canvas = element("canvas");
  canvas.setAttribute("aria-hidden", "true");
  const x = (value) => (value - xRange[0]) / (xRange[1] - xRange[0]);
  const y = (value) => (yRange[1] - value) / (yRange[1] - yRange[0]);
  yTicks.forEach((value) => {
    const line = element("span", "gridline");
    line.style.top = y(value) * 100 + "%";
    line.append(element("span", "", yFormat(value)));
    plot.append(line);
  });
  xTicks.forEach((value) => {
    const tick = element("span", "tick", value + "%");
    tick.style.left = x(value) * 100 + "%";
    plot.append(tick);
  });
  plot.append(element("span", "axis-name", xLabel), canvas);
  marks.forEach(({ id, xValue, yValue, detail, label, shape, color }) => {
    const point = element("button", "point " + shape);
    point.type = "button";
    point.dataset.key = id;
    point.style.left = x(xValue) * 100 + "%";
    point.style.top = y(yValue) * 100 + "%";
    point.style.setProperty("--series", color);
    point.setAttribute("aria-label", label);
    point.title = label;
    point.addEventListener("click", () => selectDetail(id, detail));
    plot.append(point);
  });
  byId("chart").replaceChildren(plot);
  function draw() {
    const { width, height } = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    lines.forEach(({ coordinates, color, dash = [], opacity = 1, lineWidth = 1.5 }) => {
      context.strokeStyle = color;
      context.globalAlpha = opacity;
      context.lineWidth = lineWidth;
      context.setLineDash(dash);
      context.beginPath();
      coordinates.forEach(([xValue, yValue], index) => {
        const position = [width * x(xValue), height * y(yValue)];
        if (index === 0) context.moveTo(...position);
        else context.lineTo(...position);
      });
      context.stroke();
    });
  }
  draw();
  resizeObserver = new ResizeObserver(draw);
  resizeObserver.observe(plot);
}

function methodDetail(method) {
  return method.label + " · Recall@1 " + accuracy(method.VPR_R1) +
    " · Naïve " + accuracy(method.Naive_VQA_Acc) + " · Full " + accuracy(method.Ours_VQA_Acc) +
    " · Gain " + points(method.Gain) + " · Gate AUROC " + method.auroc.toFixed(4);
}

function leastSquares(methods) {
  const meanX = methods.reduce((total, method) => total + method.VPR_R1, 0) / methods.length;
  const meanY = methods.reduce((total, method) => total + method.auroc, 0) / methods.length;
  const cross = methods.reduce((total, method) => total + (method.VPR_R1 - meanX) * (method.auroc - meanY), 0);
  const squareX = methods.reduce((total, method) => total + (method.VPR_R1 - meanX) ** 2, 0);
  const squareY = methods.reduce((total, method) => total + (method.auroc - meanY) ** 2, 0);
  const fit = (x) => meanY + cross / squareX * (x - meanX);
  const xs = methods.map((method) => method.VPR_R1);
  const r = cross / Math.sqrt(squareX * squareY);
  return { coordinates: [Math.min(...xs), Math.max(...xs)].map((x) => [x, fit(x)]), r };
}

function retrieverPlot() {
  const family = byId("family").value;
  const methods = data.methods.filter((method) => family === "all" || method.family === family);
  const gateView = plotMode === "gate";
  const xTicks = [30, 40, 50, 60, 70];
  const xRange = [25, 75];
  const marks = [];
  const lines = [];
  const entries = [];
  if (gateView) {
    const fit = leastSquares(methods);
    lines.push({ coordinates: fit.coordinates, color: "#acbdc4", dash: [6, 5] });
    entries.push(...Object.entries(families).filter(([key]) => family === "all" || key === family).map(([, style]) => style));
    const prefix = methods.length > 2 ? "Pearson r = " + fit.r.toFixed(2) + " across " + methods.length + " retrievers. " : "Two CLIP retrievers. ";
    byId("plot-detail").textContent = prefix + "Select a point or a table row for exact values.";
  } else {
    entries.push(["Naïve VPR + MLLM", "open", colors.naive], ["Full system", "", colors.full]);
    const references = [
      [data.main["Single view"].overall_acc, "#a3b4be", [2, 5]],
      [data.main["Oracle + LoRA"].overall_acc, "#94c7ff", [7, 5]],
    ];
    references.forEach(([value, color, dash]) => {
      lines.push({ coordinates: [[xRange[0], value], [xRange[1], value]], color, dash, opacity: .55 });
    });
    byId("plot-explainer").textContent += " Reference lines show the single-view baseline (44.02%) and Oracle + LoRA (75.80%).";
  }
  methods.forEach((method) => {
    const common = { id: method.id, xValue: method.VPR_R1, detail: methodDetail(method) };
    if (gateView) {
      const [, shape, color] = families[method.family];
      marks.push({ ...common, yValue: method.auroc, shape, color, label: method.label + ", gate AUROC " + method.auroc.toFixed(4) });
    } else {
      lines.push({
        coordinates: [[method.VPR_R1, method.Naive_VQA_Acc], [method.VPR_R1, method.Ours_VQA_Acc]],
        color: "#93c5bb", opacity: .32,
      });
      marks.push(
        { ...common, yValue: method.Naive_VQA_Acc, shape: "open", color: colors.naive, label: method.label + ", naïve " + accuracy(method.Naive_VQA_Acc) },
        { ...common, yValue: method.Ours_VQA_Acc, shape: "", color: colors.full, label: method.label + ", full " + accuracy(method.Ours_VQA_Acc) },
      );
    }
  });
  legend(entries);
  scatter({
    xTicks, xRange, xLabel: "VPR Recall@1", marks, lines,
    yTicks: gateView ? [.65, .70, .75, .80, .85, .90] : [40, 50, 60, 70, 80],
    yRange: gateView ? [.62, .92] : [40, 80],
    yFormat: gateView ? (value) => value.toFixed(2) : (value) => value + "%",
  });
  setTable(["Retriever", "Recall@1", "Naïve", "Full", "Gain", "Gate AUROC"], methods.map((method) => ({
    id: method.id, detail: methodDetail(method),
    cells: [method.label, accuracy(method.VPR_R1), accuracy(method.Naive_VQA_Acc), accuracy(method.Ours_VQA_Acc), points(method.Gain), method.auroc.toFixed(4)],
  })));
}

function noisePlot() {
  const protocols = [
    ["Naïve VPR + MLLM", "Naive_VQA_Acc", "open", colors.naive, "#f6c073", [5, 4]],
    ["Gated system", "Ours_Gated_VQA_Acc", "", colors.full, "#8ff4d1", []],
  ];
  const describe = (row) => row.rate + "% injected noise · Naïve " + accuracy(row.Naive_VQA_Acc) +
    " · Gated " + accuracy(row.Ours_Gated_VQA_Acc) + " · Gain " + points(row.Robustness_Gain);
  const marks = protocols.flatMap(([label, key, shape, color]) =>
    data.noise.map((row) => ({
      id: String(row.rate), xValue: row.rate, yValue: row[key], shape, color,
      detail: describe(row), label: row.rate + "% noise, " + label + " " + accuracy(row[key]),
    }))
  );
  const lines = protocols.map(([, key, , , color, dash]) => ({
    color, dash, lineWidth: 2, coordinates: data.noise.map((row) => [row.rate, row[key]]),
  }));
  legend(protocols.map(([label, , shape, color]) => [label, shape, color]));
  scatter({
    xTicks: [0, 20, 40, 60, 80], xRange: [-2, 82], xLabel: "Injected aliasing noise",
    yTicks: [40, 50, 60, 70, 80], yRange: [40, 80], yFormat: (value) => value + "%", marks, lines,
  });
  setTable(["Injected noise", "Naïve", "Gated", "Gain"], data.noise.map((row) => ({
    id: String(row.rate), detail: describe(row),
    cells: [row.rate + "%", accuracy(row.Naive_VQA_Acc), accuracy(row.Ours_Gated_VQA_Acc), points(row.Robustness_Gain)],
  })));
}

function bars(host, rows, maximum, format) {
  const chart = element("div", "bar-chart");
  rows.forEach(({ label, value, color }) => {
    const row = element("div", "bar-row");
    const track = element("span", "bar-track");
    const fill = element("span", "bar-fill");
    track.setAttribute("aria-hidden", "true");
    fill.style.setProperty("--value", value / maximum * 100 + "%");
    if (color) fill.style.setProperty("--series", color);
    track.append(fill);
    row.append(element("span", "", label), track, element("strong", "", format(value)));
    chart.append(row);
  });
  host.replaceChildren(chart);
}

function ablationPlot() {
  const rows = Object.values(data.ablation).filter((value) => typeof value === "object");
  legend([]);
  bars(byId("chart"), rows.map((row, index) => ({
    label: ablationNames[index], value: row.overall_acc,
    color: index === rows.length - 1 ? colors.full : colors.single,
  })), 100, accuracy);
  byId("chart").append(element("p", "bar-scale", "VQA accuracy · 0–100%"));
  byId("plot-detail").textContent = "Select a table row for results by reasoning task.";
  setTable(["Configuration", "Overall", "Occlusion", "Out-of-FoV", "Temporal"], rows.map((row, index) => {
    const byTask = Object.keys(taskNames).map((key) => accuracy(row.by_task[key]));
    return {
      id: String(index),
      detail: ablationNames[index] + " · Overall " + accuracy(row.overall_acc) + " · " +
        Object.values(taskNames).map((name, task) => name + " " + byTask[task]).join(" · "),
      cells: [ablationNames[index], accuracy(row.overall_acc), ...byTask],
    };
  }));
}

function renderPlot() {
  resizeObserver?.disconnect();
  const [title, description] = views[plotMode];
  byId("plot-title").textContent = title;
  byId("plot-explainer").textContent = description;
  byId("plot-detail").textContent = "Select a point or a table row for exact values.";
  byId("family-field").hidden = plotMode === "ablation" || plotMode === "noise";
  if (plotMode === "ablation") ablationPlot();
  else if (plotMode === "noise") noisePlot();
  else retrieverPlot();
}

document.querySelectorAll("[data-plot]").forEach((button, _, group) => {
  button.addEventListener("click", () => {
    pressed(group, button);
    plotMode = button.dataset.plot;
    renderPlot();
  });
});
byId("family").addEventListener("change", renderPlot);

function renderSlice() {
  const choice = byId("slice").value;
  const swatches = [colors.single, colors.naive, colors.gate, colors.full, colors.oracle];
  const rows = Object.entries(data.main).map(([label, result], index) => ({
    label, color: swatches[index],
    value: choice === "overall" ? result.overall_acc : result.by_task[choice] ?? result.by_dataset[choice],
  }));
  bars(byId("slice-chart"), rows, 100, accuracy);
  byId("slice-chart").append(element("p", "bar-scale", "VQA accuracy · 0–100%"));
}

sortedDatasets.forEach(([name]) => {
  const option = element("option", "", name + (heldOut.has(name) ? " · OOD" : ""));
  option.value = name;
  byId("dataset-options").append(option);
});
byId("slice").addEventListener("change", renderSlice);

function renderDataset() {
  const host = byId("split-chart");
  const number = new Intl.NumberFormat("en-US").format;
  bars(host, [
    { label: "Train", value: data.splits.train },
    { label: "Val", value: data.splits.val, color: colors.single },
    { label: "Test", value: data.splits.test, color: colors.oracle },
  ], sum(data.splits), number);
  const table = element("table");
  const header = table.createTHead().insertRow();
  ["Dataset", "Test", "Total"].forEach((label) => {
    const th = element("th", "", label);
    th.scope = "col";
    header.append(th);
  });
  const body = table.createTBody();
  sortedDatasets.forEach(([name, counts]) => {
    const row = body.insertRow();
    const th = element("th", "", name + (heldOut.has(name) ? " · OOD" : ""));
    th.scope = "row";
    row.append(th);
    row.insertCell().textContent = number(counts.test);
    row.insertCell().textContent = number(sum(counts));
  });
  host.append(table);
  const note = element("p", "muted", "OOD datasets are test only.");
  note.style.margin = ".6rem 0 .8rem";
  host.append(note);
}

function renderDemo() {
  const task = byId("demo-task").value;
  const samples = data.examples.filter((sample) => task === "all" || sample.task_type === task);
  const example = samples[sampleIndex % samples.length];
  const options = byId("demo-options");
  const feedback = byId("demo-feedback");
  byId("demo-meta").textContent = [example.dataset, taskNames[example.task_type], example.split, "Δ yaw " +
    example.relative_yaw_deg + "°", "Δ time " + example.year_gap + " y"].join(" · ");
  byId("demo-question").textContent = example.question;
  options.replaceChildren();
  feedback.hidden = true;
  feedback.replaceChildren();
  byId("demo-instruction").hidden = false;
  Object.entries(example.options).sort().forEach(([key, label]) => {
    const button = element("button", "answer");
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.append(element("strong", "", key), element("span", "", label));
    button.addEventListener("click", () => {
      const correct = example.correct_option;
      [...options.children].forEach((choice) => {
        const choiceKey = choice.firstElementChild.textContent;
        choice.classList.toggle("correct", choiceKey === correct);
        choice.classList.toggle("incorrect", choice === button && key !== correct);
        choice.setAttribute("aria-pressed", String(choice === button));
      });
      feedback.replaceChildren(
        element("p", "", (key === correct ? "Correct. " : "") + "Reference answer: " + correct + " · " + example.options[correct]),
        element("p", "muted", example.evidence_summary),
      );
      byId("demo-instruction").hidden = true;
      feedback.hidden = false;
    });
    options.append(button);
  });
}

byId("next-question").addEventListener("click", () => {
  sampleIndex += 1;
  renderDemo();
});
byId("demo-task").addEventListener("change", () => {
  sampleIndex = 0;
  renderDemo();
});

renderPlot();
renderSlice();
renderDataset();
renderDemo();
