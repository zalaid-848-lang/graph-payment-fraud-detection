import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const { PROJECT_ROOT, SKILL_DIR, RUNTIME_PYTHON } = process.env;
for (const [name, value] of Object.entries({ PROJECT_ROOT, SKILL_DIR, RUNTIME_PYTHON })) {
  if (!path.isAbsolute(value ?? "")) throw new Error(`${name} must be an absolute path`);
}

const {
  applyPresentationChartFont,
  finalizePresentation,
  makeNativeBulletParagraphs,
  resolvePresentationFont,
} = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href
);

const BUILD_DIR = path.join(PROJECT_ROOT, "presentation", ".build");
const COVER_PATH = path.join(PROJECT_ROOT, "presentation", "assets", "network-cover.png");
const FINAL_PPTX = path.join(
  PROJECT_ROOT,
  "presentation",
  "output",
  "graph-payment-fraud-interview-v3.pptx",
);
await fs.mkdir(BUILD_DIR, { recursive: true });
const family = resolvePresentationFont();
const coverBytes = await fs.readFile(COVER_PATH);

const W = 1280;
const H = 720;
const COLORS = {
  navy: "#091B33",
  navy2: "#12345A",
  blue: "#2878C8",
  blueLight: "#DCEBFA",
  teal: "#0F9D8A",
  tealLight: "#D9F3EE",
  amber: "#F59E0B",
  amberLight: "#FEF0C7",
  coral: "#E7524B",
  coralLight: "#FDE2E0",
  ink: "#142235",
  slate: "#53657A",
  muted: "#7A899A",
  line: "#C8D3E0",
  white: "#FFFFFF",
  canvas: "#F6F8FB",
};

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: options.fill ?? "none",
    line: options.line ?? { fill: "none", width: 0 },
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
  });
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 24,
    bold: options.bold ?? false,
    color: options.color ?? COLORS.ink,
    alignment: options.alignment ?? "left",
    verticalAlignment: options.verticalAlignment ?? "top",
    autoFit: options.autoFit ?? "none",
    insets: options.insets ?? { top: 2, right: 2, bottom: 2, left: 2 },
  };
  return shape;
}

function addBulletList(slide, items, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: options.marginLeftPoints ?? 18,
    hangingPoints: options.hangingPoints ?? 9,
    spaceAfterPoints: options.spaceAfterPoints ?? 8,
  });
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 22,
    color: options.color ?? COLORS.ink,
    autoFit: "none",
    insets: { top: 2, right: 4, bottom: 2, left: 2 },
  };
  return shape;
}

function addRect(slide, position, options = {}) {
  return slide.shapes.add({
    geometry: options.geometry ?? "roundRect",
    position,
    fill: options.fill ?? COLORS.white,
    line: options.line ?? { style: "solid", fill: COLORS.line, width: 1 },
    ...(options.borderRadius ? { borderRadius: options.borderRadius } : {}),
  });
}

function addLabeledRect(slide, text, position, options = {}) {
  const shape = addRect(slide, position, options);
  shape.text = text;
  shape.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 20,
    bold: options.bold ?? true,
    color: options.color ?? COLORS.ink,
    alignment: options.alignment ?? "center",
    verticalAlignment: "middle",
    autoFit: "none",
    insets: { top: 8, right: 10, bottom: 8, left: 10 },
  };
  return shape;
}

function addNode(slide, label, x, y, options = {}) {
  const size = options.size ?? 88;
  const width = options.width ?? size;
  const height = options.height ?? size;
  const node = slide.shapes.add({
    geometry: options.geometry ?? "ellipse",
    position: { left: x, top: y, width, height },
    fill: options.fill ?? COLORS.white,
    line: { style: "solid", fill: options.line ?? COLORS.blue, width: options.lineWidth ?? 2 },
  });
  node.text = label;
  node.text.style = {
    typeface: family,
    fontSize: options.fontSize ?? 17,
    bold: options.bold ?? true,
    color: options.color ?? COLORS.ink,
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "none",
    insets: { top: 4, right: 4, bottom: 4, left: 4 },
  };
  return node;
}

function addLine(slide, x1, y1, x2, y2, options = {}) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  return slide.shapes.add({
    geometry: "line",
    position: {
      left,
      top,
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1),
      horizontalFlip: x2 < x1,
      verticalFlip: y2 < y1,
    },
    fill: "none",
    line: {
      style: options.style ?? "solid",
      fill: options.fill ?? COLORS.line,
      width: options.width ?? 2,
    },
  });
}

function addSlideTitle(slide, title, number, options = {}) {
  addText(slide, title, { left: 64, top: 36, width: 1090, height: 58 }, {
    fontSize: options.fontSize ?? 34,
    bold: true,
    color: options.color ?? COLORS.navy,
  });
  addText(slide, `${number} / 10`, { left: 1160, top: 46, width: 70, height: 28 }, {
    fontSize: 14,
    color: options.numberColor ?? COLORS.muted,
    alignment: "right",
  });
}

function addFooter(slide, source) {
  addLine(slide, 64, 684, 1216, 684, { fill: "#DCE3EC", width: 1 });
  addText(slide, source, { left: 64, top: 690, width: 1080, height: 20 }, {
    fontSize: 11,
    color: COLORS.muted,
  });
}

function addNotes(slide, text, sources) {
  slide.speakerNotes.textFrame.setText(
    `${text}\n\nSources in this repository:\n${sources.map((item) => `- ${item}`).join("\n")}`,
  );
}

function addChartFonts(chart) {
  applyPresentationChartFont(chart, { fontFamily: family });
}

// Slide 1: cover
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.navy;
  slide.images.add({
    blob: coverBytes,
    contentType: "image/png",
    alt: "Abstract payment network with a small amber cluster highlighted for investigation",
    fit: "cover",
    position: { left: 0, top: 0, width: W, height: H },
  });
  addText(
    slide,
    "Graph-Based Payment\nFraud Ring Detection",
    { left: 70, top: 138, width: 555, height: 190 },
    { fontSize: 46, bold: true, color: COLORS.white },
  );
  addText(
    slide,
    "Investigator prioritisation with leakage-safe graph features",
    { left: 74, top: 345, width: 480, height: 78 },
    { fontSize: 24, color: "#C9DCF3" },
  );
  addText(
    slide,
    "Independent data-science project\nCampus placement interview, September 2026",
    { left: 74, top: 568, width: 470, height: 66 },
    { fontSize: 16, color: "#AFC7E3" },
  );
  addNotes(
    slide,
    "Open with the business problem: coordinated fraud can appear ordinary when each transaction is scored alone. This project adds relationship context while keeping the final decision with an investigator. Avoid saying the highlighted cluster is a confirmed ring.",
    ["README.md", "docs/limitations.md"],
  );
}

// Slide 2: business problem
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, "Why transaction-only scoring misses coordinated activity", 2);
  addText(slide, "Transaction view", { left: 90, top: 125, width: 390, height: 36 }, {
    fontSize: 23,
    bold: true,
    color: COLORS.slate,
    alignment: "center",
  });
  const isolated = [
    [150, 220, "₹"], [300, 195, "₹"], [410, 275, "₹"],
    [195, 370, "₹"], [355, 410, "₹"],
  ];
  for (const [x, y, label] of isolated) {
    addNode(slide, label, x, y, { size: 58, fill: COLORS.white, line: COLORS.line, fontSize: 20 });
  }
  addText(
    slide,
    "Each event can look ordinary when the model sees only amount, product, and email domain.",
    { left: 105, top: 520, width: 390, height: 78 },
    { fontSize: 19, color: COLORS.slate, alignment: "center" },
  );

  addText(slide, "Relationship view", { left: 715, top: 125, width: 430, height: 36 }, {
    fontSize: 23,
    bold: true,
    color: COLORS.navy,
    alignment: "center",
  });
  addLine(slide, 820, 253, 960, 240, { fill: COLORS.blue, width: 2 });
  addLine(slide, 820, 253, 940, 365, { fill: COLORS.blue, width: 2 });
  addLine(slide, 960, 240, 1070, 330, { fill: COLORS.amber, width: 3 });
  addLine(slide, 940, 365, 1070, 330, { fill: COLORS.amber, width: 3 });
  addLine(slide, 960, 240, 1000, 470, { fill: COLORS.blue, width: 2 });
  addLine(slide, 940, 365, 1000, 470, { fill: COLORS.blue, width: 2 });
  addNode(slide, "Txn", 780, 215, { size: 78, fill: COLORS.blueLight, line: COLORS.blue });
  addNode(slide, "Device", 915, 200, { width: 96, height: 82, fill: COLORS.tealLight, line: COLORS.teal, fontSize: 15 });
  addNode(slide, "Card", 900, 325, { size: 82, fill: COLORS.blueLight, line: COLORS.blue, fontSize: 16 });
  addNode(slide, "Recipient", 1012, 287, { width: 120, height: 90, fill: COLORS.amberLight, line: COLORS.amber, fontSize: 14 });
  addNode(slide, "Txn", 960, 430, { size: 78, fill: COLORS.coralLight, line: COLORS.coral });
  addText(
    slide,
    "Shared entities expose repeated behaviour and connected activity for investigator review.",
    { left: 735, top: 540, width: 410, height: 65 },
    { fontSize: 19, color: COLORS.navy, alignment: "center", bold: true },
  );
  addFooter(slide, "Conceptual view. A connection is an investigation lead, not proof of collusion.");
  addNotes(
    slide,
    "Contrast the two questions. Tabular scoring asks how risky one transaction looks. Graph scoring also asks whether the transaction shares specific entities with earlier activity. Shared infrastructure has legitimate explanations, so connections guide review rather than confirm wrongdoing.",
    ["README.md", "docs/graph_schema.md", "docs/limitations.md"],
  );
}

// Slide 3: graph schema
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  addSlideTitle(slide, "Entity-link graph", 3);
  addLine(slide, 285, 245, 590, 330, { fill: COLORS.line, width: 2 });
  addLine(slide, 285, 425, 590, 350, { fill: COLORS.line, width: 2 });
  addLine(slide, 590, 340, 935, 205, { fill: COLORS.line, width: 2 });
  addLine(slide, 590, 340, 970, 340, { fill: COLORS.line, width: 2 });
  addLine(slide, 590, 340, 935, 490, { fill: COLORS.line, width: 2 });
  addNode(slide, "Card\nproxy", 210, 195, { size: 108, fill: COLORS.blueLight, line: COLORS.blue });
  addNode(slide, "Customer\nproxy", 202, 375, { width: 124, height: 108, fill: COLORS.tealLight, line: COLORS.teal, fontSize: 15 });
  addNode(slide, "Transaction", 515, 285, { width: 155, height: 120, fill: COLORS.navy, line: COLORS.navy, color: COLORS.white, fontSize: 14 });
  addNode(slide, "Device", 895, 155, { size: 108, fill: COLORS.tealLight, line: COLORS.teal });
  addNode(slide, "Address\nproxy", 930, 285, { size: 108, fill: COLORS.blueLight, line: COLORS.blue, fontSize: 16 });
  addNode(slide, "Recipient\nproxy", 884, 440, { width: 132, height: 108, fill: COLORS.amberLight, line: COLORS.amber, fontSize: 15 });
  addLabeledRect(
    slide,
    "Email domains stay in broad degree features but are excluded from specific components and the dashboard graph.",
    { left: 245, top: 580, width: 790, height: 60 },
    { fill: "#EEF2F7", line: { style: "solid", fill: COLORS.line, width: 1 }, fontSize: 17, bold: false, color: COLORS.slate },
  );
  addFooter(slide, "IEEE-CIS customer and recipient nodes are anonymized proxies, not verified identities.");
  addNotes(
    slide,
    "Explain the heterogeneous graph. Transactions connect to entity proxies. The model never receives raw entity identifiers. Common email domains can merge unrelated activity, so the implementation separates broad features from more specific component and dashboard views.",
    ["docs/graph_schema.md", "docs/graph_feature_dictionary.md", "docs/data_dictionary.md"],
  );
}

// Slide 4: temporal validation
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, "Leakage-safe temporal validation", 4);
  const y = 205;
  const segments = [
    [80, 485, COLORS.blue, "Train\n60%"],
    [565, 90, "#D5DCE5", "24 h\npurge"],
    [655, 205, COLORS.teal, "Validation\n20%"],
    [860, 90, "#D5DCE5", "24 h\npurge"],
    [950, 250, COLORS.amber, "Test\n20%"],
  ];
  for (const [x, width, fill, label] of segments) {
    addLabeledRect(slide, label, { left: x, top: y, width, height: 92 }, {
      fill,
      line: { fill: "none", width: 0 },
      color: fill === "#D5DCE5" ? COLORS.slate : COLORS.white,
      fontSize: 18,
      borderRadius: "rounded-sm",
    });
  }
  addText(slide, "Earlier time", { left: 80, top: 165, width: 160, height: 25 }, {
    fontSize: 15,
    color: COLORS.muted,
  });
  addText(slide, "Later time", { left: 1040, top: 165, width: 160, height: 25 }, {
    fontSize: 15,
    color: COLORS.muted,
    alignment: "right",
  });
  addBulletList(
    slide,
    [
      "Preprocessing and thresholds learn from training rows only.",
      "Graph features use relationships observed strictly before each event-time batch.",
      "Neighbour outcomes use matured training labels only. Validation and test labels never enter state.",
    ],
    { left: 175, top: 385, width: 930, height: 175 },
    { fontSize: 21, spaceAfterPoints: 12 },
  );
  addLabeledRect(slide, "Same-time transactions score before graph state updates", { left: 305, top: 585, width: 670, height: 48 }, {
    fill: COLORS.navy,
    line: { fill: "none", width: 0 },
    color: COLORS.white,
    fontSize: 18,
  });
  addFooter(slide, "Current configuration uses chronological 60 / 20 / 20 periods with two 24-hour purge gaps.");
  addNotes(
    slide,
    "This is the most important technical-control slide. Walk through availability at scoring time. Earlier unlabelled relationship attributes can update structural state, as they would in streaming. Outcome-derived features are stricter and accept only training labels after the configured maturity delay.",
    ["docs/validation_plan.md", "docs/label_history_features.md", "configs/project.toml"],
  );
}

// Slide 5: feature and model design
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  addSlideTitle(slide, "Feature and model design", 5);
  const boxes = [
    [70, 165, 235, COLORS.blueLight, COLORS.blue, "Transaction inputs\nAmount, product, presence"],
    [350, 165, 235, COLORS.tealLight, COLORS.teal, "Historical graph\nDegree, components, PageRank"],
    [630, 165, 235, COLORS.amberLight, COLORS.amber, "Matured outcomes\nNeighbour counts and ratio"],
    [910, 165, 285, COLORS.navy, COLORS.navy, "Combined matrix\n48 transformed features"],
  ];
  for (const [x, y, width, fill, line, label] of boxes) {
    addLabeledRect(slide, label, { left: x, top: y, width, height: 115 }, {
      fill,
      line: { style: "solid", fill: line, width: 2 },
      color: fill === COLORS.navy ? COLORS.white : COLORS.ink,
      fontSize: 18,
    });
  }
  addLine(slide, 305, 222, 350, 222, { fill: COLORS.line, width: 3 });
  addLine(slide, 585, 222, 630, 222, { fill: COLORS.line, width: 3 });
  addLine(slide, 865, 222, 910, 222, { fill: COLORS.line, width: 3 });
  addText(slide, "Controlled comparison", { left: 85, top: 350, width: 330, height: 40 }, {
    fontSize: 24,
    bold: true,
    color: COLORS.navy,
  });
  addBulletList(
    slide,
    [
      "Rules establish an operational reference point.",
      "The same logistic classifier tests tabular features with and without graph context.",
      "LightGBM adds nonlinear interactions only after the graph-feature ablation is complete.",
    ],
    { left: 85, top: 405, width: 610, height: 190 },
    { fontSize: 20, spaceAfterPoints: 10 },
  );
  addText(slide, "Core over stretch goal", { left: 790, top: 355, width: 330, height: 40 }, {
    fontSize: 24,
    bold: true,
    color: COLORS.navy,
  });
  addLabeledRect(
    slide,
    "A complete graph-feature model, capacity evaluation, explanations, and dashboard came before GraphSAGE or graph attention.",
    { left: 790, top: 420, width: 395, height: 145 },
    { fill: "#EEF2F7", line: { style: "solid", fill: COLORS.line, width: 1 }, fontSize: 19, bold: false, color: COLORS.slate },
  );
  addFooter(slide, "28 graph features enter the model after excluding one PageRank freshness diagnostic.");
  addNotes(
    slide,
    "Explain the experiment discipline. The logistic ablation isolates whether graph context adds signal without changing classifier family. LightGBM is a later candidate, not a substitute for proving the core pipeline. Raw identifiers are excluded from the model matrix.",
    ["reports/day5_model_comparison.md", "docs/graph_feature_dictionary.md", "src/fraud_detection/model.py"],
  );
}

// Slide 6: PR-AUC
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, "Graph context improved ranking on synthetic data", 6);
  const chart = slide.charts.add("bar", {
    position: { left: 75, top: 140, width: 790, height: 440 },
    categories: ["Rules", "Tabular logistic", "Graph logistic", "Graph LightGBM"],
    series: [{
      name: "PR-AUC",
      values: [0.2976, 0.4989, 0.8935, 0.8869],
      valuesFormatCode: "0.0000",
      fill: COLORS.blue,
      points: [
        { idx: 0, fill: "#A7B4C2" },
        { idx: 1, fill: "#6FA5D7" },
        { idx: 2, fill: COLORS.teal },
        { idx: 3, fill: COLORS.amber },
      ],
    }],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 42 },
    hasLegend: false,
    xAxis: {
      visible: true,
      min: 0,
      max: 1,
      majorUnit: 0.2,
      numberFormatCode: "0.0",
      majorGridlines: { style: "solid", fill: "#D9E1EA", width: 1 },
      textStyle: { fill: COLORS.slate, fontSize: 13 },
    },
    yAxis: { textStyle: { fill: COLORS.ink, fontSize: 15 }, line: { fill: "none", width: 0 } },
    dataLabels: {
      showValue: true,
      position: "outEnd",
      textStyle: { fill: COLORS.ink, fontSize: 14, bold: true },
    },
    chartFill: COLORS.canvas,
    plotAreaFill: COLORS.canvas,
  });
  addChartFonts(chart);
  addText(slide, "+0.3946", { left: 925, top: 205, width: 230, height: 70 }, {
    fontSize: 42,
    bold: true,
    color: COLORS.teal,
    alignment: "center",
  });
  addText(slide, "PR-AUC gain from adding graph features to logistic regression", { left: 910, top: 285, width: 260, height: 90 }, {
    fontSize: 20,
    color: COLORS.slate,
    alignment: "center",
  });
  addLabeledRect(slide, "Synthetic test: 84 rows, 8 fraud labels", { left: 910, top: 440, width: 270, height: 60 }, {
    fill: COLORS.coralLight,
    line: { style: "solid", fill: "#F5A6A1", width: 1 },
    color: "#9B2924",
    fontSize: 17,
  });
  addFooter(slide, "PR-AUC measures ranking quality under class imbalance. Results are synthetic development evidence.");
  addNotes(
    slide,
    "Lead with the controlled comparison: the graph logistic model uses the same classifier as the tabular baseline, so the large improvement is attributable to added graph context within this synthetic experiment. Do not generalize the magnitude to real bank data.",
    ["reports/experiment_report.md", "artifacts/day6/metrics_and_errors.json"],
  );
}

// Slide 7: capacity impact
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  addSlideTitle(slide, "Capacity impact on the synthetic test", 7);
  addText(slide, "Recall within the top 5%", { left: 85, top: 120, width: 480, height: 38 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.navy,
    alignment: "center",
  });
  const recallChart = slide.charts.add("bar", {
    position: { left: 70, top: 170, width: 520, height: 350 },
    categories: ["Rules", "Tabular", "Graph logistic", "Graph LightGBM"],
    series: [{
      name: "Recall",
      values: [0.375, 0.375, 0.625, 0.625],
      valuesFormatCode: "0%",
      fill: COLORS.teal,
      points: [{ idx: 0, fill: "#A7B4C2" }, { idx: 1, fill: "#6FA5D7" }],
    }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    hasLegend: false,
    xAxis: { textStyle: { fill: COLORS.slate, fontSize: 12 } },
    yAxis: {
      min: 0,
      max: 0.8,
      majorUnit: 0.2,
      numberFormatCode: "0%",
      majorGridlines: { style: "solid", fill: "#E0E6ED", width: 1 },
      textStyle: { fill: COLORS.slate, fontSize: 12 },
    },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: COLORS.ink, fontSize: 13, bold: true } },
    chartFill: COLORS.white,
    plotAreaFill: COLORS.white,
  });
  addChartFonts(recallChart);

  addText(slide, "Alerts needed for 37.5% recall", { left: 700, top: 120, width: 480, height: 38 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.navy,
    alignment: "center",
  });
  const alertsChart = slide.charts.add("bar", {
    position: { left: 690, top: 170, width: 520, height: 350 },
    categories: ["Rules", "Tabular", "Graph logistic", "Graph LightGBM"],
    series: [{
      name: "Alerts",
      values: [4, 5, 3, 3],
      valuesFormatCode: "0",
      fill: COLORS.amber,
      points: [{ idx: 0, fill: "#A7B4C2" }, { idx: 1, fill: "#6FA5D7" }],
    }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    hasLegend: false,
    xAxis: { textStyle: { fill: COLORS.slate, fontSize: 12 } },
    yAxis: {
      min: 0,
      max: 6,
      majorUnit: 1,
      numberFormatCode: "0",
      majorGridlines: { style: "solid", fill: "#E0E6ED", width: 1 },
      textStyle: { fill: COLORS.slate, fontSize: 12 },
    },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: COLORS.ink, fontSize: 13, bold: true } },
    chartFill: COLORS.white,
    plotAreaFill: COLORS.white,
  });
  addChartFonts(alertsChart);
  addText(slide, "25% fewer alerts than rules at matched recall", { left: 760, top: 545, width: 370, height: 45 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.amber,
    alignment: "center",
  });
  addFooter(slide, "At a hard 5% capacity, each graph model found 5 of 8 fraud-labelled transactions with no false positives.");
  addNotes(
    slide,
    "Translate ranking into investigator workload. At the fixed top-five alert queue, graph models reached 62.5 percent recall. At a matched 37.5 percent recall, they needed three alerts instead of four for rules. The counts are tiny, so present them as a worked operational example rather than a forecast.",
    ["reports/experiment_report.md", "reports/day6_boosted_error_analysis.md"],
  );
}

// Slide 8: selection integrity
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.canvas;
  addSlideTitle(slide, "Validation selected LightGBM, and the frozen test stayed honest", 8);
  const chart = slide.charts.add("bar", {
    position: { left: 85, top: 155, width: 720, height: 390 },
    categories: ["Graph logistic", "Graph LightGBM"],
    series: [
      { name: "Validation PR-AUC", values: [0.9250, 0.9519], valuesFormatCode: "0.0000", fill: COLORS.teal },
      { name: "Frozen test PR-AUC", values: [0.8935, 0.8869], valuesFormatCode: "0.0000", fill: COLORS.amber },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 60 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fill: COLORS.slate, fontSize: 13 } },
    xAxis: { textStyle: { fill: COLORS.ink, fontSize: 15 } },
    yAxis: {
      min: 0.84,
      max: 0.98,
      majorUnit: 0.02,
      numberFormatCode: "0.00",
      majorGridlines: { style: "solid", fill: "#D9E1EA", width: 1 },
      textStyle: { fill: COLORS.slate, fontSize: 12 },
    },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fill: COLORS.ink, fontSize: 13, bold: true } },
    chartFill: COLORS.canvas,
    plotAreaFill: COLORS.canvas,
  });
  addChartFonts(chart);
  addText(slide, "Selection rule", { left: 875, top: 160, width: 280, height: 38 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.navy,
    alignment: "center",
  });
  addLabeledRect(
    slide,
    "Choose the graph candidate with higher validation PR-AUC. Report the frozen test after selection.",
    { left: 860, top: 220, width: 310, height: 130 },
    { fill: COLORS.tealLight, line: { style: "solid", fill: COLORS.teal, width: 1 }, fontSize: 19, bold: false },
  );
  addText(slide, "The project does not switch models after seeing the test result.", { left: 865, top: 400, width: 300, height: 80 }, {
    fontSize: 22,
    bold: true,
    color: COLORS.coral,
    alignment: "center",
  });
  addText(slide, "LightGBM stopped at iteration 3 of 400", { left: 850, top: 520, width: 330, height: 36 }, {
    fontSize: 17,
    color: COLORS.slate,
    alignment: "center",
  });
  addFooter(slide, "A small synthetic holdout can change model ordering. Real temporal backtests are the next evidence step.");
  addNotes(
    slide,
    "This slide demonstrates model-selection discipline. LightGBM won on validation and therefore remains the selected model. Graph logistic regression later had a slightly higher test PR-AUC. Switching after seeing the test would leak evaluation information into the choice. Early stopping at iteration three also shows limited tuning evidence.",
    ["reports/experiment_report.md", "reports/day7_explainability.md", "docs/model_card.md"],
  );
}

// Slide 9: investigator workflow
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.white;
  addSlideTitle(slide, "Investigator review workflow", 9);
  const steps = [
    [75, COLORS.blueLight, COLORS.blue, "Capacity-limited\nalert queue"],
    [345, COLORS.tealLight, COLORS.teal, "Evidence-gated\nSHAP reasons"],
    [615, COLORS.amberLight, COLORS.amber, "Earlier shared-entity\nconnection view"],
    [915, COLORS.navy, COLORS.navy, "Human\ninvestigator"],
  ];
  for (const [x, fill, line, label] of steps) {
    addLabeledRect(slide, label, { left: x, top: 205, width: 230, height: 120 }, {
      fill,
      line: { style: "solid", fill: line, width: 2 },
      color: fill === COLORS.navy ? COLORS.white : COLORS.ink,
      fontSize: 19,
    });
  }
  addLine(slide, 305, 265, 345, 265, { fill: COLORS.line, width: 3 });
  addLine(slide, 575, 265, 615, 265, { fill: COLORS.line, width: 3 });
  addLine(slide, 845, 265, 915, 265, { fill: COLORS.line, width: 3 });

  addText(slide, "What the interface protects", { left: 90, top: 405, width: 360, height: 40 }, {
    fontSize: 23,
    bold: true,
    color: COLORS.navy,
  });
  addBulletList(
    slide,
    [
      "Evaluation labels never reach the dashboard data layer.",
      "Entity references are masked and email-domain hubs are excluded.",
      "Local graphs show strictly earlier transactions and cap the view at 20 neighbours.",
    ],
    { left: 90, top: 455, width: 575, height: 165 },
    { fontSize: 19, spaceAfterPoints: 9 },
  );
  addLabeledRect(
    slide,
    "Recommendation\nINVESTIGATOR_REVIEW\n\nNo automatic blocking",
    { left: 795, top: 420, width: 340, height: 175 },
    { fill: COLORS.coralLight, line: { style: "solid", fill: COLORS.coral, width: 2 }, color: "#8F2823", fontSize: 20 },
  );
  addFooter(slide, "The dashboard describes suspected fraud-ring connections, not verified criminal networks.");
  addNotes(
    slide,
    "Demonstrate the interface after this slide if time permits. Explain that each reason requires positive SHAP contribution and factual feature evidence. The local graph deliberately shows only information available before the selected transaction. The action remains investigator review.",
    ["docs/dashboard_guide.md", "reports/day8_dashboard.md", "docs/reason_codes.md"],
  );
}

// Slide 10: close
{
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.navy;
  addSlideTitle(slide, "What the project proves and what comes next", 10, {
    color: COLORS.white,
    numberColor: "#AFC7E3",
  });
  addText(slide, "Supported by the current evidence", { left: 85, top: 145, width: 475, height: 45 }, {
    fontSize: 25,
    bold: true,
    color: "#8CD8CB",
  });
  addBulletList(
    slide,
    [
      "The pipeline is reproducible and leakage-aware.",
      "Graph context improved ranking over the tabular baseline on synthetic data.",
      "The system connects model evidence to a review-only investigator workflow.",
    ],
    { left: 85, top: 215, width: 500, height: 210 },
    { fontSize: 21, color: COLORS.white, spaceAfterPoints: 12 },
  );
  addText(slide, "Still unproven", { left: 720, top: 145, width: 420, height: 45 }, {
    fontSize: 25,
    bold: true,
    color: "#FFCA70",
  });
  addBulletList(
    slide,
    [
      "Performance on IEEE-CIS or real bank data.",
      "Verified fraud-ring detection because the dataset has no ring labels.",
      "Probability calibration, investigator utility, and production controls.",
    ],
    { left: 720, top: 215, width: 470, height: 210 },
    { fontSize: 21, color: COLORS.white, spaceAfterPoints: 12 },
  );
  addLabeledRect(
    slide,
    "Next evidence step: rerun the unchanged temporal protocol on IEEE-CIS",
    { left: 190, top: 520, width: 900, height: 82 },
    { fill: COLORS.white, line: { fill: "none", width: 0 }, color: COLORS.navy, fontSize: 24 },
  );
  addText(slide, "Questions", { left: 525, top: 635, width: 230, height: 40 }, {
    fontSize: 24,
    bold: true,
    color: "#C9DCF3",
    alignment: "center",
  });
  addNotes(
    slide,
    "Close with claim discipline. The project proves the engineering and evaluation approach on synthetic data. It does not prove deployment performance or ring-level accuracy. The next step is real-data validation, not a deep graph model added for novelty.",
    ["docs/limitations.md", "docs/model_card.md", "reports/experiment_report.md"],
  );
}

const requirements = {
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [6, 7, 8],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const expectedSlideSizeEmu = "12192000,6858000";
const stagingDir = path.join(BUILD_DIR, "finalizer-v3");
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const candidatePath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir: PROJECT_ROOT,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_package_integrity.py",
  ),
  layoutValidatorPath: path.join(
    SKILL_DIR,
    "container_tools/inspect_presentation_layout_geometry.py",
  ),
  layoutArgs: [
    "--expected-slide-size-emu",
    expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--validate-heading-punctuation",
  ],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "graph-payment-fraud-interview-v3.validation.json"),
});

console.log(JSON.stringify({ family, finalPath: FINAL_PPTX, result }, null, 2));
