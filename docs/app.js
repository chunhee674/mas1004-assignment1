// The whole demo. It loads the model your export script wrote, and runs it
// right here in the browser with ONNX Runtime Web. There is no server and
// nothing is uploaded.
//
// You do not need to change this file. Read toInput() if you are curious: it
// prepares a picture with the same six steps as prepare_image in src/data.py.

let model = null;      // the contents of model.json
let session = null;    // the network from model.onnx, ready to run

const $ = (id) => document.getElementById(id);

// GitHub Pages cannot send the headers that multi-threaded WebAssembly needs,
// so run on one thread. For one picture at a time that is fast enough.
ort.env.wasm.numThreads = 1;

// ---------------------------------------------------------------------------
// The model itself
// ---------------------------------------------------------------------------

async function forward(input) {
  const crop = model.input.crop;
  const tensor = new ort.Tensor("float32", input, [1, 3, crop, crop]);
  const outputs = await session.run({ [model.input.name]: tensor });
  return Array.from(outputs[model.output.name].data);
}

function softmax(logits) {
  const biggest = Math.max(...logits);
  const exponentials = logits.map((v) => Math.exp(v - biggest));
  const total = exponentials.reduce((a, b) => a + b, 0);
  return exponentials.map((v) => v / total);
}

// ---------------------------------------------------------------------------
// Preparing a picture, the same way prepare_image did in Python
// ---------------------------------------------------------------------------

const scratch = document.createElement("canvas");

function resizeShorterSide(source, target) {
  // Step 2: the shorter side becomes `target` and the shape is kept.
  // Going from a 4000 pixel photo to 256 in one step throws away almost every
  // pixel. Halving repeatedly keeps much more, and lands closer to what
  // Pillow does in Python.
  let width = source.videoWidth || source.naturalWidth || source.width;
  let height = source.videoHeight || source.naturalHeight || source.height;
  const scale = target / Math.min(width, height);
  const finalWidth = width <= height ? target : Math.floor(width * scale);
  const finalHeight = width <= height ? Math.floor(height * scale) : target;

  let stage = document.createElement("canvas");
  stage.width = width;
  stage.height = height;
  stage.getContext("2d").drawImage(source, 0, 0, width, height);

  while (width >= finalWidth * 2 && height >= finalHeight * 2) {
    const next = document.createElement("canvas");
    next.width = Math.floor(width / 2);
    next.height = Math.floor(height / 2);
    const context = next.getContext("2d");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(stage, 0, 0, next.width, next.height);
    stage = next;
    width = next.width;
    height = next.height;
  }

  if (width === finalWidth && height === finalHeight) return stage;
  const last = document.createElement("canvas");
  last.width = finalWidth;
  last.height = finalHeight;
  const context = last.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(stage, 0, 0, finalWidth, finalHeight);
  return last;
}

function toInput(source) {
  const { resize, crop, mean, std } = model.input;

  // Step 1 happens by itself: a canvas always holds red, green, blue.
  const resized = resizeShorterSide(source, resize);

  // Step 3: cut out the crop x crop square in the middle.
  const left = Math.round((resized.width - crop) / 2);
  const top = Math.round((resized.height - crop) / 2);
  scratch.width = crop;
  scratch.height = crop;
  const context = scratch.getContext("2d", { willReadFrequently: true });
  context.clearRect(0, 0, crop, crop);
  context.drawImage(resized, left, top, crop, crop, 0, 0, crop, crop);
  const pixels = context.getImageData(0, 0, crop, crop).data;

  // Steps 4, 5 and 6: divide by 255, subtract the mean and divide by the
  // standard deviation of each channel, and put the channels first.
  const area = crop * crop;
  const values = new Float32Array(3 * area);
  for (let p = 0; p < area; p++) {
    for (let c = 0; c < 3; c++) {
      values[c * area + p] = (pixels[p * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  return values;
}

function showWhatItSees() {
  const target = $("small");
  const context = target.getContext("2d");
  context.clearRect(0, 0, target.width, target.height);
  context.drawImage(scratch, 0, 0, target.width, target.height);
}

// ---------------------------------------------------------------------------
// Showing the answer
// ---------------------------------------------------------------------------

function drawBars(probabilities) {
  const ranked = model.labels
    .map((name, index) => ({ name, p: probabilities[index] }))
    .sort((a, b) => b.p - a.p);

  $("bars").innerHTML = ranked.map((item, place) => `
    <div class="row${place === 0 ? " top" : ""}">
      <div class="name">${item.name}</div>
      <div class="bar"><div class="fill" style="width:${(item.p * 100).toFixed(1)}%"></div></div>
      <div class="pct">${(item.p * 100).toFixed(1)}%</div>
    </div>`).join("");
}

// Running the network takes a moment. While it runs, only the newest picture
// waits its turn, so the webcam never builds up a queue.
let busy = false;
let waiting = null;

async function classify(source) {
  if (!session) return;
  if (source.videoWidth === 0) return;  // a webcam that was just turned off
  if (busy) {
    waiting = source;
    return;
  }
  busy = true;
  try {
    const input = toInput(source);
    showWhatItSees();
    drawBars(softmax(await forward(input)));
  } finally {
    busy = false;
  }
  if (waiting) {
    const next = waiting;
    waiting = null;
    classify(next);
  }
}

// ---------------------------------------------------------------------------
// The self test: does JavaScript agree with Python?
// ---------------------------------------------------------------------------

function loadImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = source;
  });
}

async function runSelfTest(cases) {
  let worst = 0;
  for (const test of cases) {
    const image = await loadImage("data:image/png;base64," + test.png);
    const got = await forward(toInput(image));
    for (let i = 0; i < got.length; i++) {
      worst = Math.max(worst, Math.abs(got[i] - test.logits[i]));
    }
  }

  const badge = $("selftest");
  if (worst < 0.02) {
    badge.className = "badge ok";
    badge.textContent =
      `self test passed: the browser agrees with Python (largest gap ${worst.toFixed(4)})`;
  } else {
    badge.className = "badge bad";
    badge.textContent =
      `self test FAILED: the browser and Python disagree by ${worst.toFixed(3)}. ` +
      `The page prepares a picture the way model.json says, and your ` +
      `prepare_image did something else. Every answer on this page is wrong ` +
      `until you fix prepare_image, train again and export again.`;
  }
}

// ---------------------------------------------------------------------------
// Wiring up the three ways of giving it a picture
// ---------------------------------------------------------------------------

function showPanel(which) {
  // A webcam left running would keep replacing the answer for an uploaded
  // picture or a drawing, so leaving its tab turns it off.
  if (which !== "Cam") cameraOff();
  for (const name of ["Cam", "File", "Draw"]) {
    $("panel" + name).hidden = name !== which;
    $("tab" + name).classList.toggle("on", name === which);
  }
}
$("tabCam").onclick = () => showPanel("Cam");
$("tabFile").onclick = () => showPanel("File");
$("tabDraw").onclick = () => showPanel("Draw");

// Webcam
const video = $("video");
let stream = null;  // the camera that is on, or null when it is off

// The browser hides the names of the cameras until the page has been allowed
// to use one, so the list is filled again after every start, and whenever a
// camera is plugged in or a virtual camera appears.
async function listCameras() {
  const picker = $("camPick");
  const chosen = picker.value;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((d) => d.kind === "videoinput" && d.deviceId);
  picker.replaceChildren(new Option("Default camera", ""));
  cameras.forEach((camera, i) => {
    picker.add(new Option(camera.label || `Camera ${i + 1}`, camera.deviceId));
  });
  if (cameras.some((camera) => camera.deviceId === chosen)) picker.value = chosen;
}

async function cameraOn() {
  const chosen = $("camPick").value;
  $("camToggle").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: chosen ? { deviceId: { exact: chosen } } : true,
    });
  } catch (error) {
    alert(window.isSecureContext
      ? `The browser would not give us the camera (${error.name}). Check that ` +
        "you allowed this page to use it, and that no other program is using it."
      : "The browser would not give us the camera.\n\n" +
        "Cameras only work on https:// pages and on http://localhost. " +
        "If you opened this file by double clicking it, run " +
        "`python -m http.server` in this folder and open " +
        "http://localhost:8000 instead.");
    return;
  } finally {
    $("camToggle").disabled = false;
  }
  // The tab may have changed while the browser was asking for permission.
  if ($("panelCam").hidden) return cameraOff();

  video.srcObject = stream;
  await video.play();
  $("camToggle").textContent = "Turn the camera off";
  await listCameras();
  $("camPick").value = stream.getVideoTracks()[0].getSettings().deviceId || "";

  const mine = stream;
  const tick = async () => {
    if (stream !== mine) return;  // turned off, or another camera took over
    await classify(video);
    setTimeout(() => requestAnimationFrame(tick), 150);
  };
  tick();
}

function cameraOff() {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
  stream = null;
  video.srcObject = null;
  $("camToggle").textContent = "Turn the camera on";
}

$("camToggle").onclick = () => (stream ? cameraOff() : cameraOn());
$("camPick").onchange = () => {
  if (stream) {
    cameraOff();
    cameraOn();
  }
};
if (navigator.mediaDevices) {
  listCameras();
  navigator.mediaDevices.addEventListener("devicechange", listCameras);
}

// Upload
$("file").onchange = async (event) => {
  const chosen = event.target.files[0];
  if (!chosen) return;
  const image = await loadImage(URL.createObjectURL(chosen));
  const preview = $("preview");
  preview.src = image.src;
  preview.hidden = false;
  preview.style.maxHeight = "240px";
  classify(image);
};

// Drawing
const pad = $("drawPad");
const padContext = pad.getContext("2d", { willReadFrequently: true });
function clearPad() {
  padContext.fillStyle = "#ffffff";
  padContext.fillRect(0, 0, pad.width, pad.height);
}
clearPad();
padContext.lineWidth = 16;
padContext.lineCap = "round";
padContext.lineJoin = "round";
padContext.strokeStyle = "#000000";

let drawing = false;
function padPoint(event) {
  const box = pad.getBoundingClientRect();
  const point = event.touches ? event.touches[0] : event;
  return {
    x: (point.clientX - box.left) * (pad.width / box.width),
    y: (point.clientY - box.top) * (pad.height / box.height),
  };
}
function startStroke(event) {
  event.preventDefault();
  drawing = true;
  const { x, y } = padPoint(event);
  padContext.beginPath();
  padContext.moveTo(x, y);
}
function continueStroke(event) {
  if (!drawing) return;
  event.preventDefault();
  const { x, y } = padPoint(event);
  padContext.lineTo(x, y);
  padContext.stroke();
  classify(pad);
}
function endStroke() {
  if (!drawing) return;
  drawing = false;
  classify(pad);
}
pad.addEventListener("pointerdown", startStroke);
pad.addEventListener("pointermove", continueStroke);
window.addEventListener("pointerup", endStroke);
$("clearPad").onclick = () => { clearPad(); };

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

async function start() {
  try {
    const modelResponse = await fetch("model.json");
    if (!modelResponse.ok) throw new Error("model.json not found");
    model = await modelResponse.json();

    $("subtitle").textContent = "Loading the network, about 45 MB. The first time takes a few seconds.";
    session = await ort.InferenceSession.create("model.onnx", {
      executionProviders: ["wasm"],
    });

    document.title = model.labels.join(" / ");
    $("title").textContent = model.labels.join("  ·  ");
    $("subtitle").textContent =
      `${model.labels.length} classes, ResNet18 on ` +
      `${model.input.crop}x${model.input.crop} colour input, ` +
      `${model.n_parameters.toLocaleString()} parameters.`;

    drawBars(model.labels.map(() => 0));

    const selfTestResponse = await fetch("selftest.json");
    if (selfTestResponse.ok) {
      await runSelfTest((await selfTestResponse.json()).cases);
    } else {
      $("selftest").textContent = "self test: selftest.json is missing";
    }
  } catch (error) {
    $("subtitle").textContent = "Could not load the model.";
    $("selftest").className = "badge bad";
    $("selftest").textContent =
      "Could not load model.json and model.onnx. Either you have not exported " +
      "a model yet, in which case run `python src/run.py` and then " +
      "`python src/export_web.py --tag run`, or you opened this file by " +
      "double clicking it, which browsers block. Run " +
      "`python -m http.server -d docs 8000` and open http://localhost:8000. " +
      "(" + error.message + ")";
  }
}

start();
