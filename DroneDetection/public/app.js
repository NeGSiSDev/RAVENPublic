// =====================
// Configuration
// =====================

const MODEL_URL = "./yolov8.onnx";  // Hugging Face drone detection model
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.4;
const NMS_THRESHOLD = 0.5;

// ⚠️ CLASS ORDER - Based on actual testing of the model
// Model output: [1, 9, 8400] = 4 bbox coords + 5 class scores
// Drone detections appear at class index [1], not [0]

const CLASSES = [
  "unknown",  // class 0
  "drone",    // class 1 - ✓ ACTUAL DRONE CLASS
  "unknown",  // class 2
  "unknown",  // class 3
  "unknown"   // class 4
];

// Class colors for visualization
const CLASS_COLORS = [
  "#666666",  // unknown - gray
  "#00ff00",  // drone - green
  "#666666",  // unknown - gray
  "#666666",  // unknown - gray
  "#666666"   // unknown - gray
];

let session = null;

// =====================
// UI Utilities
// =====================

function setStatus(message, type = 'loading') {
  const status = document.getElementById('status');
  status.textContent = message;
  status.className = `status ${type}`;
}

function updateStats(detections, inferenceTime, imgWidth, imgHeight) {
  document.getElementById('detectionCount').textContent = detections.length;
  document.getElementById('inferenceTime').textContent = `${inferenceTime}ms`;
  document.getElementById('imageSize').textContent = `${imgWidth}×${imgHeight}`;
  document.getElementById('statsGrid').style.display = 'grid';
}

function showCanvas() {
  const canvas = document.getElementById('canvas');
  const placeholder = document.getElementById('placeholder');
  const wrapper = document.getElementById('canvasWrapper');
  
  canvas.style.display = 'block';
  placeholder.style.display = 'none';
  wrapper.classList.add('has-image');
}

// =====================
// Math Utilities
// =====================

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Intersection over Union
function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);

  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const boxAArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const boxBArea = (b.x2 - b.x1) * (b.y2 - b.y1);

  return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

// Non-Max Suppression
function nms(boxes, threshold) {
  boxes.sort((a, b) => b.score - a.score);
  const result = [];

  while (boxes.length > 0) {
    const current = boxes.shift();
    result.push(current);
    boxes = boxes.filter(b => iou(current, b) < threshold);
  }

  return result;
}

// =====================
// Model loading
// =====================

async function loadModel() {
  try {
    setStatus('Loading model...', 'loading');
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"]
    });
    console.log("Model loaded successfully");
    console.log("Input:", session.inputNames);
    console.log("Output:", session.outputNames);
    setStatus('✓ Model ready - upload an image to begin', 'ready');
  } catch (error) {
    console.error("Model load error:", error);
    setStatus('✗ Failed to load model: ' + error.message, 'error');
  }
}

// =====================
// Image preprocessing
// =====================

function preprocess(img) {
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;

  const ctx = canvas.getContext("2d");

  // Letterbox resize
  const scale = Math.min(
    INPUT_SIZE / img.width,
    INPUT_SIZE / img.height
  );

  const newW = Math.round(img.width * scale);
  const newH = Math.round(img.height * scale);
  const padX = Math.round((INPUT_SIZE - newW) / 2);
  const padY = Math.round((INPUT_SIZE - newH) / 2);

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  ctx.drawImage(img, padX, padY, newW, newH);

  const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
  const input = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

  // Normalize to [0, 1] and convert to CHW format
  for (let i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) {
    input[i] = imageData[i * 4] / 255.0;                                  // R
    input[i + INPUT_SIZE * INPUT_SIZE] = imageData[i * 4 + 1] / 255.0;   // G
    input[i + 2 * INPUT_SIZE * INPUT_SIZE] = imageData[i * 4 + 2] / 255.0; // B
  }

  return {
    tensor: new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale,
    padX,
    padY
  };
}

// =====================
// YOLOv8 decoding (FIXED)
// =====================

function decodeYOLO(output, imgWidth, imgHeight, scale, padX, padY) {
  const data = output.data;
  const dims = output.dims;
  
  console.log("Output shape:", dims);
  console.log("First 20 values:", Array.from(data.slice(0, 20)));

  // YOLOv8 output format: [1, 84, 8400] or [1, numClasses+4, numPredictions]
  // We need to transpose this to [numPredictions, numClasses+4]
  
  let batch, numAttrs, numPreds;
  
  if (dims.length === 3) {
    [batch, numAttrs, numPreds] = dims;
  } else {
    console.error("Unexpected output dimensions:", dims);
    return [];
  }

  const numClasses = numAttrs - 4; // First 4 are bbox coords
  const detections = [];

  for (let i = 0; i < numPreds; i++) {
    // YOLOv8 format: data is stored as [attr][prediction]
    // So for prediction i, attributes are at indices: i, i+numPreds, i+2*numPreds, etc.
    
    const cx = data[i];
    const cy = data[i + numPreds];
    const w = data[i + 2 * numPreds];
    const h = data[i + 3 * numPreds];

    // Find best class and collect all scores for debugging
    let bestScore = 0;
    let bestClass = -1;
    const classScores = [];

    for (let c = 0; c < Math.min(numClasses, CLASSES.length); c++) {
      const score = data[i + (4 + c) * numPreds];
      classScores.push(score);
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }

    if (bestScore < CONF_THRESHOLD) continue;
    
    // Debug: log class scores for first few detections
    if (detections.length < 3) {
      console.log(`Detection ${detections.length + 1} at (${cx.toFixed(1)}, ${cy.toFixed(1)}):`, 
        classScores.map((s, idx) => `${CLASSES[idx]}=${(s*100).toFixed(1)}%`).join(', '));
    }

    // Convert from center/width/height to corner coordinates
    // These are already in pixel space relative to INPUT_SIZE
    const x1_model = cx - w / 2;
    const y1_model = cy - h / 2;
    const x2_model = cx + w / 2;
    const y2_model = cy + h / 2;

    // Map from letterboxed space back to original image space
    const x1 = (x1_model - padX) / scale;
    const y1 = (y1_model - padY) / scale;
    const x2 = (x2_model - padX) / scale;
    const y2 = (y2_model - padY) / scale;

    // Clamp to image boundaries
    detections.push({
      x1: Math.max(0, Math.min(imgWidth, x1)),
      y1: Math.max(0, Math.min(imgHeight, y1)),
      x2: Math.max(0, Math.min(imgWidth, x2)),
      y2: Math.max(0, Math.min(imgHeight, y2)),
      score: bestScore,
      classId: bestClass
    });
  }

  console.log(`Found ${detections.length} detections before NMS`);
  const nmsResults = nms(detections, NMS_THRESHOLD);
  console.log(`${nmsResults.length} detections after NMS`);
  
  return nmsResults;
}

// =====================
// Drawing (IMPROVED)
// =====================

function drawDetections(img, detections) {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");

  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  ctx.lineWidth = 3;
  ctx.font = "bold 16px sans-serif";

  detections.forEach(det => {
    const { x1, y1, x2, y2, score, classId } = det;
    const color = CLASS_COLORS[classId] || "#00ff00";
    const className = CLASSES[classId] || 'drone';

    // Draw bounding box
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Draw label background
    const pct = (score * 100).toFixed(1);
    const label = `${className} ${pct}%`;
    
    ctx.font = "bold 16px sans-serif";
    const textMetrics = ctx.measureText(label);
    const textHeight = 20;
    const padding = 4;

    const labelY = Math.max(y1 - textHeight - padding, textHeight);
    
    ctx.fillStyle = color;
    ctx.fillRect(
      x1, 
      labelY - textHeight, 
      textMetrics.width + padding * 2, 
      textHeight + padding
    );

    // Draw label text
    ctx.fillStyle = "white";
    ctx.fillText(label, x1 + padding, labelY - padding);
  });

  // Show canvas, hide placeholder
  showCanvas();
}

// =====================
// Inference
// =====================

async function runInference(img) {
  if (!session) {
    setStatus('Model not loaded yet', 'error');
    return;
  }

  try {
    setStatus('Running inference...', 'loading');
    
    const startTime = performance.now();
    const prep = preprocess(img);
    const feeds = { images: prep.tensor };

    const results = await session.run(feeds);
    const output = results.output0;

    const detections = decodeYOLO(
      output,
      img.width,
      img.height,
      prep.scale,
      prep.padX,
      prep.padY
    );

    const inferenceTime = Math.round(performance.now() - startTime);

    drawDetections(img, detections);
    updateStats(detections, inferenceTime, img.width, img.height);

    if (detections.length === 0) {
      setStatus('No objects detected', 'ready');
    } else {
      setStatus(`✓ Detected ${detections.length} object(s)`, 'ready');
    }

  } catch (error) {
    console.error("Inference error:", error);
    setStatus('Inference failed: ' + error.message, 'error');
  }
}

// =====================
// UI wiring
// =====================

document.getElementById("imageInput").addEventListener("change", e => {
  const file = e.target.files[0];
  if (!file) return;

  const img = new Image();
  img.onload = () => runInference(img);
  img.onerror = () => setStatus('Failed to load image', 'error');
  img.src = URL.createObjectURL(file);
});

// =====================
// Start
// =====================

console.log("crossOriginIsolated:", crossOriginIsolated);
console.log("Cores:", navigator.hardwareConcurrency);

ort.env.wasm.simd = true;
ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

console.log("SIMD enabled:", ort.env.wasm.simd);
console.log("Threads:", ort.env.wasm.numThreads);


loadModel();