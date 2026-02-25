const cards = Array.from(document.querySelectorAll(".card"));
const container = document.querySelector(".card-container");
const nextBtn = document.querySelector(".next");
const prevBtn = document.querySelector(".prev");

let current = 0;
let cardWidth = 0;
let autoSlideInterval = null;
let isInteracting = false;

function recalcCardWidth() {
  if (!cards.length) return;
  const rect = cards[0].getBoundingClientRect();
  const style = getComputedStyle(cards[0]);
  const margin = parseFloat(style.marginLeft) + parseFloat(style.marginRight);
  cardWidth = rect.width + margin;
}

function showCard(index) {
  current = ((index % cards.length) + cards.length) % cards.length;
  container.style.transform = `translateX(-${current * cardWidth}px)`;
}

function startAutoSlide() {
  if (autoSlideInterval) return;
  autoSlideInterval = setInterval(() => {
    if (!isInteracting) showCard(current + 1);
  }, 3000);
}

function stopAutoSlide() {
  if (autoSlideInterval) {
    clearInterval(autoSlideInterval);
    autoSlideInterval = null;
  }
}

if (nextBtn) {
  nextBtn.addEventListener("click", () => {
    isInteracting = true;
    stopAutoSlide();
    showCard(current + 1);
    setTimeout(() => {
      isInteracting = false;
      startAutoSlide();
    }, 1000);
  });
}

if (prevBtn) {
  prevBtn.addEventListener("click", () => {
    isInteracting = true;
    stopAutoSlide();
    showCard(current - 1);
    setTimeout(() => {
      isInteracting = false;
      startAutoSlide();
    }, 1000);
  });
}

window.addEventListener("resize", () => {
  recalcCardWidth();
  showCard(current);
});

window.addEventListener("load", () => {
  recalcCardWidth();
  showCard(current);
  startAutoSlide();
});

let touchStartX = 0;
container?.addEventListener("touchstart", (e) => {
  touchStartX = e.touches[0].clientX;
  stopAutoSlide();
});
container?.addEventListener("touchend", (e) => {
  const diff = touchStartX - e.changedTouches[0].clientX;
  if (Math.abs(diff) > 50) showCard(current + (diff > 0 ? 1 : -1));
  startAutoSlide();
});

const API_BASE_URL = "http://127.0.0.1:5000/api"; 
let predictionInterval = null;
let isMonitoring = false;

const statusEl = document.getElementById("connectionStatus");
const rateEl = document.getElementById("connectionRate");
const labelEl = document.getElementById("connectionStatusText");
const checkBtn = document.getElementById("checkBtn");
const stopBtn = document.getElementById("stopBtn");

async function startMonitoring() {
  try {
    const staleInput = document.getElementById("staleInput");
    let staleVal = null;
    if (staleInput) {
      staleVal = parseFloat(staleInput.value);
      if (Number.isNaN(staleVal) || staleVal <= 0) staleVal = null;
    }

    const startUrl = staleVal
      ? `${API_BASE_URL}/start?stale=${encodeURIComponent(staleVal)}`
      : `${API_BASE_URL}/start`;
    const res = await fetch(startUrl);
    const data = await res.json();

    if (data.status === "success" || data.status === "running") {
      isMonitoring = true;
      checkBtn.textContent = "Monitoring Live";
      checkBtn.style.backgroundColor = "#28a745";
      checkBtn.disabled = true;
      statusEl.textContent = "Monitoring Active";
      statusEl.style.color = "#28a745";
      updatePrediction();
      predictionInterval = setInterval(updatePrediction, 500);  
    }
  } catch (err) {
    alert("السيرفر مش شغال! شغل: python api.py");
    resetButton();
  }
}

async function stopMonitoring(manualTrigger = false) {
  if (!isMonitoring && !manualTrigger) return;
  isMonitoring = false;
  if (predictionInterval) {
    clearInterval(predictionInterval);
    predictionInterval = null;
  }
  try {
    await fetch(`${API_BASE_URL}/stop`);
  } catch (err) {
    console.log("Unable to reach stop endpoint, assuming stopped.");
  } finally {
    resetButton();
    hideStopButton();
    if (stopBtn) {
      stopBtn.disabled = false;
      stopBtn.textContent = "Stop Monitoring";
    }
  }
}

async function updatePrediction() {
  if (!isMonitoring) return;

  try {
    const res = await fetch(`${API_BASE_URL}/predict`);
    const data = await res.json();

    if (data.status === "success") {
      rateEl.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
      labelEl.textContent = data.label;
      labelEl.style.color = data.is_attack ? "#dc3545" : "#28a745";
      statusEl.textContent = data.is_attack
        ? "DDoS Attack Detected!"
        : "Normal Traffic - Safe";
      statusEl.style.color = data.is_attack ? "#dc3545" : "#28a745";
      showStopButton();

      if (data.is_attack) {
        document.body.style.backgroundColor = "#ff002233";
        setTimeout(() => (document.body.style.backgroundColor = ""), 400);
      }
    } else if (
      data.status === "collecting" ||
      data.status === "stale" ||
      data.status === "waiting_for_new"
    ) {

      if (data.status === "stale") {
        statusEl.textContent = `No recent traffic (last ${
          data.age || "?"
        }s). Waiting...`;
      } else if (data.status === "waiting_for_new") {
        statusEl.textContent = `Waiting for new data (last update ${
          data.age || "?"
        }s ago)`;
      } else {
        statusEl.textContent = "Collecting traffic data...";
      }
      rateEl.textContent = "Confidence: --%";
      labelEl.textContent = data.label || "Waiting...";
      labelEl.style.color = "#999";
      showStopButton();
    }
  } catch (err) {
    console.log("Waiting for data...");
  }
}

function resetButton() {
  checkBtn.disabled = false;
  checkBtn.textContent = "Check Connection";
  checkBtn.style.backgroundColor = "";
  statusEl.textContent = "Not Connected";
  statusEl.style.color = "";
  rateEl.textContent = "Confidence: --%";
  labelEl.textContent = "Waiting...";
  hideStopButton();
}

checkBtn?.addEventListener("click", () => {
  if (isMonitoring) return;
  checkBtn.textContent = "Connecting...";
  checkBtn.disabled = true;
  startMonitoring();
});

stopBtn?.addEventListener("click", () => {
  if (!isMonitoring) return;
  stopBtn.disabled = true;
  stopBtn.textContent = "Stopping...";
  stopMonitoring(true);
});

function showStopButton() {
  if (!stopBtn) return;
  stopBtn.classList.add("visible");
}

function hideStopButton() {
  if (!stopBtn) return;
  stopBtn.classList.remove("visible");
}

window.addEventListener("beforeunload", () => {
  if (predictionInterval) clearInterval(predictionInterval);
  fetch(`${API_BASE_URL}/stop`).catch(() => {});
});