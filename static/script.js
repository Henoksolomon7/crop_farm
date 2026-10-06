// ---------- Beautiful, larger horizontal bar chart ----------
const chartData = window.CHART_DATA || [];

// Sort descending (largest on top)
const sorted = [...chartData].sort((a, b) => b.value - a.value);

const labels = sorted.map(d => d.label);
const values = sorted.map(d => d.value);

// Nice gradient palette (green → teal → amber accent at the top)
const palette = [
    "#134e3a", "#1b5e47", "#256e54", "#2f7f60", "#3a916d",
    "#4aa37c", "#5cb58c", "#71c69d", "#88d6af", "#a2e4c2",
    "#bfeed3", "#d9f5e4", "#eefaf2",
];

const backgroundColors = sorted.map((_, i) => {
    // If you want the smallest bar to stand out, tint the last one
    if (i === sorted.length - 1) return "#f4a261"; // soft amber accent
    return palette[Math.min(i, palette.length - 1)];
});

const ctx = document.getElementById("importanceChart").getContext("2d");

// Custom plugin: draw value labels at the end of each bar
const valueLabelPlugin = {
    id: "valueLabel",
    afterDatasetsDraw(chart) {
        const { ctx } = chart;
        const meta = chart.getDatasetMeta(0);
        ctx.save();
        ctx.font = "600 12px Inter, system-ui, sans-serif";
        ctx.fillStyle = "#1e4b3c";
        ctx.textBaseline = "middle";
        meta.data.forEach((bar, i) => {
            const val = (values[i] * 100).toFixed(1) + "%";
            ctx.fillText(val, bar.x + 8, bar.y);
        });
        ctx.restore();
    },
};

new Chart(ctx, {
    type: "bar",
    data: {
        labels,
        datasets: [
            {
                label: "Relative influence",
                data: values,
                backgroundColor: backgroundColors,
                hoverBackgroundColor: "#f4a261",
                borderRadius: { topRight: 10, bottomRight: 10, topLeft: 4, bottomLeft: 4 },
                borderSkipped: false,
                barPercentage: 0.82,
                categoryPercentage: 0.9,
            },
        ],
    },
    options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        layout: {
            padding: { right: 45, left: 6, top: 8, bottom: 8 },
        },
        animation: {
            duration: 900,
            easing: "easeOutQuart",
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: "#0d2e21",
                titleColor: "#d9f5e4",
                bodyColor: "#eefaf2",
                padding: 12,
                cornerRadius: 10,
                displayColors: false,
                titleFont: { size: 13, weight: "600" },
                bodyFont: { size: 12 },
                callbacks: {
                    title: (items) => items[0].label,
                    label: (item) => `  Influence: ${(item.raw * 100).toFixed(2)}%`,
                },
            },
        },
        scales: {
            x: {
                beginAtZero: true,
                grid: { color: "#e6f0eb", drawBorder: false },
                border: { display: false },
                ticks: {
                    color: "#7a95a5",
                    font: { size: 11 },
                    callback: (v) => (v * 100).toFixed(0) + "%",
                    maxTicksLimit: 6,
                },
            },
            y: {
                grid: { display: false },
                border: { display: false },
                ticks: {
                    color: "#1e4b3c",
                    font: { size: 12, weight: "600" },
                    padding: 8,
                },
            },
        },
    },
    plugins: [valueLabelPlugin],
});

// ---------- Prediction (unchanged) ----------
const form = document.getElementById("predictForm");
const predictBtn = document.getElementById("predictBtn");
const errorMsg = document.getElementById("errorMsg");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContainer = document.getElementById("resultContainer");
const predictedYield = document.getElementById("predictedYield");
const yieldNote = document.getElementById("yieldNote");

function collectFormData() {
    const fd = new FormData(form);
    const data = {};
    fd.forEach((v, k) => (data[k] = v));
    return data;
}

predictBtn.addEventListener("click", async () => {
    errorMsg.style.display = "none";
    predictBtn.disabled = true;
    predictBtn.textContent = "⏳ Predicting…";

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(collectFormData()),
        });
        const json = await res.json();
        if (!json.success) throw new Error(json.error || "Prediction failed");

        resultPlaceholder.style.display = "none";
        resultContainer.style.display = "block";
        predictedYield.innerHTML =
            `${json.yield_t_per_ha.toFixed(2)} <span class="result-unit">t/ha</span>`;

        const d = collectFormData();
        yieldNote.textContent =
            `${d.crop_type} · ${d.region} · ${d.planting_month} ${d.survey_year} · ` +
            `${json.yield_kg_per_ha.toFixed(0)} kg/ha`;
    } catch (e) {
        errorMsg.textContent = e.message;
        errorMsg.style.display = "block";
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "⚡ Predict yield";
    }
});