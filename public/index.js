const features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'];
const presets = [
    {
        name: 'francis_lee',
        label: 'Francis Lee',
        data: { "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2, "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0 }
    },
    {
        name: 'john_penn',
        label: 'John Penn',
        data: { "LIMIT_BAL": 120000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 26, "PAY_0": -1, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 2, "BILL_AMT1": 2682, "BILL_AMT2": 1725, "BILL_AMT3": 2682, "BILL_AMT4": 3272, "BILL_AMT5": 3455, "BILL_AMT6": 3261, "PAY_AMT1": 0, "PAY_AMT2": 1000, "PAY_AMT3": 1000, "PAY_AMT4": 1000, "PAY_AMT5": 0, "PAY_AMT6": 2000 },
    },
    {
        name: 'samuel_adams',
        label: 'Samuel Adams',
        data: { "LIMIT_BAL": 90000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 34, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 29239, "BILL_AMT2": 14027, "BILL_AMT3": 13559, "BILL_AMT4": 14331, "BILL_AMT5": 14948, "BILL_AMT6": 15549, "PAY_AMT1": 1518, "PAY_AMT2": 1500, "PAY_AMT3": 1000, "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 5000 }
    }
];

function loadPreset(name) {
    const preset = presets.find(x => x.name === name);
    if (!preset) throw `invalid preset name: ${name}`;

    for (const key in preset.data) {
        const el = document.getElementById(key);
        if (el) el.value = String(preset.data[key]);
    }
}

let askInProgress = false;
async function askModels(e) {
    e.preventDefault();
    if (askInProgress) return;
    askInProgress = true;

    const X = {};
    for (const feature of features) {
        const el = document.getElementById(feature);
        X[feature] = el ? Number(el.value) : 0;
    }

    try {
        const res = await fetch('/models', {
            method: 'POST',
            body: JSON.stringify(X),
            headers: { 'Content-Type': 'application/json' },
        });
        const json = await res.json();
        renderPredictions(json);
    } finally {
        askInProgress = false;
    }
}
function renderPredictions(results) {
    const predictionsEl = document.getElementById('predictions');
    if (!predictionsEl) return;

    predictionsEl.innerHTML = '';

    for (const result of results) {
        const confidence = Number(result.confidence) || 0;
        const confidencePct = (confidence * 100).toFixed(1);
        const card = document.createElement('article');
        card.className = 'result-card';

        card.innerHTML = `
            <h3>${result.label}</h3>
            <p><strong>Prediction:</strong> ${result.defaults ? 'Defaults' : 'No Default'}</p>
            <p><strong>Confidence:</strong> ${confidencePct}%</p>
            <div class="meter">
                <div class="meter-fill" style="width: ${confidencePct}%"></div>
            </div>
        `;

        predictionsEl.appendChild(card);
    }
}

window.onload = () => {
    const el = document.getElementById('preset_buttons');
    for (const preset of presets) {
        const button = document.createElement('button');
        button.innerText = preset.label;
        button.onclick = () => loadPreset(preset.name);
        el.appendChild(button);
    }
};
