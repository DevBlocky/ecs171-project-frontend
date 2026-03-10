import { useEffect, useMemo, useState } from "react";
import "./styles.css";

type ModelSummary = {
  id: string;
  label: string;
};

type PredictionResult = {
  label: string;
  defaults: boolean;
  confidence: number;
};

type SampleInputResponse = {
  source: string;
  row_index: number;
  target: number;
  record: Record<string, number>;
};

type PresetInput = {
  name: string;
  label: string;
  source: string;
  row_index: number;
  target: number;
  reference_model: string;
  reference_probability: number;
  explanation: string;
  record: Record<string, number>;
};

type FeatureValue = number | "";

type FormState = Record<string, FeatureValue>;

type SelectOption = {
  value: number;
  label: string;
};

type NumberField = {
  key: string;
  label: string;
  type: "number";
};

type SelectField = {
  key: string;
  label: string;
  type: "select";
  options: SelectOption[];
};

type FieldDefinition = NumberField | SelectField;

type FeatureGroup = {
  title: string;
  tooltip: string;
  fields: FieldDefinition[];
};

const featureGroups: FeatureGroup[] = [
  {
    title: "Profile",
    tooltip:
      "Customer demographics and account context. LIMIT_BAL is the credit limit, while SEX, EDUCATION, MARRIAGE, and AGE provide profile information.",
    fields: [
      { key: "LIMIT_BAL", label: "Credit Limit", type: "number" },
      { key: "AGE", label: "Age", type: "number" },
      {
        key: "SEX",
        label: "Sex",
        type: "select",
        options: [
          { value: 1, label: "Male" },
          { value: 2, label: "Female" },
        ],
      },
      {
        key: "EDUCATION",
        label: "Education",
        type: "select",
        options: [
          { value: 0, label: "Unknown" },
          { value: 1, label: "Graduate School" },
          { value: 2, label: "University" },
          { value: 3, label: "High School" },
          { value: 4, label: "Other" },
          { value: 5, label: "Other" },
          { value: 6, label: "Other" },
        ],
      },
      {
        key: "MARRIAGE",
        label: "Marriage",
        type: "select",
        options: [
          { value: 0, label: "Unknown" },
          { value: 1, label: "Married" },
          { value: 2, label: "Single" },
          { value: 3, label: "Other" },
        ],
      },
    ],
  },
  {
    title: "Repayment Status",
    tooltip:
      "Monthly repayment status for recent months. Positive PAY values indicate delayed payment, while 0 or negative values indicate on-time or early payment.",
    fields: ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"].map((key) => ({
      key,
      label: key,
      type: "number",
    })),
  },
  {
    title: "Bill Amounts",
    tooltip:
      "Credit card statement balances for the last six months. Higher balances can signal heavier utilization and repayment pressure.",
    fields: [1, 2, 3, 4, 5, 6].map((index) => ({
      key: `BILL_AMT${index}`,
      label: `Bill ${index}`,
      type: "number",
    })),
  },
  {
    title: "Payment Amounts",
    tooltip:
      "Actual payment amounts made across the last six months. These help show repayment behavior relative to the billed balances.",
    fields: [1, 2, 3, 4, 5, 6].map((index) => ({
      key: `PAY_AMT${index}`,
      label: `Payment ${index}`,
      type: "number",
    })),
  },
];

const featureKeys = featureGroups.flatMap((group) =>
  group.fields.map((field) => field.key),
);

const emptyFormState = featureKeys.reduce<FormState>((acc, key) => {
  acc[key] = "";
  return acc;
}, {});

const fallbackPresets: PresetInput[] = [
  {
    name: "high_risk",
    label: "High Risk",
    source: "Manual fallback",
    row_index: -1,
    target: 1,
    reference_model: "Fallback",
    reference_probability: 0.8,
    explanation: "Fallback preset used because grounded dataset presets could not be loaded.",
    record: {
      LIMIT_BAL: 20000,
      SEX: 2,
      EDUCATION: 2,
      MARRIAGE: 1,
      AGE: 24,
      PAY_0: 2,
      PAY_2: 2,
      PAY_3: -1,
      PAY_4: -1,
      PAY_5: -2,
      PAY_6: -2,
      BILL_AMT1: 3913,
      BILL_AMT2: 3102,
      BILL_AMT3: 689,
      BILL_AMT4: 0,
      BILL_AMT5: 0,
      BILL_AMT6: 0,
      PAY_AMT1: 0,
      PAY_AMT2: 689,
      PAY_AMT3: 0,
      PAY_AMT4: 0,
      PAY_AMT5: 0,
      PAY_AMT6: 0,
    },
  },
  {
    name: "borderline",
    label: "Borderline",
    source: "Manual fallback",
    row_index: -1,
    target: 0,
    reference_model: "Fallback",
    reference_probability: 0.5,
    explanation: "Fallback preset used because grounded dataset presets could not be loaded.",
    record: {
      LIMIT_BAL: 120000,
      SEX: 2,
      EDUCATION: 2,
      MARRIAGE: 2,
      AGE: 26,
      PAY_0: -1,
      PAY_2: 2,
      PAY_3: 0,
      PAY_4: 0,
      PAY_5: 0,
      PAY_6: 2,
      BILL_AMT1: 2682,
      BILL_AMT2: 1725,
      BILL_AMT3: 2682,
      BILL_AMT4: 3272,
      BILL_AMT5: 3455,
      BILL_AMT6: 3261,
      PAY_AMT1: 0,
      PAY_AMT2: 1000,
      PAY_AMT3: 1000,
      PAY_AMT4: 1000,
      PAY_AMT5: 0,
      PAY_AMT6: 2000,
    },
  },
  {
    name: "lower_risk",
    label: "Lower Risk",
    source: "Manual fallback",
    row_index: -1,
    target: 0,
    reference_model: "Fallback",
    reference_probability: 0.1,
    explanation: "Fallback preset used because grounded dataset presets could not be loaded.",
    record: {
      LIMIT_BAL: 90000,
      SEX: 2,
      EDUCATION: 2,
      MARRIAGE: 2,
      AGE: 34,
      PAY_0: 0,
      PAY_2: 0,
      PAY_3: 0,
      PAY_4: 0,
      PAY_5: 0,
      PAY_6: 0,
      BILL_AMT1: 29239,
      BILL_AMT2: 14027,
      BILL_AMT3: 13559,
      BILL_AMT4: 14331,
      BILL_AMT5: 14948,
      BILL_AMT6: 15549,
      PAY_AMT1: 1518,
      PAY_AMT2: 1500,
      PAY_AMT3: 1000,
      PAY_AMT4: 1000,
      PAY_AMT5: 1000,
      PAY_AMT6: 5000,
    },
  },
];

const formatConfidence = (value: number) => `${(value * 100).toFixed(1)}%`;

export default function App() {
  const [formState, setFormState] = useState<FormState>(emptyFormState);
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [modelSummaries, setModelSummaries] = useState<ModelSummary[]>([]);
  const [presets, setPresets] = useState<PresetInput[]>(fallbackPresets);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [sampleMessage, setSampleMessage] = useState("");

  useEffect(() => {
    const loadModels = async () => {
      try {
        const response = await fetch("/models");
        if (!response.ok) {
          throw new Error("Unable to load model list.");
        }
        const data = (await response.json()) as ModelSummary[];
        setModelSummaries(data);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unable to load model list.";
        setErrorMessage(message);
      }
    };

    void loadModels();
  }, []);

  useEffect(() => {
    const loadPresets = async () => {
      try {
        const response = await fetch("/preset-inputs");
        if (!response.ok) {
          throw new Error("Unable to load grounded presets.");
        }
        const data = (await response.json()) as PresetInput[];
        if (data.length > 0) {
          setPresets(data);
        }
      } catch {
        setPresets(fallbackPresets);
      }
    };

    void loadPresets();
  }, []);

  const completedFieldCount = useMemo(
    () => Object.values(formState).filter((value) => value !== "").length,
    [formState],
  );

  const updateField = (key: string, value: FeatureValue) => {
    setFormState((current) => ({ ...current, [key]: value }));
  };

  const loadPreset = (preset: PresetInput) => {
    setFormState((current) => ({
      ...current,
      ...preset.record,
    }));
    setErrorMessage("");
    setSampleMessage(
      `${preset.label} preset grounded in ${preset.source}` +
        (preset.row_index >= 0 ? ` row ${preset.row_index}. ` : ". ") +
        `${preset.explanation}`,
    );
  };

  const resetForm = () => {
    setFormState(emptyFormState);
    setResults([]);
    setErrorMessage("");
    setSampleMessage("");
  };

  const fillRandomSample = async () => {
    setErrorMessage("");
    try {
      const response = await fetch("/sample-input");
      if (!response.ok) {
        throw new Error("Unable to load a random dataset sample.");
      }
      const data = (await response.json()) as SampleInputResponse;
      setFormState((current) => ({
        ...current,
        ...data.record,
      }));
      setSampleMessage(
        `Random sample loaded from ${data.source} row ${data.row_index}. Actual outcome: ${
          data.target === 1 ? "Default" : "No Default"
        }.`,
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Unable to load a random dataset sample.";
      setErrorMessage(message);
    }
  };

  const runPrediction = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setErrorMessage("");

    const payload = Object.fromEntries(
      featureKeys.map((key) => [key, Number(formState[key] ?? 0)]),
    );

    try {
      const response = await fetch("/models", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }
      const data = (await response.json()) as PredictionResult[];
      setResults(data);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Prediction request failed.";
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">ECS171 Team 21</p>
        <h1>Credit Default Prediction Demo</h1>
        <p className="hero-copy">
          Compare model outputs on the UCI credit default dataset using the
          same borrower profile and repayment history.
        </p>
      </header>

      <main className="layout">
        <section className="panel form-panel">
          <div className="panel-header panel-header-stacked">
            <div>
              <h2>Borrower Input</h2>
              <p>
                Load a preset or enter values manually. The form maps directly
                to the dataset fields used by the backend.
              </p>
            </div>
            <div className="field-count field-count-row">
              <span>Filled fields</span>
              <strong>
                {completedFieldCount}/{featureKeys.length}
              </strong>
            </div>
          </div>

          <div className="preset-row">
            <div className="preset-actions">
              {presets.map((preset) => (
                <button
                  key={preset.name}
                  className="preset-button"
                  type="button"
                  onClick={() => loadPreset(preset)}
                >
                  {preset.label}
                </button>
              ))}
              <button
                className="preset-button accent-button"
                type="button"
                onClick={fillRandomSample}
              >
                Random Sample
              </button>
              <button className="ghost-button" type="button" onClick={resetForm}>
                Clear
              </button>
            </div>
            <button
              className="primary-button top-run-button"
              type="submit"
              form="prediction-form"
              disabled={isLoading}
            >
              {isLoading ? "Running..." : "Run Prediction"}
            </button>
          </div>
          {sampleMessage ? <p className="sample-banner">{sampleMessage}</p> : null}

          <form className="input-form" id="prediction-form" onSubmit={runPrediction}>
            {featureGroups.map((group) => (
              <fieldset key={group.title} className="group-card">
                <legend className="group-legend">
                  <span>{group.title}</span>
                  <span className="tooltip-wrap">
                    <span className="tooltip-trigger">?</span>
                    <span className="tooltip-bubble">{group.tooltip}</span>
                  </span>
                </legend>
                <div className="field-grid">
                  {group.fields.map((field) => (
                    <label key={field.key} className="field">
                      <span>{field.label}</span>
                      {field.type === "select" ? (
                        <select
                          value={formState[field.key]}
                          onChange={(event) =>
                            updateField(
                              field.key,
                              event.target.value === ""
                                ? ""
                                : Number(event.target.value),
                            )
                          }
                        >
                          <option value="">Select</option>
                          {field.options?.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type="number"
                          step="1"
                          value={formState[field.key]}
                          onChange={(event) =>
                            updateField(
                              field.key,
                              event.target.value === ""
                                ? ""
                                : Number(event.target.value),
                            )
                          }
                        />
                      )}
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}

          </form>
        </section>

        <aside className="sidebar-column">
          <div className="sidebar-stack">
            <section className="panel result-panel">
              <div className="panel-header">
                <div>
                  <h2>Model Outputs</h2>
                  <p>
                    Results are stacked by model so you can compare the same input
                    across every registered predictor.
                  </p>
                </div>
              </div>

              <div className="result-summary">
                <div className="result-summary-grid">
                  <div>
                    <span className="metric-label">Dataset</span>
                    <strong>30,000 rows</strong>
                  </div>
                  <div>
                    <span className="metric-label">Raw Predictors</span>
                    <strong>23 features</strong>
                  </div>
                  <div>
                    <span className="metric-label">Target</span>
                    <strong>Default next month</strong>
                  </div>
                  <div>
                    <span className="metric-label">Current API Models</span>
                    <strong>{modelSummaries.length || 1}</strong>
                  </div>
                  <div className="result-summary-models">
                    <span className="metric-label">Available models</span>
                    <div className="model-list">
                      {modelSummaries.length > 0 ? (
                        modelSummaries.map((model) => (
                          <strong key={model.id}>{model.label}</strong>
                        ))
                      ) : (
                        <strong>Loading...</strong>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}

              <div className="table-shell">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Prediction</th>
                      <th>Confidence</th>
                      <th>Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.length > 0 ? (
                      results.map((result) => (
                        <tr key={result.label}>
                          <td>{result.label}</td>
                          <td>{result.defaults ? "Default" : "No Default"}</td>
                          <td>{formatConfidence(result.confidence)}</td>
                          <td>
                            <span
                              className={
                                result.defaults ? "risk-chip danger" : "risk-chip safe"
                              }
                            >
                              {result.defaults ? "High" : "Lower"}
                            </span>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={4} className="empty-state">
                          Run a prediction to populate the comparison table.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

            </section>
          </div>
        </aside>
      </main>
    </div>
  );
}
