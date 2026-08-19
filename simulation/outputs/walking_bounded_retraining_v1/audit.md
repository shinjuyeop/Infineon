# Walking-domain bounded float retraining v1

The existing deployable Terrain 50 ms, Slip 5 ms, and Sink 20 ms architectures
were retained.  Development used existing train/validation sources plus only
d2209cd v00/v01 training and v02 walking validation.  All candidate selections
were written and hashed before fd5b9f0 v03/v04/v05 or the new Sink holdout was
read or generated.

Selected candidates:

```json
{
  "terrain": {
    "detector": "terrain",
    "selection_data": "existing validation + d2209cd v02 only",
    "holdout_runs_accessed_before_selection": 0,
    "architecture_unchanged": true,
    "window_ms": 50,
    "mixture_ratio": 1.0,
    "training_seed": 20261001,
    "model_path": "models/terrain_walking_candidate.keras",
    "model_sha256": "49ff0672b3a0374e8e05756e6d39a93c33e0380c6cce055a8ccc9d4a6d7732c1",
    "normalization_path": "normalization/terrain.json",
    "normalization_sha256": "43df9c0e379c04827231b71b4e5b72dad7049adb5902d699ed34766ab4a002c6",
    "parameters": 1272,
    "candidate_gate_pass": true,
    "selection_rationale": "all mandatory gates, then walking normal FPR, walking positive recall, static/controlled retention, p95 latency, unchanged model size, deterministic ties",
    "production_artifact_replaced": false,
    "selected_probability_threshold": null,
    "selected_runtime_persistence": 3,
    "static_validation_accuracy": 0.954070981210856,
    "walking_validation_accuracy": 0.4751513622603431,
    "walking_validation_macro_accuracy": 0.2534067193987221,
    "walking_class_recall_json": "{\"concrete\": {\"recall\": 0.0, \"support\": 5068}, \"ice\": {\"recall\": 0.09866102889358704, \"support\": 5676}, \"marble\": {\"recall\": 0.006354249404289118, \"support\": 5036}, \"sand\": {\"recall\": 0.9086115992970123, \"support\": 15932}}"
  },
  "slip": {
    "detector": "slip",
    "selection_data": "existing validation + d2209cd v02 only",
    "holdout_runs_accessed_before_selection": 0,
    "architecture_unchanged": true,
    "window_ms": 5,
    "mixture_ratio": 0.5,
    "training_seed": 20261002,
    "model_path": "models/slip_walking_candidate.keras",
    "model_sha256": "e89ab3a99bd147b6e33847dfaffdd01cae539e6a975f5f6bcd46b4aa226fd130",
    "normalization_path": "normalization/slip.json",
    "normalization_sha256": "0eb3a256febd0aa84c07dfbfdb30fa28176a20374d294a77b6dfa5debd2c08c3",
    "parameters": 1237,
    "candidate_gate_pass": false,
    "selection_rationale": "all mandatory gates, then walking normal FPR, walking positive recall, static/controlled retention, p95 latency, unchanged model size, deterministic ties",
    "production_artifact_replaced": false,
    "selected_probability_threshold": 0.9999994039535522,
    "selected_runtime_persistence": 5,
    "controlled_validation_causal_run_fpr": 0.0,
    "controlled_validation_run_recall": 0.0,
    "walking_normal_false_positive_runs": 0,
    "walking_positive_run_recall": 1.0,
    "walking_anticipation_runs": 0,
    "walking_p95_stable_latency_ms": 23.299999999999997,
    "diagnostic_fallback_selected": true,
    "validation_gate_failure_reasons": [
      "no runtime configuration for this trained candidate passed every mandatory validation gate"
    ]
  },
  "sink": {
    "detector": "sink",
    "selection_data": "existing validation + d2209cd v02 only",
    "holdout_runs_accessed_before_selection": 0,
    "architecture_unchanged": true,
    "window_ms": 20,
    "mixture_ratio": 1.0,
    "training_seed": 20261002,
    "model_path": "models/sink_walking_candidate.keras",
    "model_sha256": "c99c8addd39cfafbea5f3b042b8ce57d558cf5a310ccd0e285e01560c4dc36d0",
    "normalization_path": "normalization/sink.json",
    "normalization_sha256": "a44859b3eff543d64cf61b5c612500a42171a4ae7bafab68f7823ebd715660cb",
    "parameters": 1221,
    "candidate_gate_pass": false,
    "selection_rationale": "all mandatory gates, then walking normal FPR, walking positive recall, static/controlled retention, p95 latency, unchanged model size, deterministic ties",
    "production_artifact_replaced": false,
    "selected_probability_threshold": 1.0,
    "selected_runtime_persistence": 1,
    "controlled_validation_causal_run_fpr": 0.0,
    "controlled_validation_run_recall": 0.0,
    "walking_normal_false_positive_runs": 0,
    "walking_positive_run_recall": 0.0,
    "walking_anticipation_runs": 0,
    "walking_p95_stable_latency_ms": 0.0,
    "diagnostic_fallback_selected": true,
    "validation_gate_failure_reasons": [
      "no runtime configuration for this trained candidate passed every mandatory validation gate"
    ]
  }
}
```

Validation and holdout results:

```json
{
  "terrain_candidate": {
    "accuracy": 0.27970094757889247,
    "macro_accuracy": 0.2720897223723793,
    "class_metrics": {
      "concrete": {
        "support": 14732,
        "recall": 0.0
      },
      "marble": {
        "support": 14665,
        "recall": 0.016979202182066142
      },
      "ice": {
        "support": 13152,
        "recall": 0.10819647201946472
      },
      "sand": {
        "support": 14966,
        "recall": 0.9631832152879861
      }
    }
  },
  "terrain_frozen_baseline": {
    "accuracy": 0.2792836651308354,
    "macro_accuracy": 0.27169180335444,
    "class_metrics": {
      "concrete": {
        "support": 14732,
        "recall": 0.0
      },
      "marble": {
        "support": 14665,
        "recall": 0.017115581316058642
      },
      "ice": {
        "support": 13152,
        "recall": 0.10827250608272507
      },
      "sand": {
        "support": 14966,
        "recall": 0.9613791260189763
      }
    }
  },
  "slip_candidate": {
    "runs": 36,
    "normal_runs": 27,
    "normal_false_positive_runs": 6,
    "physical_positive_runs": 9,
    "detected_positive_runs": 9,
    "positive_run_recall": 1.0,
    "anticipation_runs": 3,
    "label_mask_violation_samples": 0,
    "latency_median_ms": 16.0,
    "latency_p95_ms": 390.2,
    "latency_max_ms": 393,
    "profile_results": {
      "ice": {
        "physical_positive_runs": 9,
        "detected_runs": 9,
        "detected_speeds_mps": [
          0.1,
          0.15,
          0.2
        ]
      }
    },
    "model_inference_compute_ms_per_window": 0.015667489059030416,
    "probability_threshold_endpoint_metrics": {
      "support": 53101,
      "positive_support": 2411,
      "precision": 0.32539118065433853,
      "recall": 0.3795105765242638,
      "false_positive_rate": 0.03742355494180312
    },
    "stable_persistence_endpoint_metrics": {
      "support": 53101,
      "positive_support": 2411,
      "precision": 0.3181818181818182,
      "recall": 0.2061385317295728,
      "false_positive_rate": 0.021010061156046558
    }
  },
  "sink_normal_candidate": {
    "runs": 27,
    "normal_runs": 27,
    "normal_false_positive_runs": 0,
    "physical_positive_runs": 0,
    "detected_positive_runs": 0,
    "positive_run_recall": 0.0,
    "anticipation_runs": 0,
    "label_mask_violation_samples": 0,
    "latency_median_ms": null,
    "latency_p95_ms": null,
    "latency_max_ms": null,
    "profile_results": {},
    "model_inference_compute_ms_per_window": 0.016180181283026254,
    "probability_threshold_endpoint_metrics": {
      "support": 42989,
      "positive_support": 0,
      "precision": 0.0,
      "recall": 0.0,
      "false_positive_rate": 0.0
    },
    "stable_persistence_endpoint_metrics": {
      "support": 42989,
      "positive_support": 0,
      "precision": 0.0,
      "recall": 0.0,
      "false_positive_rate": 0.0
    }
  },
  "sink_positive_candidate": {
    "runs": 18,
    "normal_runs": 0,
    "normal_false_positive_runs": 0,
    "physical_positive_runs": 18,
    "detected_positive_runs": 2,
    "positive_run_recall": 0.1111111111111111,
    "anticipation_runs": 0,
    "label_mask_violation_samples": 0,
    "latency_median_ms": 1482.5,
    "latency_p95_ms": 1568.45,
    "latency_max_ms": 1578,
    "profile_results": {
      "sand_solref_interpolation_1of3": {
        "physical_positive_runs": 9,
        "detected_runs": 0,
        "detected_speeds_mps": []
      },
      "sand_solref_interpolation_2of3": {
        "physical_positive_runs": 9,
        "detected_runs": 2,
        "detected_speeds_mps": [
          0.1,
          0.15
        ]
      }
    },
    "model_inference_compute_ms_per_window": 0.01567056977511418,
    "probability_threshold_endpoint_metrics": {
      "support": 28947,
      "positive_support": 16073,
      "precision": 0.8333333333333334,
      "recall": 0.0006221613886642195,
      "false_positive_rate": 0.00015535187199005747
    },
    "stable_persistence_endpoint_metrics": {
      "support": 28947,
      "positive_support": 16073,
      "precision": 0.8333333333333334,
      "recall": 0.0006221613886642195,
      "false_positive_rate": 0.00015535187199005747
    }
  },
  "sink_complete_profiles": [],
  "terrain_air_transitions": 0
}
```

Spatial A/B/C/D replay is read-only and every source run is fall-confounded:

```json
{
  "all_runs_fall_confounded": true,
  "read_only_after_selection": true,
  "case_results": {
    "A": {
      "runs": 3,
      "frozen_correct": 0,
      "candidate_correct": 0,
      "delta": 0
    },
    "B": {
      "runs": 3,
      "frozen_correct": 3,
      "candidate_correct": 3,
      "delta": 0
    },
    "C": {
      "runs": 3,
      "frozen_correct": 3,
      "candidate_correct": 0,
      "delta": -3
    },
    "D": {
      "runs": 3,
      "frozen_correct": 0,
      "candidate_correct": 0,
      "delta": 0
    }
  },
  "a_or_d_improved": false
}
```

No production model, normalization, threshold, persistence, INT8, Vela/E84,
U55, board, final-test, or System-v1 artifact was changed.

## Readiness gates

- WALKING_RETRAIN_DATASET_READY=true
- WALKING_RETRAIN_SPLIT_INTEGRITY_READY=true
- WALKING_TERRAIN_FLOAT_CANDIDATE_READY=true
- WALKING_SLIP_FLOAT_CANDIDATE_READY=false
- WALKING_SINK_FLOAT_CANDIDATE_READY=false
- WALKING_MODEL_HOLDOUT_NON_ACCESS_READY=true
- WALKING_TERRAIN_HOLDOUT_READY=true
- WALKING_SLIP_HOLDOUT_READY=false
- WALKING_SINK_HOLDOUT_READY=false
- WALKING_BOUNDED_FLOAT_RETRAINING_READY=false
- WALKING_INT8_PREPARATION_AUTHORIZED=false