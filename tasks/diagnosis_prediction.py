from typing import List, Dict, Tuple



def _to_ccs_codes(icd9_codes: List[str], ccs_mapping) -> List[str]:
    mapped: List[str] = []
    for code in icd9_codes:
        try:
            ccs_codes = ccs_mapping.map(code)
        except Exception:
            ccs_codes = None

        if ccs_codes is None:
            continue

        if isinstance(ccs_codes, str):
            mapped.append(ccs_codes)
        else:
            mapped.extend([c for c in ccs_codes if c is not None])

    # deduplicate while preserving order
    deduped = list(dict.fromkeys(mapped))
    return deduped



def sequential_diagnosis_prediction_mimic3(patient, ccs_mapping) -> Tuple[List[Dict], List[Dict]]:
    """Builds sequential diagnosis prediction samples for MIMIC-III.

    Returns two sample lists:
    - ICD-9 label samples
    - CCS label samples
    """

    visits: List[Dict] = []
    for i in range(len(patient)):
        visit = patient[i]
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        # ATC-4 level (already mapped in dataset construction, keep defensive slicing)
        drugs = [drug[:4] if isinstance(drug, str) else drug for drug in drugs]

        # keep visits with complete modalities
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        visits.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
            }
        )

    # need at least history + next visit target
    if len(visits) < 2:
        return [], []

    icd_samples: List[Dict] = []
    ccs_samples: List[Dict] = []

    for t in range(1, len(visits)):
        hist = visits[:t]
        target_visit = visits[t]

        hist_visit_ids = [v["visit_id"] for v in hist]
        hist_conditions = [v["conditions"] for v in hist]
        hist_procedures = [v["procedures"] for v in hist]
        hist_drugs = [v["drugs"] for v in hist]

        target_icd9 = target_visit["conditions"]
        target_ccs = _to_ccs_codes(target_icd9, ccs_mapping)

        icd_samples.append(
            {
                "visit_id": target_visit["visit_id"],
                "patient_id": target_visit["patient_id"],
                "conditions": hist_conditions,
                "procedures": hist_procedures,
                "drugs": hist_drugs,
                "visit_index_list": [[v] for v in hist_visit_ids],
                "label": target_icd9,
            }
        )

        ccs_samples.append(
            {
                "visit_id": target_visit["visit_id"],
                "patient_id": target_visit["patient_id"],
                "conditions": hist_conditions,
                "procedures": hist_procedures,
                "drugs": hist_drugs,
                "visit_index_list": [[v] for v in hist_visit_ids],
                "label": target_ccs,
            }
        )

    return icd_samples, ccs_samples
