#!/usr/bin/env python3
"""
Model Evaluation Script for GST Invoice Extraction
Loads Ground Truth JSON and computes accuracy, precision, recall, and F1 score against model predictions.
"""

import json
import sys
import difflib

def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip().lower().replace(" ", "").replace("-", "").replace("/", "")

def string_similarity(a, b):
    a = str(a).strip().lower()
    b = str(b).strip().lower()
    return difflib.SequenceMatcher(None, a, b).ratio()

def evaluate_invoice(ground_truth_inv, predicted_inv):
    scores = {}
    
    # 1. Header & Entity Field Exact Matches
    key_fields = [
        ("invoice_number", ground_truth_inv["invoice_metadata"]["invoice_number"], predicted_inv.get("invoice_number")),
        ("invoice_date", ground_truth_inv["invoice_metadata"]["invoice_date"], predicted_inv.get("invoice_date")),
        ("place_of_supply", ground_truth_inv["invoice_metadata"]["place_of_supply"], predicted_inv.get("place_of_supply")),
        ("seller_name", ground_truth_inv["seller_details"]["legal_name"], predicted_inv.get("seller_name")),
        ("seller_gstin", ground_truth_inv["seller_details"]["gstin"], predicted_inv.get("seller_gstin")),
        ("buyer_name", ground_truth_inv["buyer_details"]["legal_name"], predicted_inv.get("buyer_name")),
        ("buyer_gstin", ground_truth_inv["buyer_details"]["gstin"], predicted_inv.get("buyer_gstin")),
        ("total_taxable_value", ground_truth_inv["summary"]["total_taxable_value"], predicted_inv.get("total_taxable_value")),
        ("grand_total", ground_truth_inv["summary"]["grand_total"], predicted_inv.get("grand_total"))
    ]
    
    field_results = {}
    exact_matches = 0
    for field_name, gt_val, pred_val in key_fields:
        is_exact = False
        sim = 0.0
        if isinstance(gt_val, (int, float)):
            try:
                pred_float = float(str(pred_val).replace(",", "").replace("₹", "").replace("INR", "").strip())
                is_exact = abs(gt_val - pred_float) < 1.0  # Within ₹1 tolerance for roundoff
                sim = 1.0 if is_exact else 0.0
            except:
                is_exact = False
        else:
            norm_gt = normalize_text(gt_val)
            norm_pred = normalize_text(pred_val)
            is_exact = (norm_gt == norm_pred)
            sim = string_similarity(str(gt_val), str(pred_val))
            
        if is_exact:
            exact_matches += 1
            
        field_results[field_name] = {
            "ground_truth": gt_val,
            "predicted": pred_val,
            "exact_match": is_exact,
            "similarity": round(sim, 3)
        }
        
    scores["field_level"] = field_results
    scores["field_accuracy"] = round(exact_matches / len(key_fields), 4)
    
    # 2. Line Items Table Evaluation
    gt_items = ground_truth_inv.get("items", [])
    pred_items = predicted_inv.get("items", [])
    
    tp = 0
    for gt_it in gt_items:
        # Match by HSN or Description similarity
        matched = False
        for pr_it in pred_items:
            hsn_match = normalize_text(gt_it.get("hsn_code")) == normalize_text(pr_it.get("hsn_code"))
            desc_sim = string_similarity(gt_it.get("description", ""), pr_it.get("description", ""))
            if hsn_match or desc_sim > 0.75:
                matched = True
                break
        if matched:
            tp += 1
            
    fp = max(0, len(pred_items) - tp)
    fn = max(0, len(gt_items) - tp)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    scores["table_level"] = {
        "gt_items_count": len(gt_items),
        "pred_items_count": len(pred_items),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }
    
    return scores

def main():
    gt_file = "ground_truth_dataset.json"
    try:
        with open(gt_file, "r") as f:
            gt_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {gt_file} not found. Run inside the gst-sample-invoices directory.")
        return

    print("=" * 65)
    print(" GST Invoice Model Evaluation Framework")
    print(f" Loaded {len(gt_data['invoices'])} Ground Truth Invoices from {gt_file}")
    print("=" * 65)
    
    # Demonstration using perfect prediction mock
    sample_inv = gt_data["invoices"][0]
    print(f"\nEvaluating Mock Prediction on: {sample_inv['invoice_metadata']['invoice_number']}")
    
    mock_prediction = {
        "invoice_number": sample_inv["invoice_metadata"]["invoice_number"],
        "invoice_date": sample_inv["invoice_metadata"]["invoice_date"],
        "place_of_supply": sample_inv["invoice_metadata"]["place_of_supply"],
        "seller_name": sample_inv["seller_details"]["legal_name"],
        "seller_gstin": sample_inv["seller_details"]["gstin"],
        "buyer_name": sample_inv["buyer_details"]["legal_name"],
        "buyer_gstin": sample_inv["buyer_details"]["gstin"],
        "total_taxable_value": sample_inv["summary"]["total_taxable_value"],
        "grand_total": sample_inv["summary"]["grand_total"],
        "items": sample_inv["items"]
    }
    
    results = evaluate_invoice(sample_inv, mock_prediction)
    print(f"Header Field Accuracy: {results['field_accuracy'] * 100:.1f}%")
    print(f"Table Line Item F1 Score: {results['table_level']['f1_score'] * 100:.1f}%\n")
    print("Field-by-Field Breakdown:")
    for fld, res in results["field_level"].items():
        status = "✓ MATCH" if res["exact_match"] else "✗ MISMATCH"
        print(f"  [{status}] {fld}: GT='{res['ground_truth']}' | PRED='{res['predicted']}'")

if __name__ == "__main__":
    main()
