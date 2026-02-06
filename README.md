# Credit Card Customer Clustering

## Overview
This project performs customer segmentation for a credit card company using the K-Means Clustering algorithm. The goal is to identify distinct customer groups to improve marketing strategies and risk management.

## Project Structure
- clustering_final.py: Main Python script for data cleaning, EDA, and modeling.
- final_report.html: Self-contained HTML dashboard with all insights and graphs.
- output_images/: Directory containing generated visualizations.
- Customer Data.csv: Dataset used for analysis.

## Key Findings (Clusters)
Based on the analysis (K=4), we identified four customer segments:
1. Sleeping Customers: Low activity and balance.
2. VIPs (Big Spenders): High purchases and credit limit.
3. Cash Users (High Risk): High cash advance usage and debt.
4. Active Middle Class: Balanced usage and payments.

## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Run the script:
   python clustering_final.py
3. Open final_report.html to view the results.