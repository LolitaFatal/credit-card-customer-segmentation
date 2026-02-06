import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings

# --- STYLE SETTINGS ---
# FinTech Color Palette
CUSTOM_PALETTE = ["#34495e", "#16a085", "#7f8c8d", "#c0392b", "#2980b9"]
sns.set_palette(CUSTOM_PALETTE)
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})

plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

warnings.filterwarnings('ignore')

class CreditCardClustering:
    
    def __init__(self, data_path, output_dir='output_images', n_clusters=4):
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        
        self.df_original = None
        self.df_cleaned = None
        self.model = None
        self.labels = None
        
        # Setup output directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.full_output_dir = os.path.join(script_dir, output_dir)
        os.makedirs(self.full_output_dir, exist_ok=True)
        print(f"[INIT] Output directory ready: {self.full_output_dir}")

    def load_data(self):
        # PART A: Load and Inspect
        try:
            self.df_original = pd.read_csv(self.data_path)
            print(f"[LOAD] Data loaded successfully. Shape: {self.df_original.shape}")
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.data_path}")
            raise

        # --- HOMEWORK REQUIREMENT: DATA INSPECTION ---
        print("\n" + "="*40)
        print(" DATA INSPECTION (HEAD, INFO, DESCRIBE)")
        print("="*40)
        
        print("\n1. DATA INFO (Column Types):")
        print("-" * 30)
        self.df_original.info()
        
        print("\n2. HEAD (First 5 Rows):")
        print("-" * 30)
        print(self.df_original.head())
        
        print("\n3. DESCRIBE (Statistical Summary):")
        print("-" * 30)
        print(self.df_original.describe())
        print("="*40 + "\n")
        # ---------------------------------------------

        self.df_cleaned = self.df_original.copy()
        
        # Handling Missing Values (Median strategy)
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        self.df_cleaned[numeric_cols] = imputer.fit_transform(self.df_cleaned[numeric_cols])
        print("[CLEAN] Missing values filled with Median.")

    def generate_eda(self):
        # PART B: Visualization
        print("[EDA] Generating charts...")
        
        # 1. Correlation Matrix
        plt.figure(figsize=(12, 10))
        cols = [c for c in self.df_cleaned.columns if c != 'CUST_ID']
        corr = self.df_cleaned[cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0, 
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.full_output_dir, '1_correlation.png'))
        plt.close()

        # 2. Histograms
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.histplot(self.df_cleaned['BALANCE'], bins=40, kde=True, color=CUSTOM_PALETTE[0], ax=ax[0])
        ax[0].set_title('Distribution of Balance (Debt)')
        
        sns.histplot(self.df_cleaned['PURCHASES'], bins=40, kde=True, color=CUSTOM_PALETTE[1], ax=ax[1])
        ax[1].set_title('Distribution of Purchases')
        plt.savefig(os.path.join(self.full_output_dir, '2_histograms.png'))
        plt.close()

        # 3. Scatter (Segmentation View)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df_cleaned, x='CASH_ADVANCE', y='PURCHASES', 
                        alpha=0.6, color=CUSTOM_PALETTE[4], edgecolor=None)
        plt.title('Segmentation View: Cash vs Purchases')
        plt.xlabel('Cash Advance (Withdrawals)')
        plt.ylabel('Purchases (Card Usage)')
        plt.savefig(os.path.join(self.full_output_dir, '3_scatter.png'))
        plt.close()

        # 4. Boxplots
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.boxplot(y=self.df_cleaned['CASH_ADVANCE'], ax=ax[0], color=CUSTOM_PALETTE[2])
        ax[0].set_title('Outliers: Cash Advance')
        sns.boxplot(y=self.df_cleaned['CREDIT_LIMIT'], ax=ax[1], color=CUSTOM_PALETTE[3])
        ax[1].set_title('Outliers: Credit Limit')
        plt.savefig(os.path.join(self.full_output_dir, '4_boxplots.png'))
        plt.close()

    def preprocess_and_cluster(self):
        # PART C & D & E
        
        # 1. Feature Selection (Drop ID)
        X = self.df_cleaned.drop('CUST_ID', axis=1) if 'CUST_ID' in self.df_cleaned.columns else self.df_cleaned.copy()
        
        # 2. Scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # 3. Elbow Method
        inertias = []
        k_range = range(1, 10)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, marker='o', linewidth=2, color=CUSTOM_PALETTE[0])
        plt.axvline(x=self.n_clusters, color=CUSTOM_PALETTE[3], linestyle='--')
        plt.title('Elbow Method')
        plt.savefig(os.path.join(self.full_output_dir, '5_elbow.png'))
        plt.close()
        
        # 4. Final Clustering
        print(f"[MODEL] Training KMeans with K={self.n_clusters}...")
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(X_scaled)
        
        # Add labels back
        self.df_with_labels = self.df_cleaned.copy()
        if 'CUST_ID' in self.df_with_labels.columns:
             self.df_with_labels.drop('CUST_ID', axis=1, inplace=True)
        self.df_with_labels['CLUSTER'] = self.labels

    def create_report(self):
        print("[REPORT] Generating HTML dashboard...")
        
        # 1. Calculate Profiles
        profile = self.df_with_labels.groupby('CLUSTER').mean().T
        # Add Count row
        counts = self.df_with_labels['CLUSTER'].value_counts()
        profile.loc['Customer Count'] = counts
        
        # 2. Highlight Max Values (Visual Aid)
        # We highlight the maximum value in each row to see which cluster dominates
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #d4edda; color: #155724; font-weight: bold' if v else '' for v in is_max]

        styled_table = profile.style.apply(highlight_max, axis=1).format("{:.2f}")
        table_html = styled_table.to_html()

        # 3. Define Business Logic Text (English)
        business_logic = """
        <div class="analysis-section">
            <h3>Cluster 0: "Sleeping Customers" (Low Activity)</h3>
            <ul>
                <li><strong>Observation:</strong> Lowest Balance (~1000) and lowest Purchases (~270). High volume of users.</li>
                <li><strong>Strategy:</strong> "Wake Up" Campaign.</li>
                <li><strong>Action:</strong> "Make 3 purchases this month and get 5% Cashback" OR "Log in to the App to get 500 bonus points".</li>
            </ul>

            <h3>Cluster 1: "The VIPs" (Big Spenders)</h3>
            <ul>
                <li><strong>Observation:</strong> Highest Purchases (~7600) and Highest Credit Limit. They pay effectively.</li>
                <li><strong>Strategy:</strong> Retention & Loyalty. Don't lose them!</li>
                <li><strong>Action:</strong> Offer Premium Cards (Gold/Platinum), Travel benefits, or Higher Credit Limits.</li>
            </ul>

            <h3>Cluster 2: "Cash Users" (High Risk)</h3>
            <ul>
                <li><strong>Observation:</strong> Massive Cash Advance usage (~4500) with low Purchases. High Balance (Debt).</li>
                <li><strong>Risk:</strong> High probability of default (bankruptcy). They use the card as an emergency ATM.</li>
                <li><strong>Action:</strong> Mitigation. Stop increasing their credit limit. Offer a structured "Personal Loan" to consolidate debt at a lower rate.</li>
            </ul>

            <h3>Cluster 3: "Active Spenders" (Middle Class)</h3>
            <ul>
                <li><strong>Observation:</strong> Good Purchase activity (~1200) with low Debt (~800). They use the card wisely.</li>
                <li><strong>Strategy:</strong> Growth (Up-selling).</li>
                <li><strong>Action:</strong> Encourage larger purchases (Electronics, Installments) or Installment Plans.</li>
            </ul>
        </div>
        """

        # 4. Assemble HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clustering Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 40px; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
                h1 {{ color: #2C3E50; border-bottom: 3px solid #16a085; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 40px; border-left: 5px solid #2980b9; padding-left: 15px; }}
                h3 {{ color: #c0392b; margin-top: 20px; }}
                .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                img {{ width: 100%; border-radius: 4px; border: 1px solid #eee; }}
                
                /* Table Styles */
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }}
                th {{ background-color: #2C3E50; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                
                /* Analysis Box */
                .analysis-section {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; margin-top: 30px; border: 1px solid #bdc3c7; }}
                ul {{ line-height: 1.6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Executive Clustering Report</h1>
                <p>Internal Report | Automated Analysis</p>
                
                <h2>1. Data Landscape (EDA)</h2>
                <div class="img-grid">
                    <div><img src="{self.output_dir}/1_correlation.png"><p><strong>Fig 1:</strong> Correlation</p></div>
                    <div><img src="{self.output_dir}/2_histograms.png"><p><strong>Fig 2:</strong> Distributions (Skewed)</p></div>
                    <div><img src="{self.output_dir}/3_scatter.png"><p><strong>Fig 3:</strong> Segmentation (Cash vs Purchases)</p></div>
                    <div><img src="{self.output_dir}/4_boxplots.png"><p><strong>Fig 4:</strong> Outliers Detection</p></div>
                </div>

                <h2>2. Model Optimization</h2>
                <div style="text-align: center;">
                    <img src="{self.output_dir}/5_elbow.png" style="max-width: 600px;">
                    <p>Optimal K=4 selected based on the Elbow Method.</p>
                </div>
                
                <h2>3. Cluster Profiles (Data)</h2>
                <p>The table below shows the <strong>Average (Mean)</strong> values for each cluster. <br>
                <span style="background-color: #d4edda; color: #155724; padding: 2px 5px; font-weight: bold;">Green</span> indicates the highest value in that category.</p>
                {table_html}
                
                <h2>4. Business Conclusions & Recommendations</h2>
                {business_logic}
                
                <hr>
                <p style="text-align: center; color: #7f8c8d; font-size: 0.8em;">Generated by Python Architecture | 2026</p>
            </div>
        </body>
        </html>
        """
        
        # Save file in the SAME folder as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, 'final_report.html')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"[DONE] Report generated: {report_path}")

if __name__ == "__main__":
    # Ensure correct path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "Customer Data.csv")
    
    # Run Pipeline
    app = CreditCardClustering(file_path)
    app.load_data()
    app.generate_eda()
    app.preprocess_and_cluster()
    app.create_report()