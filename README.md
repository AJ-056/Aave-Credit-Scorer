#Aave V2 Wallet Credit Scorer:
This project provides a robust model to assign a credit score between 0 and 1000 to blockchain wallets based on their historical transaction behavior on the Aave V2 protocol. The model is delivered as both a command-line script and an interactive web application.

Overview:
The challenge is to score wallet addresses solely from their raw transaction-level data, including actions like deposits, borrows, repays, and liquidations. This solution uses an unsupervised heuristic scoring model. This approach was chosen because it is transparent, easily explainable, and does not require pre-labeled "good" or "bad" wallets for training.

The model translates a wallet's transaction history into a set of quantifiable features that represent its financial reliability and risk profile. These features are then combined using a weighted formula to generate a final, scaled credit score.

Architecture and Processing Flow:
The scoring process follows a clear, multi-step pipeline:

Data Ingestion & Preprocessing: The raw JSON data is loaded. Timestamps are converted to datetime objects, and transaction amounts are standardized into a USD equivalent by multiplying the token amount by its USD price at the time of the transaction.

Feature Engineering: For each unique wallet, a vector of behavioral features is calculated. These features are the core of the model's intelligence. Key features include:

liquidation_count: The number of times a wallet has been liquidated. This is the most significant negative indicator.

repay_to_borrow_ratio: Total USD value of repayments divided by the total USD value of borrows. A healthy ratio is >= 1.

net_liquidity_provided_usd: The net USD value a wallet has contributed to the protocol (Deposits - Redemptions).

wallet_age_days: The number of days between the wallet's first and last transaction.

transaction_count: Total number of interactions with the protocol.

Heuristic Scoring: A raw score is calculated using a weighted sum of the normalized features. Feature weights are set to prioritize responsible behavior (high repayment ratio, no liquidations) over simple activity metrics. For example, liquidation_count has a large negative weight, while repay_to_borrow_ratio has a large positive weight.

Score Scaling: The raw scores for all wallets are scaled to fit the required 0-1000 range using a Min-Max scaler. This ensures a consistent and understandable final output.

How to Use:
You can use this project in two ways:

A. Command-Line Script:
This method is for generating a CSV file of all wallet scores.

Setup:

Place score_wallets.py, requirements.txt, and your data file (user-wallet-transactions.json) in a folder.

Install dependencies: pip install -r requirements.txt

Run:

python score_wallets.py --input_file user-wallet-transactions.json --output_file wallet_scores.csv

B. Interactive Streamlit Web App
This method provides a user-friendly web interface to explore the scores.

Setup:

Place app.py, requirements.txt in a folder.

Install dependencies: pip install -r requirements.txt

Run:

streamlit run app.py

Your browser will open with the application. Upload your user-wallet-transactions.json.zip file to begin.
