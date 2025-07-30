import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import sys

def load_data(file_path):
    """Loads transaction data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        sys.exit(1)

def preprocess_data(df):
    """Preprocesses the raw DataFrame."""
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    action_df = pd.json_normalize(df['actionData'])
    df = pd.concat([df.drop('actionData', axis=1), action_df], axis=1)
    numeric_cols = ['amount', 'assetPriceUSD']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['amount_decimal'] = df['amount'] / 10**18
    df['amountUSD'] = df['amount_decimal'] * df['assetPriceUSD']
    df['amountUSD'] = df['amountUSD'].fillna(0)
    return df

def get_feature_engineering(df):
    """Engineers features for each wallet."""
    wallets = df['userWallet'].unique()
    features_list = []
    epsilon = 1e-9
    for wallet in wallets:
        wallet_df = df[df['userWallet'] == wallet].sort_values('timestamp')
        if wallet_df.empty:
            continue
        wallet_age_days = (wallet_df['timestamp'].max() - wallet_df['timestamp'].min()).days
        tx_counts = wallet_df['action'].value_counts()
        deposits_usd = wallet_df[wallet_df['action'] == 'deposit']['amountUSD'].sum()
        borrows_usd = wallet_df[wallet_df['action'] == 'borrow']['amountUSD'].sum()
        repays_usd = wallet_df[wallet_df['action'] == 'repay']['amountUSD'].sum()
        redeems_usd = wallet_df[wallet_df['action'] == 'redeemunderlying']['amountUSD'].sum()
        features_list.append({
            'wallet': wallet, 'wallet_age_days': wallet_age_days,
            'transaction_count': len(wallet_df),
            'liquidation_count': tx_counts.get('liquidationcall', 0),
            'repay_to_borrow_ratio': repays_usd / (borrows_usd + epsilon),
            'net_liquidity_provided_usd': deposits_usd - redeems_usd,
            'borrows_usd': borrows_usd,
            'unique_assets_used': wallet_df['assetSymbol'].nunique()
        })
    return pd.DataFrame(features_list)

def calculate_scores(features_df):
    """Calculates credit scores based on engineered features."""
    df = features_df.copy()
    weights = {
        'wallet_age_days': 0.15, 'transaction_count': 0.10, 'liquidation_count': -0.40,
        'repay_to_borrow_ratio': 0.25, 'net_liquidity_provided_usd': 0.15,
        'borrows_usd': 0.05, 'unique_assets_used': 0.05
    }
    df['raw_score'] = 0
    for feature, weight in weights.items():
        if feature in df.columns:
            if feature == 'repay_to_borrow_ratio':
                df[feature] = np.minimum(df[feature], 5)
            scaler = MinMaxScaler()
            norm_feature = scaler.fit_transform(df[[feature]])
            df['raw_score'] += norm_feature.flatten() * weight
    final_scaler = MinMaxScaler(feature_range=(0, 1000))
    df['credit_score'] = final_scaler.fit_transform(df[['raw_score']])
    df['credit_score'] = df['credit_score'].astype(int)
    return df[['wallet', 'credit_score']].sort_values(by='credit_score', ascending=False)

def main():
    """Main function to run the scoring script."""
    parser = argparse.ArgumentParser(description="Generate credit scores for wallets from Aave v2 transaction data.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input JSON file with transaction data.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output CSV file with scores.")
    
    args = parser.parse_args()

    print("1. Loading data...")
    raw_df = load_data(args.input_file)
    
    print("2. Preprocessing data...")
    processed_df = preprocess_data(raw_df)

    print("3. Engineering features...")
    features_df = get_feature_engineering(processed_df)

    print("4. Calculating credit scores...")
    scores_df = calculate_scores(features_df)
    
    print(f"5. Saving scores to {args.output_file}...")
    scores_df.to_csv(args.output_file, index=False)

    print("✅ Done!")

if __name__ == "__main__":
    main()
