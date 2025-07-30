import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# App Configuration
st.set_page_config(
    page_title="Aave Wallet Credit Scorer",
    page_icon="🪙",
    layout="wide"
)

# Core Scoring and Data Processing Functions
@st.cache_data
def load_and_process_data(uploaded_file_obj):
    """
    Loads data from an uploaded file object, processes it, engineers features,
    and calculates scores. This function is cached for performance.
    """
    try:
        with zipfile.ZipFile(uploaded_file_obj, 'r') as z:
            json_filename = z.namelist()[0]
            with z.open(json_filename) as f:
                data = json.load(f)
        raw_df = pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

    # Preprocessing
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], unit='s')
    action_df = pd.json_normalize(raw_df['actionData'])
    df = pd.concat([raw_df.drop('actionData', axis=1), action_df], axis=1)
    numeric_cols = ['amount', 'assetPriceUSD']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['amount_decimal'] = df['amount'] / 10**18
    df['amountUSD'] = df['amount_decimal'] * df['assetPriceUSD']
    df['amountUSD'] = df['amountUSD'].fillna(0)

    # Feature Engineering
    features_list = []
    epsilon = 1e-9
    for wallet in df['userWallet'].unique():
        wallet_df = df[df['userWallet'] == wallet].sort_values('timestamp')
        if wallet_df.empty: continue
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
    features_df = pd.DataFrame(features_list)

    # Scoring
    weights = {
        'wallet_age_days': 0.15, 'transaction_count': 0.10, 'liquidation_count': -0.40,
        'repay_to_borrow_ratio': 0.25, 'net_liquidity_provided_usd': 0.15,
        'borrows_usd': 0.05, 'unique_assets_used': 0.05
    }
    features_df['raw_score'] = 0
    for feature, weight in weights.items():
        if feature == 'repay_to_borrow_ratio':
            features_df[feature] = np.minimum(features_df[feature], 5)
        scaler = MinMaxScaler()
        norm_feature = scaler.fit_transform(features_df[[feature]])
        features_df['raw_score'] += norm_feature.flatten() * weight
    final_scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df['credit_score'] = final_scaler.fit_transform(features_df[['raw_score']])
    features_df['credit_score'] = features_df['credit_score'].astype(int)

    return features_df.set_index('wallet')

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

# Streamlit User Interface
st.title("🪙 Aave V2 Wallet Credit Scorer")
st.markdown("An interactive app to assess wallet reliability based on historical transaction data.")

uploaded_file = st.file_uploader("Upload 'user-wallet-transactions.json.zip'", type=['zip'])

if uploaded_file is not None:
    with st.spinner('Processing data... This may take a minute.'):
        final_df = load_and_process_data(uploaded_file)

    st.success(f"✅ Data processed successfully! Found {len(final_df)} unique wallets.")

    st.header("🔍 Wallet Score Lookup")
    wallet_address = st.text_input("Enter a wallet address to check its score:", "").strip().lower()

    if wallet_address:
        if wallet_address in final_df.index:
            wallet_data = final_df.loc[wallet_address]
            score = wallet_data['credit_score']
            color = "normal" if score > 700 else "off" if score > 400 else "inverse"
            st.metric(label=f"Credit Score for `{wallet_address[:6]}...{wallet_address[-4:]}`", value=score, delta_color=color)
            with st.expander("Show detailed features for this wallet"):
                st.dataframe(wallet_data)
        else:
            st.error("Wallet address not found in the dataset.")

    st.header("📊 Overall Analytics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        fig, ax = plt.subplots()
        sns.histplot(final_df['credit_score'], bins=20, kde=True, ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Top 5 Highest Scoring Wallets")
        st.dataframe(final_df[['credit_score']].sort_values(by='credit_score', ascending=False).head(5))
        st.subheader("Top 5 Lowest Scoring Wallets")
        st.dataframe(final_df[['credit_score']].sort_values(by='credit_score', ascending=True).head(5))

    st.markdown("---")
    st.header("📥 Download Results")
    scores_csv = convert_df_to_csv(final_df[['credit_score']].sort_values(by='credit_score', ascending=False))
    features_csv = convert_df_to_csv(final_df)
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(label="Download Scores CSV", data=scores_csv, file_name='credit_scores.csv', mime='text/csv')
    with dl_col2:
        st.download_button(label="Download Full Features CSV", data=features_csv, file_name='credit_features.csv', mime='text/csv')
else:
    st.info("Awaiting for the transaction data file to be uploaded.")
