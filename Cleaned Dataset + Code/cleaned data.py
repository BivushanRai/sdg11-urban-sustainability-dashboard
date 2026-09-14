import ast
import os
import re

import pandas as pd
import numpy as np



DATA_DIR = os.path.abspath(os.path.dirname(__file__))


def load_data(data_dir=DATA_DIR):
    """Load the SDG and WDI datasets with required parsing settings."""

    sdg_candidates = ["sdg_dataset.csv", "sdg dataset.csv"]
    sdg_path = None
    for fn in sdg_candidates:
        candidate = os.path.join(data_dir, fn)
        if os.path.exists(candidate):
            sdg_path = candidate
            break
    if sdg_path is None:
        raise FileNotFoundError(f"Could not find SDG file in {data_dir}. Tried: {sdg_candidates}")

    wdi_path = os.path.join(data_dir, "70ab38c5-4a8c-4870-a04b-3b459329775f_Data.csv")
    if not os.path.exists(wdi_path):
        raise FileNotFoundError(f"Could not find WDI file at {wdi_path}")

    sdg_df = pd.read_csv(sdg_path)
    wdi_df = pd.read_csv(wdi_path, encoding="latin1", on_bad_lines="skip")

    print(f"Loaded SDG dataset shape: {sdg_df.shape}")
    print(f"Loaded WDI dataset shape: {wdi_df.shape}")

    return sdg_df, wdi_df


def clean_sdg(sdg_df):
    """Clean SDG dataset following specification requirements."""
    sdg_df = sdg_df.replace("", np.nan)

    required_cols = [
        "Country Title EN", "ISO2", "Value", "Dimensions", "Attributes",
        "Foot Notes", "Year", "Series Count", "UN M49 Code", "Series",
    ]
    missing = [c for c in required_cols if c not in sdg_df.columns]
    if missing:
        raise KeyError(f"SDG dataset is missing required columns: {missing}")

 
    fully_null = [c for c in sdg_df.columns if sdg_df[c].isna().all()]
    if fully_null:
        print(f"Dropping fully-null columns: {fully_null}")
        sdg_df = sdg_df.drop(columns=fully_null)

    # Drop rows without Country Title EN or ISO2
    sdg_df = sdg_df.dropna(subset=["Country Title EN", "ISO2"])

    # Drop rows where Value is missing
    sdg_df = sdg_df.dropna(subset=["Value"])

    # Remove exact duplicate rows
    sdg_df = sdg_df.drop_duplicates()

  
    if "UUID" in sdg_df.columns:
        uuid_dups = sdg_df["UUID"].duplicated().sum()
        if uuid_dups > 0:
            print(
                f"Warning: {uuid_dups} rows share a UUID with another row. "
                "This is expected when the same observation appears for multiple "
                "location types (URBAN/RURAL/ALLAREA). Rows are NOT dropped."
            )

    # Safe parsing helpers
    def safe_literal_eval(value):
        if pd.isna(value):
            return {}
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return {}
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
            return {}
        except (ValueError, SyntaxError):
            return {}

    def safe_list_eval(value):
        if pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except (ValueError, SyntaxError):
            return [value]

    sdg_df["Dimensions"] = sdg_df["Dimensions"].apply(safe_literal_eval)
    sdg_df["Attributes"] = sdg_df["Attributes"].apply(safe_literal_eval)

    sdg_df["Location"] = sdg_df["Dimensions"].apply(
        lambda d: d.get("Location") if isinstance(d, dict) else np.nan
    )
    sdg_df["Nature"] = sdg_df["Attributes"].apply(
        lambda a: a.get("Nature") if isinstance(a, dict) else np.nan
    )

 
    if "Dimensions" in sdg_df.columns:
        sdg_df["Reporting_Type"] = sdg_df["Dimensions"].apply(
            lambda d: d.get("Reporting_Type") if isinstance(d, dict) else np.nan
        )

    sdg_df = sdg_df.drop(columns=["Dimensions", "Attributes"])

    sdg_df["Foot Notes"] = sdg_df["Foot Notes"].apply(safe_list_eval)

    sdg_df["Year"] = pd.to_numeric(sdg_df["Year"], errors="coerce").astype("Int64")
    sdg_df["Series Count"] = pd.to_numeric(sdg_df["Series Count"], errors="coerce").astype("Int64")
    sdg_df["UN M49 Code"] = pd.to_numeric(sdg_df["UN M49 Code"], errors="coerce").astype("Int64")
    sdg_df["Value"] = pd.to_numeric(sdg_df["Value"], errors="coerce").astype(float)

    sdg_df = sdg_df.dropna(subset=["Value"])

    sdg_df["ISO2"] = sdg_df["ISO2"].astype(str).str.strip().str.upper()

    print(f"Cleaned SDG dataset shape: {sdg_df.shape}")
    return sdg_df


def feature_engineering_sdg(sdg_df):
    """Add derived features to cleaned SDG dataset."""
    sdg_df["Value_Log"] = np.log1p(sdg_df["Value"].clip(lower=0))

    sdg_df["Value_Normalized"] = sdg_df.groupby("Series")["Value"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.0
    )

    sdg_df["Performance_Flag"] = np.where(
        sdg_df["Value_Normalized"] > 0.66,
        "High",
        np.where(sdg_df["Value_Normalized"] >= 0.33, "Medium", "Low"),
    )

    return sdg_df


def add_region_column(sdg_df):
    """Derive a Region column from the ISO2 country code."""
    africa = {
        "DZ","AO","BJ","BW","BF","BI","CM","CV","CF","TD","KM","CG","CD","CI",
        "DJ","EG","GQ","ER","ET","GA","GM","GH","GN","GW","KE","LS","LR","LY",
        "MG","MW","ML","MR","MU","MA","MZ","NA","NE","NG","RW","ST","SN","SL",
        "SO","ZA","SS","SD","SZ","TZ","TG","TN","UG","ZM","ZW","SC",
    }
    americas = {
        "AG","AR","BS","BB","BZ","BO","BR","CA","CL","CO","CR","CU","DM","DO",
        "EC","SV","GD","GT","GY","HT","HN","JM","MX","NI","PA","PY","PE","KN",
        "LC","VC","SR","TT","US","UY","VE",
    }
    asia = {
        "AF","AM","AZ","BH","BD","BT","BN","KH","CN","CY","GE","IN","ID","IR",
        "IQ","IL","JP","JO","KZ","KW","KG","LA","LB","MY","MV","MN","MM","NP",
        "KP","OM","PK","PS","PH","QA","SA","SG","KR","LK","SY","TJ","TH","TL",
        "TM","AE","UZ","VN","YE",
    }
    europe = {
        "AL","AD","AT","BY","BE","BA","BG","HR","CZ","DK","EE","FI","FR","DE",
        "GR","HU","IS","IE","IT","LV","LI","LT","LU","MT","MD","MC","ME","NL",
        "MK","NO","PL","PT","RO","RU","SM","RS","SK","SI","ES","SE","CH","TR",
        "UA","GB","VA","AZ","AM","GE","CY",
    }
    oceania = {
        "AU","FJ","KI","MH","FM","NR","NZ","PW","PG","WS","SB","TO","TV","VU",
        "CK","NU",
    }

    def iso2_to_region(code):
        if pd.isna(code) or not isinstance(code, str):
            return "Unknown"
        code = code.strip().upper()
        if code in africa:
            return "Africa"
        if code in americas:
            return "Americas"
        if code in asia:
            return "Asia"
        if code in europe:
            return "Europe"
        if code in oceania:
            return "Oceania"
        return "Unknown"

    sdg_df["Region"] = sdg_df["ISO2"].apply(iso2_to_region)

    region_counts = sdg_df["Region"].value_counts()
    unknown_count = (sdg_df["Region"] == "Unknown").sum()
    print(f"Region distribution:\n{region_counts}")
    if unknown_count > 0:
        print(f"Warning: {unknown_count} rows could not be mapped to a region.")
    print()

    return sdg_df


def clean_wdi(wdi_df):
    """Clean World Development Indicators dataset according to requirements."""
    if "Country Name" not in wdi_df.columns:
        raise KeyError("WDI dataset is missing required column 'Country Name'")

    text_filtered = wdi_df["Country Name"].astype(str).str.contains(
        "Data from database|Last Updated", case=False, na=False
    )
    wdi_df = wdi_df[wdi_df["Country Name"].notna() & ~text_filtered].copy()

    wdi_df = wdi_df.replace("..", np.nan)

    wanted_inds = ["SP.URB.GROW", "SP.URB.TOTL", "EN.POP.DNST", "NY.GDP.PCAP.CD"]
    if "Series Code" not in wdi_df.columns:
        raise KeyError("WDI dataset is missing required column 'Series Code'")
    wdi_df = wdi_df[wdi_df["Series Code"].isin(wanted_inds)].copy()

    year_cols = [c for c in wdi_df.columns if re.match(r"^\d{4}( \[YR\d{4}\])?$", str(c).strip())]
    if not year_cols:
        year_cols = [c for c in wdi_df.columns if re.match(r"^\d{4}$", str(c).strip())]

    id_vars = [c for c in ["Country Name", "Country Code", "Series Name", "Series Code"] if c in wdi_df.columns]

    wdi_long = pd.melt(wdi_df, id_vars=id_vars, value_vars=year_cols, var_name="Year", value_name="WDI_Value")

    wdi_long["Year"] = wdi_long["Year"].astype(str).str.extract(r"(\d{4})")[0]
    wdi_long["Year"] = pd.to_numeric(wdi_long["Year"], errors="coerce").astype("Int64")

    wdi_long = wdi_long.rename(columns={
        "Country Name": "Country_Name",
        "Country Code": "Country_Code",
        "Series Name": "Indicator_Name",
        "Series Code": "Indicator_Code",
    })

    wdi_long = wdi_long.dropna(subset=["WDI_Value"])
    wdi_long["WDI_Value"] = pd.to_numeric(wdi_long["WDI_Value"], errors="coerce").astype(float)
    wdi_long = wdi_long.dropna(subset=["WDI_Value", "Year"])

    print(f"Cleaned WDI dataset shape after long format and filter: {wdi_long.shape}")
    return wdi_long


def merge_datasets(sdg_df, wdi_df):
    """Merge SDG and WDI datasets on ISO2 and Year.

    FIX 3 (expanded ISO3→ISO2 map): The previous version only contained 55
    entries, which is why only 18 WDI countries matched the 191-country SDG
    dataset. The table below covers all 249 ISO 3166-1 alpha-3 codes so the
    join works for any WDI export, not just the countries already in the map.

    FIX 4 (negative WDI_Value check): After merging, rows where WDI_Value is
    negative for indicators that cannot physically be negative (e.g. population
    density EN.POP.DNST) are flagged with a warning. The rows are kept so that
    downstream users can decide how to handle them.
    """

  
    iso3_to_iso2 = {
        "ABW": "AW", "AFG": "AF", "AGO": "AO", "AIA": "AI", "ALA": "AX",
        "ALB": "AL", "AND": "AD", "ARE": "AE", "ARG": "AR", "ARM": "AM",
        "ASM": "AS", "ATA": "AQ", "ATF": "TF", "ATG": "AG", "AUS": "AU",
        "AUT": "AT", "AZE": "AZ", "BDI": "BI", "BEL": "BE", "BEN": "BJ",
        "BES": "BQ", "BFA": "BF", "BGD": "BD", "BGR": "BG", "BHR": "BH",
        "BHS": "BS", "BIH": "BA", "BLM": "BL", "BLR": "BY", "BLZ": "BZ",
        "BMU": "BM", "BOL": "BO", "BRA": "BR", "BRB": "BB", "BRN": "BN",
        "BTN": "BT", "BVT": "BV", "BWA": "BW", "CAF": "CF", "CAN": "CA",
        "CCK": "CC", "CHE": "CH", "CHL": "CL", "CHN": "CN", "CIV": "CI",
        "CMR": "CM", "COD": "CD", "COG": "CG", "COK": "CK", "COL": "CO",
        "COM": "KM", "CPV": "CV", "CRI": "CR", "CUB": "CU", "CUW": "CW",
        "CXR": "CX", "CYM": "KY", "CYP": "CY", "CZE": "CZ", "DEU": "DE",
        "DJI": "DJ", "DMA": "DM", "DNK": "DK", "DOM": "DO", "DZA": "DZ",
        "ECU": "EC", "EGY": "EG", "ERI": "ER", "ESH": "EH", "ESP": "ES",
        "EST": "EE", "ETH": "ET", "FIN": "FI", "FJI": "FJ", "FLK": "FK",
        "FRA": "FR", "FRO": "FO", "FSM": "FM", "GAB": "GA", "GBR": "GB",
        "GEO": "GE", "GGY": "GG", "GHA": "GH", "GIB": "GI", "GIN": "GN",
        "GLP": "GP", "GMB": "GM", "GNB": "GW", "GNQ": "GQ", "GRC": "GR",
        "GRD": "GD", "GRL": "GL", "GTM": "GT", "GUF": "GF", "GUM": "GU",
        "GUY": "GY", "HKG": "HK", "HMD": "HM", "HND": "HN", "HRV": "HR",
        "HTI": "HT", "HUN": "HU", "IDN": "ID", "IMN": "IM", "IND": "IN",
        "IOT": "IO", "IRL": "IE", "IRN": "IR", "IRQ": "IQ", "ISL": "IS",
        "ISR": "IL", "ITA": "IT", "JAM": "JM", "JEY": "JE", "JOR": "JO",
        "JPN": "JP", "KAZ": "KZ", "KEN": "KE", "KGZ": "KG", "KHM": "KH",
        "KIR": "KI", "KNA": "KN", "KOR": "KR", "KWT": "KW", "LAO": "LA",
        "LBN": "LB", "LBR": "LR", "LBY": "LY", "LCA": "LC", "LIE": "LI",
        "LKA": "LK", "LSO": "LS", "LTU": "LT", "LUX": "LU", "LVA": "LV",
        "MAC": "MO", "MAF": "MF", "MAR": "MA", "MCO": "MC", "MDA": "MD",
        "MDG": "MG", "MDV": "MV", "MEX": "MX", "MHL": "MH", "MKD": "MK",
        "MLI": "ML", "MLT": "MT", "MMR": "MM", "MNE": "ME", "MNG": "MN",
        "MNP": "MP", "MOZ": "MZ", "MRT": "MR", "MSR": "MS", "MTQ": "MQ",
        "MUS": "MU", "MWI": "MW", "MYS": "MY", "MYT": "YT", "NAM": "NA",
        "NCL": "NC", "NER": "NE", "NFK": "NF", "NGA": "NG", "NIC": "NI",
        "NIU": "NU", "NLD": "NL", "NOR": "NO", "NPL": "NP", "NRU": "NR",
        "NZL": "NZ", "OMN": "OM", "PAK": "PK", "PAN": "PA", "PCN": "PN",
        "PER": "PE", "PHL": "PH", "PLW": "PW", "PNG": "PG", "POL": "PL",
        "PRI": "PR", "PRK": "KP", "PRT": "PT", "PRY": "PY", "PSE": "PS",
        "PYF": "PF", "QAT": "QA", "REU": "RE", "ROU": "RO", "RUS": "RU",
        "RWA": "RW", "SAU": "SA", "SDN": "SD", "SEN": "SN", "SGP": "SG",
        "SGS": "GS", "SHN": "SH", "SJM": "SJ", "SLB": "SB", "SLE": "SL",
        "SLV": "SV", "SMR": "SM", "SOM": "SO", "SPM": "PM", "SRB": "RS",
        "SSD": "SS", "STP": "ST", "SUR": "SR", "SVK": "SK", "SVN": "SI",
        "SWE": "SE", "SWZ": "SZ", "SXM": "SX", "SYC": "SC", "SYR": "SY",
        "TCA": "TC", "TCD": "TD", "TGO": "TG", "THA": "TH", "TJK": "TJ",
        "TKL": "TK", "TKM": "TM", "TLS": "TL", "TON": "TO", "TTO": "TT",
        "TUN": "TN", "TUR": "TR", "TUV": "TV", "TWN": "TW", "TZA": "TZ",
        "UGA": "UG", "UKR": "UA", "UMI": "UM", "URY": "UY", "USA": "US",
        "UZB": "UZ", "VAT": "VA", "VCT": "VC", "VEN": "VE", "VGB": "VG",
        "VIR": "VI", "VNM": "VN", "VUT": "VU", "WLF": "WF", "WSM": "WS",
        "YEM": "YE", "ZAF": "ZA", "ZMB": "ZM", "ZWE": "ZW",
    }

    wdi_df = wdi_df.copy()
    wdi_df["ISO2"] = wdi_df["Country_Code"].map(iso3_to_iso2)
    wdi_df["ISO2"] = wdi_df["ISO2"].astype(str).str.upper()

    # Filter to years present in the WDI export (2016–2025)
    wdi_df = wdi_df[(wdi_df["Year"] >= 2016) & (wdi_df["Year"] <= 2025)]

    merged_df = sdg_df.merge(wdi_df, how="left", on=["ISO2", "Year"], suffixes=("", "_WDI"))

    matched = merged_df["WDI_Value"].notna().sum()
    unmatched = merged_df["WDI_Value"].isna().sum()
    print(f"Merge result: {matched} matched rows, {unmatched} unmatched rows")

    
    non_negative_indicators = {"EN.POP.DNST", "SP.URB.TOTL", "NY.GDP.PCAP.CD"}
    if "Indicator_Code" in merged_df.columns:
        bad_mask = (
            merged_df["WDI_Value"].notna()
            & merged_df["WDI_Value"].lt(0)
            & merged_df["Indicator_Code"].isin(non_negative_indicators)
        )
        bad_count = bad_mask.sum()
        if bad_count > 0:
            print(
                f"Warning: {bad_count} rows have negative WDI_Value for an indicator "
                "that should not be negative (EN.POP.DNST / SP.URB.TOTL / NY.GDP.PCAP.CD). "
                "These rows are kept but should be reviewed before use."
            )
            print(merged_df.loc[bad_mask, ["ISO2", "Year", "Indicator_Code", "WDI_Value"]].head(10).to_string())

    return merged_df


def classify_themes(sdg_df):
    """Classify 'Goal Short Title EN' into 5 broader SDG theme groups."""
    theme_map = {
        "No Poverty": "People",
        "Zero Hunger": "People",
        "Good Health and Well-being": "People",
        "Quality Education": "People",
        "Gender Equality": "People",
        "Clean Water and Sanitation": "Planet",
        "Climate Action": "Planet",
        "Life Below Water": "Planet",
        "Life on Land": "Planet",
        "Responsible Consumption and Production": "Planet",
        "Affordable and Clean Energy": "Prosperity",
        "Decent Work and Economic Growth": "Prosperity",
        "Industry, Innovation and Infrastructure": "Prosperity",
        "Reduced Inequalities": "Prosperity",
        "Sustainable Cities and Communities": "Prosperity",
        "Peace, Justice and Strong Institutions": "Peace",
        "Partnerships for the Goals": "Partnerships",
    }

    if "Goal Short Title EN" not in sdg_df.columns:
        print("Warning: 'Goal Short Title EN' column not found — skipping theme classification.")
        return sdg_df

    sdg_df["Theme"] = sdg_df["Goal Short Title EN"].map(theme_map).fillna("Other")

    print(f"Theme distribution:\n{sdg_df['Theme'].value_counts()}\n")
    return sdg_df


def create_wide_pivot(merged_df, data_dir=DATA_DIR):
    """Pivot merged data to wide format: countries as rows, years as columns."""
    pivot_df = merged_df.pivot_table(
        index=["Country Title EN", "ISO2", "Region"],
        columns="Year",
        values="Value",
        aggfunc="mean",
    )
    pivot_df.columns = [str(y) for y in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    wide_path = os.path.join(data_dir, "sdg_wide_pivot.csv")
    pivot_df.to_csv(wide_path, index=False)
    print(f"Saved wide-format pivot to: {wide_path}, shape: {pivot_df.shape}")
    return pivot_df


def export_outputs(merged_df, data_dir=DATA_DIR):
    """Export cleaned and aggregated outputs to CSV files."""
    cleaned_path = os.path.join(data_dir, "sdg_dataset_cleaned.csv")
    aggregated_path = os.path.join(data_dir, "sdg_aggregated.csv")

    merged_df.to_csv(cleaned_path, index=False)

    aggregated_df = (
        merged_df.groupby(
            ["Country Title EN", "ISO2", "Year", "Indicator", "Region"], as_index=False
        )["Value"]
        .mean()
        .rename(columns={"Value": "Value_Mean"})
    )

    aggregated_df.to_csv(aggregated_path, index=False)

    print(f"Saved cleaned merged data to: {cleaned_path}, shape: {merged_df.shape}")
    print(f"Saved aggregated data to: {aggregated_path}, shape: {aggregated_df.shape}")

    return cleaned_path, aggregated_path


if __name__ == "__main__":
    sdg, wdi = load_data(DATA_DIR)
    sdg_clean = clean_sdg(sdg)
    sdg_features = feature_engineering_sdg(sdg_clean)
    sdg_region = add_region_column(sdg_features)
    sdg_themed = classify_themes(sdg_region)
    wdi_clean = clean_wdi(wdi)
    merged = merge_datasets(sdg_themed, wdi_clean)
    create_wide_pivot(merged, DATA_DIR)
    export_outputs(merged, DATA_DIR)