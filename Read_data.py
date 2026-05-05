import pandas as pd

start = "1970-01-01"
end = "2026-05-04"

series = {
    "UNRATE": "unemployment_rate",
    "CPIAUCSL": "cpi",
    "FEDFUNDS": "fed_funds_rate",
    "GS10": "ten_year_treasury",
    "TB3MS": "three_month_tbill",
    "INDPRO": "industrial_production",
    "USREC": "recession"
}
# pull the data from the FRED
df = pd.DataFrame()

for code, name in series.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    temp = pd.read_csv(url)
    temp["observation_date"] = pd.to_datetime(temp["observation_date"])
    temp = temp.set_index("observation_date")
    temp = temp.rename(columns={code: name})

    if df.empty:
        df = temp
    else:
        df = df.join(temp, how="outer")

df = df[df.index >= start]

# Convert values to numbers
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


#convert the cpi to the inflation rate
df["Inflation_rate"] = df["cpi"].pct_change(12) * 100

# remove cpi
df = df.drop(columns = ["cpi"])
df = df.dropna()

#classify recession as a binary 1 or 0
df["recession"] = df["recession"].astype(int)

df.to_csv("FRED_data.csv")

print(df.head())
print(df.tail())
print(df.info())
print(df["recession"].value_counts())
