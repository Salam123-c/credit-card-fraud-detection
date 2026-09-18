import pandas as pd

d = pd.read_excel(r"C:\Users\hp\Desktop\PRO_1_10\CREDIT\top_500_rows.xlsx")
print("shape:", d.shape)
print(d.dtypes.to_string())
print()
print("is_fraud:", d["is_fraud"].value_counts().to_dict())
print("amt stats: min", d["amt"].min(), "max", d["amt"].max(), "median", d["amt"].median())
print("category:", d["category"].value_counts().to_dict())
print("gender:", d["gender"].value_counts().to_dict())
print("unique: cc_num", d["cc_num"].nunique(), "| merchant", d["merchant"].nunique(),
      "| job", d["job"].nunique(), "| city", d["city"].nunique())
print("date range:", d["trans_date_trans_time"].min(), "->", d["trans_date_trans_time"].max())
print()
print(d[["trans_date_trans_time","merchant","category","amt","city","state","city_pop","job","is_fraud","merch_zipcode"]].head(3).to_string())
