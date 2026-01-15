import pandas as pd

# Load original dataset
df = pd.read_csv("fake reviews dataset.csv")

print("Original columns:", df.columns)

# Rename review text column
df["review_text"] = df["text_"]

# Convert labels:
# CG = Genuine (0)
# Anything else = Fake (1)
df["label"] = df["label"].apply(lambda x: 0 if x == "CG" else 1)

# Ensure rating is numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Keep only required columns
df = df[["review_text", "label", "rating"]]

# Remove empty rows
df.dropna(inplace=True)

# Save cleaned dataset
df.to_csv("reviews.csv", index=False)

print("✅ Clean reviews.csv created successfully")
print(df.head())
print("Label counts:")
print(df['label'].value_counts())
