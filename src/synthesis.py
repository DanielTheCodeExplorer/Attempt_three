import json
import pandas as pd
from pathlib import Path

def main():
    create_empty_dataset()

def create_empty_dataset():
    schema = get_schema()
    columns = get_columns(schema)
    data = get_data(schema)
    create_df = pd.DataFrame(columns=columns, data=data)
    create_df.to_excel("data/outputs/dataset.xlsx")
    return create_df

def get_schema():
    schema_path = Path("data/inputs/schema_profile.json")
    schema = json.loads(schema_path.read_text())
    return schema

def get_columns(schema):
    return schema["sheets"][0]["columns_in_order"]

def get_data(schema):
    sheet = schema["sheets"][0]
    columns = sheet["columns_in_order"]
    row_count = sheet["data_row_count"]
    return pd.DataFrame(index=range(row_count), columns=columns)


if __name__ == "__main__":
    main()
