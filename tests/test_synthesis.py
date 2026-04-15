from src.synthesis import create_empty_dataset, create_synthetic_dataset, get_schema


def test_create_empty_dataset_has_expected_columns():
    df = create_empty_dataset()

    # These columns cover the main child-row families that the generator must emit.
    expected_columns = [
        "Summary/bio",
        "Skill Level 1",
        "Capability Name",
        "Proficiency",
        "Interest",
        "Certification Issued Date",
        "Certification Expiration Date",
        "Job History Start Date",
        "Job History End Date",
        "Position",
        "Company",
        "Location",
        "Description",
        "Education title/type of degree",
        "Area of Study",
        "Academic Institution",
    ]

    for column in expected_columns:
        assert column in df.columns


def test_create_synthetic_dataset_matches_schema_shape():
    schema = get_schema()
    df = create_synthetic_dataset(schema=schema, seed=42)
    sheet = schema["sheets"][0]

    # The synthetic export should match the profiled workbook dimensions exactly.
    assert df.shape == (sheet["data_row_count"], sheet["column_count"])
    assert df["TalentLink ID"].nunique() == 106


def test_duplicate_columns_and_constant_person_fields_are_consistent():
    df = create_synthetic_dataset(seed=42)

    # Report-layout duplicate columns should carry the same business value.
    assert (df["Global Network"] == df["Global Network__2"]).all()
    assert (df["Global Net Competency"] == df["Global Net Competency__2"]).all()
    assert (df["Go to Market"] == df["Go to Market__2"]).all()
    assert (df["Unnamed_BC"] == "r" + df["Local employee ID"]).all()

    # Person-level identifiers and profile metadata should stay fixed within each ID block.
    per_person_uniques = df.groupby("TalentLink ID")[["Local employee ID", "PwC email", "Profile Last Edited (YYYY/MM/DD)"]].nunique()
    assert (per_person_uniques == 1).all().all()


def test_row_family_conditional_fields_do_not_leak():
    df = create_synthetic_dataset(seed=42)

    # Each row family owns a small set of fields; everything else should stay blank.
    language_rows = df["Skill Level 1"] == "Language"
    job_rows = df["Skill Level 1"] == "Job History"
    education_rows = df["Skill Level 1"] == "Education"
    travel_rows = df["Skill Level 1"] == "Travel"
    capability_rows = df["Skill Level 1"].isin(["Skills", "Certifications/Credentials", "Language"])

    assert df.loc[~language_rows, "Proficiency"].isna().all()
    assert df.loc[~job_rows, ["Job History Start Date", "Position", "Company"]].isna().all().all()
    assert df.loc[~education_rows, ["Academic Institution", "In Process"]].isna().all().all()
    assert df.loc[~travel_rows, ["Project Type Interest", "Travel Interest", "Additional Information"]].isna().all().all()
    assert df.loc[~capability_rows, "Capability Name"].isna().all()


def test_section_order_and_travel_cap_hold_per_person():
    df = create_synthetic_dataset(seed=42)

    # Rows should be emitted in the same section order used by the flattened export.
    order = {
        "Skills": 0,
        "Certifications/Credentials": 1,
        "Language": 2,
        "Job History": 3,
        "Education": 4,
        "Travel": 5,
    }

    for _, group in df.groupby("TalentLink ID", sort=False):
        observed = [order[value] for value in group["Skill Level 1"]]
        assert observed == sorted(observed)
        # Travel is modeled as an optional single child row per person.
        assert (group["Skill Level 1"] == "Travel").sum() <= 1
