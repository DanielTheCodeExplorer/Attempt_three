from src.synthesis import create_empty_dataset


def test_create_empty_dataset():

    df = create_empty_dataset()

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

    