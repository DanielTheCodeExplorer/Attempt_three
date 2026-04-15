import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from faker import Faker


SECTIONS = [
    "Skills",
    "Certifications/Credentials",
    "Language",
    "Job History",
    "Education",
    "Travel",
]

SECTION_TARGET_SHARE = {
    "Skills": 0.752,
    "Job History": 0.142,
    "Certifications/Credentials": 0.053,
    "Language": 0.029,
    "Education": 0.017,
    "Travel": 0.008,
}

PERSON_OPTIONAL_FIELDS = [
    "Resource Management contact",
    "Strategic Region",
    "Strategic Market",
    "Summary/bio",
]

ALWAYS_BLANK_COLUMNS = [
    "Local Industry",
    "Local Sector",
    "Supervisory Org Code",
]

DUPLICATE_COLUMN_PAIRS = [
    ("Global Network", "Global Network__2"),
    ("Global Net Competency", "Global Net Competency__2"),
    ("Go to Market", "Go to Market__2"),
]

CAPABILITY_SECTIONS = {"Skills", "Certifications/Credentials", "Language"}

fake = Faker()


@dataclass(frozen=True)
class Person:
    employee_id: str
    talentlink_id: int
    name: str
    email: str
    management_level: str
    job_level: str
    relationship_leader: str
    resource_type: str
    grade_code: str
    global_los: str
    global_network: str
    global_net_competency: str
    go_to_market: str
    office: str
    territory: str
    career_coach: str
    rm_contact: str | None
    hr_contact: str
    availability: str
    local_los: str
    cost_centre: str
    local_sublos2: str
    local_sublos3: str
    local_sublos4: str
    strategic_region: str | None
    strategic_market: str | None
    summary: str | None
    profile_last_edited: str
    unnamed_bc: str
    unnamed_bd: int
    unnamed_be: str


def main() -> None:
    schema = get_schema()
    output_path = Path("data/outputs/synthetic talent link data.xlsx")
    df = create_synthetic_dataset(schema=schema, seed=42)
    write_workbook(df, schema, output_path)


def create_empty_dataset(schema: dict | None = None) -> pd.DataFrame:
    schema = schema or get_schema()
    sheet = schema["sheets"][0]
    return pd.DataFrame(index=range(sheet["data_row_count"]), columns=sheet["columns_in_order"])


def create_synthetic_dataset(schema: dict | None = None, seed: int = 42) -> pd.DataFrame:
    schema = schema or get_schema()
    sheet = schema["sheets"][0]
    rng = random.Random(seed)
    Faker.seed(seed)

    columns = get_columns(schema)
    total_rows = sheet["data_row_count"]
    person_count = 106
    target_counts = compute_target_counts(total_rows, SECTION_TARGET_SHARE)
    people = build_people(person_count, rng)
    section_counts = allocate_person_section_counts(people, target_counts, rng)

    rows: list[dict] = []
    for person in people:
        rows.extend(build_person_rows(person, section_counts[person.talentlink_id], columns, rng))

    df = pd.DataFrame(rows, columns=columns)
    return df


def write_workbook(df: pd.DataFrame, schema: dict | None = None, output_path: str | Path = "data/outputs/synthetic talent link data.xlsx") -> Path:
    schema = schema or get_schema()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workbook_name = schema.get("workbook_name", "TalentLink Synthetic.xlsx")
    sheet = schema["sheets"][0]
    sheet_name = sheet["sheet_name"]

    metadata_rows = [
        [f"Workbook: {workbook_name}"],
        [f"Sheet: {sheet_name}"],
        [f"Generated rows: {len(df)}"],
        ["Synthetic export generated from reverse-engineered schema rules."],
        [None],
        ["Employee Details"],
    ]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(metadata_rows).to_excel(
            writer,
            sheet_name=sheet_name,
            header=False,
            index=False,
        )
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=6,
            index=False,
        )

    return output_path


def get_schema() -> dict:
    schema_path = Path("data/inputs/schema_profile.json")
    return json.loads(schema_path.read_text())


def get_columns(schema: dict) -> list[str]:
    return schema["sheets"][0]["columns_in_order"]


def compute_target_counts(total_rows: int, shares: dict[str, float]) -> dict[str, int]:
    total_share = sum(shares.values())
    raw = {name: total_rows * (share / total_share) for name, share in shares.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total_rows - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def build_people(person_count: int, rng: random.Random) -> list[Person]:
    offices = ["London", "Birmingham", "Manchester", "Leeds", "Bristol", "Glasgow"]
    territories = ["United Kingdom"]
    global_los_values = ["Consulting", "Advisory", "Risk"]
    networks = ["Risk", "Deals", "Technology", "Operations"]
    competencies = ["Cyber", "Data", "Controls", "Transformation"]
    markets = ["FS", "Energy", "Public Sector", "TMT"]
    local_sublos4_by_cost = {
        "CC-1001": "Digital",
        "CC-1002": "Cyber",
        "CC-1003": "Analytics",
        "CC-1004": "Controls",
    }
    management_pairs = {
        "Director": ("Senior Manager", "RL-Director"),
        "Senior Manager": ("Manager", "RL-Senior"),
        "Manager": ("Associate", "RL-Manager"),
        "Associate": ("Analyst", "RL-Associate"),
    }

    people: list[Person] = []
    used_employee_ids: set[str] = set()
    used_talentlink_ids: set[int] = set()

    for _ in range(person_count):
        employee_id = unique_numeric_string(9, used_employee_ids, rng)
        talentlink_id = unique_int(5, 8, used_talentlink_ids, rng)
        name = fake.name()
        cost_centre = rng.choice(list(local_sublos4_by_cost))
        management_level = rng.choice(list(management_pairs))
        job_level, relationship_leader = management_pairs[management_level]
        strategic_region = rng.choice(["Risk North", "Risk South", "Risk Central", None, None, None])
        strategic_market = rng.choice(["Financial Services", "Industry", None]) if strategic_region else None
        summary = generate_summary(rng) if rng.random() < 0.7 else None
        rm_contact = fake.name() if rng.random() < 0.65 else None
        office = rng.choice(offices)
        domain = "example.pwc.com"
        email = derive_email(name, domain, rng)

        people.append(
            Person(
                employee_id=employee_id,
                talentlink_id=talentlink_id,
                name=name,
                email=email,
                management_level=management_level,
                job_level=job_level,
                relationship_leader=relationship_leader,
                resource_type=rng.choice(["Employee", "Associate"]),
                grade_code=rng.choice(["A1", "A2", "M1", "SM1", "D1"]),
                global_los=rng.choice(global_los_values),
                global_network=rng.choice(networks),
                global_net_competency=rng.choice(competencies),
                go_to_market=rng.choice(markets),
                office=office,
                territory=rng.choice(territories),
                career_coach=fake.name(),
                rm_contact=rm_contact,
                hr_contact=fake.name(),
                availability=rng.choice(["0%", "25%", "50%", "75%", "100%"]),
                local_los="Consulting",
                cost_centre=cost_centre,
                local_sublos2="Advisory",
                local_sublos3="Risk Consulting",
                local_sublos4=local_sublos4_by_cost[cost_centre],
                strategic_region=strategic_region,
                strategic_market=strategic_market,
                summary=summary,
                profile_last_edited=random_profile_edit_date(rng),
                unnamed_bc=f"r{employee_id}",
                unnamed_bd=45712,
                unnamed_be="Active TalentLink filter",
            )
        )

    return people


def allocate_person_section_counts(
    people: list[Person],
    target_counts: dict[str, int],
    rng: random.Random,
) -> dict[int, Counter]:
    assignments = {person.talentlink_id: Counter() for person in people}

    for person in people:
        assignments[person.talentlink_id]["Skills"] = 1

    remaining = target_counts.copy()
    remaining["Skills"] -= len(people)

    allocate_weighted_counts(people, assignments, remaining["Skills"], "Skills", rng, min_extra=12, max_extra=65)
    allocate_weighted_counts(people, assignments, remaining["Job History"], "Job History", rng, min_extra=2, max_extra=12)
    allocate_weighted_counts(people, assignments, remaining["Certifications/Credentials"], "Certifications/Credentials", rng, min_extra=0, max_extra=5)
    allocate_weighted_counts(people, assignments, remaining["Language"], "Language", rng, min_extra=0, max_extra=2)
    allocate_weighted_counts(people, assignments, remaining["Education"], "Education", rng, min_extra=0, max_extra=2)

    travel_people = rng.sample(people, k=min(target_counts["Travel"], len(people)))
    for person in travel_people:
        assignments[person.talentlink_id]["Travel"] = 1

    return assignments


def allocate_weighted_counts(
    people: list[Person],
    assignments: dict[int, Counter],
    total: int,
    section: str,
    rng: random.Random,
    min_extra: int,
    max_extra: int,
) -> None:
    eligible = people[:]
    rng.shuffle(eligible)
    remaining = total

    while remaining > 0:
        progress = False
        for person in eligible:
            current = assignments[person.talentlink_id][section]
            if current >= max_extra:
                continue
            lower_bound = 0 if current > 0 else min_extra
            add = rng.randint(lower_bound, max_extra - current)
            if add <= 0:
                continue
            add = min(add, remaining)
            assignments[person.talentlink_id][section] += add
            remaining -= add
            progress = True
            if remaining == 0:
                break
        if not progress:
            break


def build_person_rows(person: Person, section_counts: Counter, columns: list[str], rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    section_builders = {
        "Skills": build_skill_row,
        "Certifications/Credentials": build_certification_row,
        "Language": build_language_row,
        "Job History": build_job_history_row,
        "Education": build_education_row,
        "Travel": build_travel_row,
    }

    for section in SECTIONS:
        for index in range(section_counts[section]):
            row = build_base_row(person, columns)
            row["Skill Level 1"] = section
            section_builders[section](row, person, index, rng)
            rows.append(row)

    return rows


def build_base_row(person: Person, columns: list[str]) -> dict:
    row = {column: None for column in columns}
    row.update(
        {
            "Local employee ID": person.employee_id,
            "TalentLink ID": person.talentlink_id,
            "Resource Name": person.name,
            "Resource Status": "Enabled",
            "Resource Type": person.resource_type,
            "Management level": person.management_level,
            "Job level": person.job_level,
            "Grade Code": person.grade_code,
            "Global LoS": person.global_los,
            "Global Network": person.global_network,
            "Global Net Competency": person.global_net_competency,
            "Go to Market": person.go_to_market,
            "PwC office": person.office,
            "PwC Territory": person.territory,
            "Relationship Leader": person.relationship_leader,
            "Career Coach": person.career_coach,
            "Resource Management contact": person.rm_contact,
            "HR contact": person.hr_contact,
            "PwC email": person.email,
            "30 day availability forecast": person.availability,
            "Local LoS": person.local_los,
            "Cost Centre": person.cost_centre,
            "Global Network__2": person.global_network,
            "Global Net Competency__2": person.global_net_competency,
            "Go to Market__2": person.go_to_market,
            "Local SubLoS2": person.local_sublos2,
            "Local SubLoS3": person.local_sublos3,
            "Local SubLoS4": person.local_sublos4,
            "Strategic Region": person.strategic_region,
            "Strategic Market": person.strategic_market,
            "Summary/bio": person.summary,
            "Profile Last Edited (YYYY/MM/DD)": person.profile_last_edited,
            "Unnamed_BC": person.unnamed_bc,
            "Unnamed_BD": person.unnamed_bd,
            "Unnamed_BE": person.unnamed_be,
        }
    )
    for column in ALWAYS_BLANK_COLUMNS:
        row[column] = None
    return row


def build_skill_row(row: dict, person: Person, _: int, rng: random.Random) -> None:
    row["Capability Name"] = rng.choice(
        [
            "Python",
            "SQL",
            "Data Analysis",
            "Power BI",
            "Risk Controls",
            "Stakeholder Management",
            "ETL Design",
            "Programme Delivery",
            "Regulatory Reporting",
            "Data Governance",
        ]
    )
    row["Interest"] = "Yes" if rng.random() < 0.18 else "No"


def build_certification_row(row: dict, person: Person, _: int, rng: random.Random) -> None:
    row["Capability Name"] = rng.choice(
        [
            "AWS Practitioner",
            "PRINCE2 Foundation",
            "Scrum Master",
            "Microsoft Power BI",
            "Azure Fundamentals",
            "CISA",
        ]
    )
    row["Interest"] = "Yes" if rng.random() < 0.1 else "No"
    issue_date = random_date_string(2017, 2025, rng) if rng.random() < 0.35 else None
    expiry_date = random_date_string(2024, 2028, rng) if rng.random() < 0.18 else None
    row["Certification Issued Date"] = issue_date
    row["Certification Expiration Date"] = expiry_date


def build_language_row(row: dict, person: Person, index: int, rng: random.Random) -> None:
    languages = ["English", "Spanish", "French", "German", "Italian", "Polish", "Arabic"]
    row["Capability Name"] = "English" if index == 0 else rng.choice(languages[1:])
    row["Proficiency"] = "Upper Advanced" if row["Capability Name"] == "English" else rng.choice(
        ["Intermediate", "Advanced", "Upper Advanced"]
    )
    row["Interest"] = "No"


def build_job_history_row(row: dict, person: Person, index: int, rng: random.Random) -> None:
    companies = ["PwC", "Deloitte", "KPMG", "EY", "HSBC", "Barclays", "NatWest"]
    positions = [
        "Business Analyst",
        "Consultant",
        "Senior Consultant",
        "Risk Analyst",
        "Data Analyst",
        "Project Analyst",
    ]
    start_year = 2012 + min(index, 10)
    row["Job History Start Date"] = random_date_string(start_year, min(start_year + 1, 2025), rng)
    row["Job History End Date"] = random_date_string(start_year + 1, 2026, rng) if rng.random() < 0.65 else None
    row["Position"] = rng.choice(positions)
    row["Company"] = rng.choice(companies)
    row["Location"] = rng.choice(["London", "Manchester", "Remote", None, None])
    row["Description"] = generate_job_description(rng) if rng.random() < 0.8 else None


def build_education_row(row: dict, person: Person, _: int, rng: random.Random) -> None:
    row["Education title/type of degree"] = rng.choice(
        ["BSc", "BA", "MSc", "MBA", "Diploma", None]
    )
    row["Area of Study"] = rng.choice(
        ["Computer Science", "Economics", "Business", "Finance", "Mathematics", None]
    )
    row["Academic Institution"] = rng.choice(
        [
            "University of Leeds",
            "University of Manchester",
            "University of Bristol",
            "University of Birmingham",
            "University of Glasgow",
        ]
    )
    row["In Process"] = rng.choice(["Yes", "No"])


def build_travel_row(row: dict, person: Person, _: int, rng: random.Random) -> None:
    travel_interest = rng.choice(["Yes", None, None, None])
    project_interest = rng.choice(["Transformation", "Risk", "Data", None])
    if travel_interest is None and project_interest is None:
        travel_interest = "Yes"
    row["Travel Interest"] = travel_interest
    row["Project Type Interest"] = project_interest
    row["Additional Information"] = "Open to short-term travel projects." if rng.random() < 0.15 else None


def random_profile_edit_date(rng: random.Random) -> str:
    return f"{rng.randint(2024, 2026):04d}/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}"


def random_date_string(start_year: int, end_year: int, rng: random.Random) -> str:
    year = rng.randint(start_year, max(start_year, end_year))
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def generate_summary(rng: random.Random) -> str:
    openings = [
        "Experienced consultant with a background in data, risk, and transformation delivery.",
        "Business-focused analyst with experience across reporting, controls, and stakeholder engagement.",
        "Consulting professional supporting data quality, governance, and delivery outcomes.",
    ]
    endings = [
        "Brings a practical approach to structured problem solving and cross-functional delivery.",
        "Comfortable translating business needs into clear analysis and execution plans.",
        "Supports complex programmes with a focus on quality, pace, and collaboration.",
    ]
    return f"{rng.choice(openings)} {rng.choice(endings)}"


def generate_job_description(rng: random.Random) -> str:
    phrases = [
        "Supported reporting and control improvements across a regulated programme.",
        "Worked with business and technology teams to refine requirements and test delivery outputs.",
        "Contributed to stakeholder coordination, issue tracking, and process improvement activity.",
    ]
    return " ".join(rng.sample(phrases, k=2))


def derive_email(name: str, domain: str, rng: random.Random) -> str:
    parts = "".join(char if char.isalpha() or char == " " else " " for char in name).lower().split()
    return f"{parts[0]}.{parts[-1]}{rng.randint(10, 99)}@{domain}"


def unique_numeric_string(length: int, seen: set[str], rng: random.Random) -> str:
    while True:
        value = "".join(rng.choice("0123456789") for _ in range(length))
        if value not in seen:
            seen.add(value)
            return value


def unique_int(min_digits: int, max_digits: int, seen: set[int], rng: random.Random) -> int:
    while True:
        digits = rng.randint(min_digits, max_digits)
        value = rng.randint(10 ** (digits - 1), (10**digits) - 1)
        if value not in seen:
            seen.add(value)
            return value


if __name__ == "__main__":
    main()
