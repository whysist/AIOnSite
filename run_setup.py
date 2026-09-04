# run_setup.py
from src.database.bootstrap import initialize_database
from src.database.seed import seed_synthetic_database

def main():
    db_name = "industrial_data.db"
    print(f"[*] Bootstrapping {db_name} schema & indexes...")
    initialize_database(db_name)
    print(f"[*] Seeding synthetic records for V-101...")
    seed_synthetic_database(db_name)
    print("[+] Database ready for sovereign, air-gapped agent operations.")

if __name__ == "__main__":
    main()