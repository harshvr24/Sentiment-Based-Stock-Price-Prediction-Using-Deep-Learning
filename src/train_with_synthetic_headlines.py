import os
import random
import pandas as pd
import subprocess
import tempfile

# Number of synthetic samples to generate
NUM_SAMPLES = 1000

# Example headline templates
POSITIVE_TEMPLATES = [
    "{company} stock surges after strong earnings report",
    "{company} announces record profits for the quarter",
    "{company} shares rise on positive outlook",
    "{company} beats analyst expectations, stock jumps",
    "{company} secures major new contract, shares up",
]
NEGATIVE_TEMPLATES = [
    "{company} stock plunges after weak earnings report",
    "{company} announces losses for the quarter",
    "{company} shares fall on negative outlook",
    "{company} misses analyst expectations, stock drops",
    "{company} faces regulatory issues, shares down",
]
COMPANIES = [
    "Apple", "Google", "Microsoft", "Amazon", "Tesla", "Meta", "Netflix", "Nvidia", "JPMorgan", "Visa"
]

def generate_headlines(num_samples):
    data = []
    for _ in range(num_samples):
        label = random.randint(0, 1)
        company = random.choice(COMPANIES)
        if label == 1:
            template = random.choice(POSITIVE_TEMPLATES)
        else:
            template = random.choice(NEGATIVE_TEMPLATES)
        headline = template.format(company=company)
        data.append({"headline": headline, "label": label})
    return pd.DataFrame(data)

def main():
    # Generate synthetic data
    df = generate_headlines(NUM_SAMPLES)
    # Save to a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        temp_csv_path = tmp.name
    print(f"Synthetic data saved to: {temp_csv_path}")

    # Call the training script with the synthetic data
    train_cmd = [
        'python', 'src/train_model.py',
        '--data_paths', temp_csv_path,
        '--epochs', '100',  # Increased epochs for thorough training
        '--save_final_model'
    ]
    print(f"Running training: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    # Delete the temporary CSV file
    os.remove(temp_csv_path)
    print(f"Temporary synthetic data deleted: {temp_csv_path}")

if __name__ == '__main__':
    main() 