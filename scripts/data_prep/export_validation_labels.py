import pandas as pd
from pathlib import Path

def create_validation_csv():
    input_path = Path('data/processed/unified_v3/unified_train_v3.csv')
    output_path = Path('data/processed/unified_v3/validation_labels_v3_fixed.csv')
    
    print(f"Reading from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Apply the same fix as in the training script
    print("Applying patient_id fix (extracting from filename)...")
    df['patient_id_derived'] = df['filename'].apply(lambda x: x.split('_')[0])
    
    # Create global patient ID
    print("Creating global_patient_id...")
    df['global_patient_id'] = df['source_dataset'].astype(str) + '_' + df['patient_id_derived'].astype(str)
    
    # Select columns for validation
    cols_to_export = [
        'filename', 
        'source_dataset', 
        'patient_id_derived', 
        'global_patient_id',
        'Label_D', 'Label_G', 'Label_C', 'Label_A', 'Label_H', 'Label_M', 'Label_O',
        'image_path'
    ]
    
    # Add original ID if it exists
    if 'ID' in df.columns:
        cols_to_export.insert(0, 'ID')
        
    # Add original patient_id if it exists (to compare)
    if 'patient_id' in df.columns:
        cols_to_export.insert(3, 'patient_id')
        
    validation_df = df[cols_to_export]
    
    print(f"Saving validation CSV to {output_path}...")
    validation_df.to_csv(output_path, index=False)
    print("Done!")
    print(f"Total rows: {len(validation_df)}")
    print(f"Sample rows:\n{validation_df.head()}")

if __name__ == "__main__":
    create_validation_csv()
