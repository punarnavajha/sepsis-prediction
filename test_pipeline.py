import os
import glob
import pandas as pd

def test_data_generation():
    # Run the generator
    os.system('python3 generate_data.py')
    
    # Check if files were created
    files = glob.glob('sepsis_data/training/*.psv')
    assert len(files) > 0, "No synthetic files were generated"
    
    # Check if a sample file has the correct columns
    df = pd.read_csv(files[0], sep='|')
    assert 'SepsisLabel' in df.columns
    assert 'HR' in df.columns
