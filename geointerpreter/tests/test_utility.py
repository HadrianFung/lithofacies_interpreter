import numpy as np
import pandas as pd
import pytest
from geointerpreter.extras.utility import combine_columns, load_porosity, calculate_porosity, \
    fill_random_permeability, fill_random_facies

def test_combine_columns():
    """
    Test the combine_columns function
    """
    # Create a sample dataframe
    df = pd.DataFrame({
        'DTS_1': [1, 2, 3],
        'DTS_2': [4, 5, 6],
        'RMIC': [7, 8, 9]
    })

    # Call the combine_columns function
    original_df = df.copy()
    combined_df = combine_columns(df)

    # Check if the columns have been correctly combined
    assert 'DTS1' in combined_df.columns
    assert 'DTS2' in combined_df.columns
    assert 'RMED' in combined_df.columns
    assert combined_df['DTS1'].equals(original_df['DTS_1'])
    assert combined_df['DTS2'].equals(original_df['DTS_2'])
    assert combined_df['RMED'].equals(original_df['RMIC'])

def test_load_porosity():
    """
    Test the load_porosity function
    """
    test_csv_path = "./geointerpreter/tests/test_porosity.csv"

   # Call the load_porosity function
    por_df, por_col, depth_col = load_porosity(test_csv_path)
    
    # Check the dataframe and column names
    assert isinstance(por_df, pd.DataFrame)
    assert 'por' in por_col.lower()
    assert 'depth' in depth_col.lower()

def test_calculate_porosity():
    """
    Test the calculate_porosity function
    """
    # Create a sample depth array and porosity dataframe
    depth_array = np.array([100, 200, 300, 400, 500])
    por_df = pd.DataFrame({
        'depth': [100, 200, 300, 400, 500],
        'porosity': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # Call the calculate_porosity function
    porosity = calculate_porosity(depth_array, por_df, 'porosity', 'depth')
    
    # Check if the porosity values are correctly calculated
    np.testing.assert_array_equal(porosity, por_df['porosity'].values)

    # Test with unordered depth array should raise ValueError
    unordered_depth_array = np.array([500, 100, 400, 200, 300])
    with pytest.raises(ValueError):
        calculate_porosity(unordered_depth_array, por_df, 'porosity', 'depth')

def create_test_df():
    """
    Utility function to create a test DataFrame.
    """
    return pd.DataFrame({
        'well': ['A', 'A', 'B', 'B'],
        'DEPTH': [100, 150, 100, 150],
        'core_depths': [(90, 110), (140, 160), (90, 110), (140, 160)]
    })

def test_fill_random_permeability():
    """
    Test the fill_random_permeability function
    """
    df = create_test_df()
    permeability = fill_random_permeability(df)
    
    # Check if output is a pandas Series
    assert isinstance(permeability, pd.Series)
    
    # Check if all values are within the expected range, including NaNs for out-of-range depths
    assert all((val >= 1 and val <= 100) or np.isnan(val) for val in permeability)

def test_fill_random_facies():
    """
    Test the fill_random_facies function
    """
    df = create_test_df()
    facies = fill_random_facies(df)

    # Check if output is a pandas Series
    assert isinstance(facies, pd.Series)

    # Check if all values are from the specified categories or NaNs for out-of-range depths
    expected_categories = {"os", "s", "sh", "ms", np.nan}
    assert all(f in expected_categories for f in facies)
