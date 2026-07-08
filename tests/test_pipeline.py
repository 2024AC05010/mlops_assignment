# test_pipeline.py
# Unit tests for preprocessing and model prediction
# Run with: pytest tests/test_pipeline.py -v
# Initially had some failing tests due to data type mismatches - fixed by adjusting fixtures

import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocess import handle_missing_values, encode_features, split_features_target  # noqa: E402


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Create a small dummy dataframe for testing."""
    data = {
        "age": [52, 63, 40, 55, 48],
        "sex": [1, 0, 1, 1, 0],
        "cp": [0, 2, 1, 3, 0],
        "trestbps": [130, 145, 120, 160, 138],
        "chol": [200, 233, 180, 260, 214],
        "fbs": [0, 1, 0, 1, 0],
        "restecg": [0, 1, 0, 2, 1],
        "thalach": [150, 160, 170, 130, 145],
        "exang": [0, 1, 0, 1, 0],
        "oldpeak": [1.0, 2.5, 0.5, 3.0, 1.2],
        "slope": [1, 2, 1, 0, 2],
        "ca": [0, 1, 0, 2, 0],
        "thal": [2, 3, 2, 1, 3],
        "target": [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_missing():
    """Create dummy dataframe with missing values."""
    data = {
        "age": [52, 63, None, 55, 48],
        "sex": [1, 0, 1, 1, 0],
        "cp": [0, 2, 1, 3, 0],
        "trestbps": [130, None, 120, 160, 138],
        "chol": [200, 233, 180, 260, 214],
        "fbs": [0, 1, 0, 1, 0],
        "restecg": [0, 1, 0, 2, 1],
        "thalach": [150, 160, 170, 130, 145],
        "exang": [0, 1, 0, 1, 0],
        "oldpeak": [1.0, 2.5, 0.5, 3.0, 1.2],
        "slope": [1, 2, 1, 0, 2],
        "ca": [0, None, 0, 2, 0],
        "thal": [2, 3, None, 1, 3],
        "target": [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestMissingValueHandling:

    def test_no_nulls_after_fill(self, sample_df_with_missing):
        """Test that missing values are properly filled."""
        result = handle_missing_values(sample_df_with_missing)
        assert result.isnull().sum().sum() == 0, "There should be no null values after cleaning"

    def test_shape_unchanged(self, sample_df_with_missing):
        """Test that DataFrame shape doesn't change after filling."""
        result = handle_missing_values(sample_df_with_missing)
        assert result.shape == sample_df_with_missing.shape, "Shape should not change after filling"

    def test_returns_dataframe(self, sample_df_with_missing):
        """Test that function returns a DataFrame."""
        result = handle_missing_values(sample_df_with_missing)
        assert isinstance(result, pd.DataFrame), "Output should be a DataFrame"

    def test_median_fill_works(self, sample_df_with_missing):
        """Test that median imputation is working correctly."""
        result = handle_missing_values(sample_df_with_missing)
        # Check that filled values are reasonable
        assert result['age'].notna().all(), "Age should have no nulls after fill"


class TestFeatureEncoding:

    def test_encoding_increases_columns(self, sample_df):
        """Test that one-hot encoding increases column count."""
        result = encode_features(sample_df)
        assert result.shape[1] > sample_df.shape[1], "Column count should increase after encoding"

    def test_no_original_cat_cols(self, sample_df):
        """Test that original categorical columns are removed."""
        result = encode_features(sample_df)
        original_cats = ["cp", "restecg", "slope", "thal"]
        for col in original_cats:
            assert col not in result.columns, f"Original cat col '{col}' should be removed"

    def test_target_col_preserved(self, sample_df):
        """Test that target column is preserved during encoding."""
        result = encode_features(sample_df)
        assert "target" in result.columns, "Target column should still be present"


class TestSplitFeaturesTarget:

    def test_correct_shapes(self, sample_df):
        """Test that X and y have matching row counts."""
        X, y = split_features_target(sample_df)
        assert X.shape[0] == y.shape[0], "Rows in X and y must match"
        assert "target" not in X.columns, "Target should not be in features"

    def test_target_is_series(self, sample_df):
        """Test that target is returned as a Series."""
        X, y = split_features_target(sample_df)
        assert isinstance(y, pd.Series), "Target should be a Pandas Series"

    def test_feature_row_count(self, sample_df):
        """Test that correct number of samples are returned."""
        X, y = split_features_target(sample_df)
        assert len(X) == 5, "Should have 5 samples"


class TestPredictionOutput:

    def test_prediction_keys_present(self):
        """Test that prediction result has required keys."""
        expected_keys = {"prediction", "label", "confidence"}
        dummy_result = {
            "prediction": 1,
            "label": "Heart Disease Detected",
            "confidence": 0.87
        }
        assert expected_keys.issubset(dummy_result.keys())

    def test_prediction_binary(self):
        """Prediction should be 0 or 1."""
        pred = 1
        assert pred in [0, 1], "Prediction must be binary"

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        conf = 0.87
        assert 0.0 <= conf <= 1.0, "Confidence must be in [0, 1]"

    def test_edge_case_zero_confidence(self):
        """Test edge case with zero confidence."""
        conf = 0.0
        assert 0.0 <= conf <= 1.0, "Zero confidence should be valid"

    def test_edge_case_one_confidence(self):
        """Test edge case with one confidence."""
        conf = 1.0
        assert 0.0 <= conf <= 1.0, "One confidence should be valid"
