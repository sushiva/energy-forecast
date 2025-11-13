"""
Feature Engineering Module
Create domain knowledge and automated features for energy forecasting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class EnergyFeatureEngineer:
    """
    Feature engineering for energy forecasting
    Combines domain knowledge with automated feature generation
    """
    
    def __init__(self, create_domain_features=True, create_interactions=True, 
                 create_polynomial=False, poly_degree=2):
        """
        Initialize feature engineer
        
        Parameters:
        -----------
        create_domain_features : bool
            Create physics/engineering-based features
        create_interactions : bool
            Create interaction features
        create_polynomial : bool
            Create polynomial features (automated)
        poly_degree : int
            Degree for polynomial features
        """
        self.create_domain_features = create_domain_features
        self.create_interactions = create_interactions
        self.create_polynomial = create_polynomial
        self.poly_degree = poly_degree
        self.poly_transformer = None
        self.feature_names = []
        
    def fit(self, X, y=None):
        """
        Fit the feature engineer (mainly for polynomial features)
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable (not used)
        """
        if self.create_polynomial:
            self.poly_transformer = PolynomialFeatures(
                degree=self.poly_degree, 
                include_bias=False
            )
            # Fit on original features only
            self.poly_transformer.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Transform features by adding engineered features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Features with engineered columns added
        """
        X_eng = X.copy()
        
        if self.create_domain_features:
            X_eng = self._add_domain_features(X_eng)
        
        if self.create_interactions:
            X_eng = self._add_interaction_features(X_eng)
        
        if self.create_polynomial and self.poly_transformer is not None:
            X_eng = self._add_polynomial_features(X_eng, X)
        
        return X_eng
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series, optional
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def _add_domain_features(self, X):
        """
        Add domain knowledge features based on building physics
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Features with domain features added
        """
        print("\nCreating domain knowledge features...")
        
        # Feature 1: Total Window Area (Surface Area × Glazing Ratio)
        # Domain: Total glass area affects heat loss
        X['total_window_area'] = X['X2'] * X['X7']
        print("  ✓ Created: total_window_area = X2 × X7")
        
        # Feature 2: Wall-to-Surface Ratio
        # Domain: More walls relative to surface = less compact = more heat loss
        X['wall_surface_ratio'] = X['X3'] / (X['X2'] + 1e-6)  # Avoid division by zero
        print("  ✓ Created: wall_surface_ratio = X3 / X2")
        
        # Feature 3: Surface-to-Volume Ratio (approximation)
        # Domain: Higher ratio = more surface per volume = more heat loss
        # Volume ≈ Surface Area × Height / 2 (simplified)
        volume_approx = X['X2'] * X['X5'] / 2
        X['surface_volume_ratio'] = X['X2'] / (volume_approx + 1e-6)
        print("  ✓ Created: surface_volume_ratio = X2 / (X2 × X5 / 2)")
        
        # Feature 4: Thermal Mass Indicator
        # Domain: Compact buildings retain heat better
        X['thermal_mass'] = X['X1'] * (1 / (X['X2'] + 1e-6))
        print("  ✓ Created: thermal_mass = X1 / X2")
        
        # Feature 5: Solar Exposure Factor
        # Domain: South/West get more sun, amplified by glazing
        # Orientation: 2=North(low), 3=East(med), 4=South(high), 5=West(high)
        orientation_factors = {2: 0.3, 3: 0.7, 4: 1.0, 5: 0.9}
        X['orientation_factor'] = X['X6'].map(orientation_factors)
        X['solar_exposure'] = X['orientation_factor'] * X['X7']
        print("  ✓ Created: solar_exposure = orientation_factor × X7")
        
        # Feature 6: Roof-to-Floor Ratio
        # Domain: Roof area relative to floor indicates building footprint
        X['roof_floor_ratio'] = X['X4'] / (X['X2'] + 1e-6)
        print("  ✓ Created: roof_floor_ratio = X4 / X2")
        
        # Feature 7: Glazing Density (non-zero glazing indicator)
        # Domain: Buildings with any glazing behave differently
        X['has_glazing'] = (X['X7'] > 0).astype(int)
        print("  ✓ Created: has_glazing = (X7 > 0)")
        
        # Feature 8: Compactness Score
        # Domain: Combined measure of building efficiency
        X['compactness_score'] = X['X1'] * (1 - X['X7'])  # High compact, low glazing = efficient
        print("  ✓ Created: compactness_score = X1 × (1 - X7)")
        
        print(f"\nTotal domain features created: 8")
        
        return X
    
    def _add_interaction_features(self, X):
        """
        Add meaningful interaction features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            Features with interactions added
        """
        print("\nCreating interaction features...")
        
        # Interaction 1: Height × Compactness
        # Meaning: Tall compact buildings behave differently
        X['height_compact'] = X['X5'] * X['X1']
        print("  ✓ Created: height_compact = X5 × X1")
        
        # Interaction 2: Surface × Glazing
        # Meaning: Large buildings with high glazing lose much more energy
        X['surface_glazing'] = X['X2'] * X['X7']
        print("  ✓ Created: surface_glazing = X2 × X7")
        
        # Interaction 3: Wall × Glazing
        # Meaning: More wall area with more windows = more thermal boundary
        X['wall_glazing'] = X['X3'] * X['X7']
        print("  ✓ Created: wall_glazing = X3 × X7")
        
        # Interaction 4: Height × Surface
        # Meaning: Building volume indicator
        X['height_surface'] = X['X5'] * X['X2']
        print("  ✓ Created: height_surface = X5 × X2")
        
        print(f"\nTotal interaction features created: 4")
        
        return X
    
    def _add_polynomial_features(self, X_current, X_original):
        """
        Add polynomial features from original features only
        
        Parameters:
        -----------
        X_current : pd.DataFrame
            Current features (with engineered features)
        X_original : pd.DataFrame
            Original input features
            
        Returns:
        --------
        pd.DataFrame
            Features with polynomial terms added
        """
        print("\nCreating polynomial features...")
        
        # Create polynomial features from original features only
        X_poly_array = self.poly_transformer.transform(X_original)
        
        # Get feature names
        poly_feature_names = self.poly_transformer.get_feature_names_out(X_original.columns)
        
        # Create DataFrame
        X_poly = pd.DataFrame(
            X_poly_array,
            columns=poly_feature_names,
            index=X_current.index
        )
        
        # Remove original features (already in X_current)
        original_features = X_original.columns.tolist()
        poly_only = X_poly.drop(columns=original_features, errors='ignore')
        
        # Concatenate
        X_result = pd.concat([X_current, poly_only], axis=1)
        
        print(f"  ✓ Created {len(poly_only.columns)} polynomial features")
        
        return X_result
    
    def get_feature_names(self, X):
        """
        Get names of all features after transformation
        
        Parameters:
        -----------
        X : pd.DataFrame
            Transformed features
            
        Returns:
        --------
        list
            Feature names
        """
        return X.columns.tolist()


def create_engineered_features(X, y=None, domain_only=True):
    """
    Convenience function to create engineered features
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series, optional
        Target variable
    domain_only : bool
        If True, create only domain features. If False, include polynomial
        
    Returns:
    --------
    pd.DataFrame
        Engineered features
    """
    engineer = EnergyFeatureEngineer(
        create_domain_features=True,
        create_interactions=True,
        create_polynomial=not domain_only,
        poly_degree=2
    )
    
    X_eng = engineer.fit_transform(X, y)
    
    return X_eng