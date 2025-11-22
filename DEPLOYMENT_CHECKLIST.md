# HuggingFace Spaces Deployment Checklist

## 🎯 Overview

Your app needs different model paths for:
- **Local**: `models/advanced/xgboost_best.pkl`
- **Deployment**: `xgboost_best.pkl` (in project root)

## ✅ Complete Deployment Process

### Step 1: Prepare Model for Deployment

```bash
cd ~/portfolio-project/energy-forecast

# Copy the FIXED model to project root for deployment
cp models/advanced/xgboost_best.pkl xgboost_best.pkl

# Verify it's there
ls -lh xgboost_best.pkl
```

### Step 2: Update app.py for Environment Detection

**Option A: Auto-detect (Recommended)**

Add this to the top of your `deployment/api/app.py`:

```python
import os
import joblib

# Auto-detect environment
def load_model_auto():
    """Load model with auto-detected path"""
    # Check for HuggingFace Spaces environment
    is_deployment = os.getenv('SPACE_ID') is not None or os.path.exists('xgboost_best.pkl')
    
    if is_deployment:
        model_path = 'xgboost_best.pkl'  # HuggingFace Spaces
        print("🚀 Loading model from deployment path")
    else:
        model_path = 'models/advanced/xgboost_best.pkl'  # Local
        print("💻 Loading model from local path")
    
    return joblib.load(model_path)

# Use this instead of direct joblib.load()
model_data = load_model_auto()
model = model_data['model']
```

**Option B: Manual Toggle (Simple)**

```python
import joblib

# Toggle this flag for deployment
DEPLOYMENT = False  # Set to True when deploying to HuggingFace

if DEPLOYMENT:
    model_data = joblib.load('xgboost_best.pkl')
else:
    model_data = joblib.load('models/advanced/xgboost_best.pkl')

model = model_data['model']
```

**Option C: Environment Variable (Most Flexible)**

```python
import os
import joblib

# Set environment variable: export DEPLOYMENT=true
is_deployment = os.getenv('DEPLOYMENT', 'false').lower() == 'true'

model_path = 'xgboost_best.pkl' if is_deployment else 'models/advanced/xgboost_best.pkl'
model_data = joblib.load(model_path)
model = model_data['model']
```

### Step 3: Prepare Files for HuggingFace

```bash
cd ~/portfolio-project/energy-forecast

# Create a clean deployment directory (optional)
mkdir -p deployment/huggingface
cd deployment/huggingface

# Copy necessary files
cp ../../xgboost_best.pkl .
cp ../api/app.py .
cp ../../requirements.txt .

# Create README for HuggingFace (optional)
cp ../../README.md .
```

### Step 4: Git Setup for Deployment

**If deploying from root directory:**

```bash
cd ~/portfolio-project/energy-forecast

# Make sure model is in root
cp models/advanced/xgboost_best.pkl .

# Add to git
git add xgboost_best.pkl
git add deployment/api/app.py
git add requirements.txt

# Commit
git commit -m "Deploy: Add fixed model to root for HuggingFace Spaces"

# Push to your repository
git push origin main
```

**If deploying from deployment/api/ subdirectory:**

```bash
cd ~/portfolio-project/energy-forecast/deployment/api

# Copy model here
cp ../../models/advanced/xgboost_best.pkl .

# Update app.py to use local path
# model_path = 'xgboost_best.pkl'

git add xgboost_best.pkl app.py
git commit -m "Deploy: Prepare for HuggingFace Spaces deployment"
git push origin main
```

### Step 5: HuggingFace Spaces Configuration

When creating/updating your HuggingFace Space:

1. **Space Type**: Gradio
2. **SDK**: Gradio
3. **Python Version**: 3.11

4. **Files Structure** should be:
   ```
   your-space/
   ├── app.py                    # Your Gradio app
   ├── xgboost_best.pkl         # Model file (CRITICAL!)
   ├── requirements.txt          # Dependencies
   └── README.md                 # Optional
   ```

5. **App File Path**: 
   - If files are in root: `app.py`
   - If in subdirectory: `deployment/api/app.py`

### Step 6: Update requirements.txt

Make sure your `requirements.txt` includes all dependencies:

```txt
gradio==4.44.0
xgboost==2.0.3
scikit-learn==1.3.2
shap==0.44.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
joblib==1.3.2
```

### Step 7: Test Locally Before Deployment

```bash
# Test with deployment path
cd ~/portfolio-project/energy-forecast

# Make sure model is in root
ls -lh xgboost_best.pkl

# Update app.py temporarily to use root path
# model_path = 'xgboost_best.pkl'

# Run app
python deployment/api/app.py

# Test in browser at http://localhost:7860
# Verify X1 = 1.61 → ~16 kWh (compact, efficient)
# Verify X1 = 1.02 → ~35 kWh (elongated, inefficient)
```

### Step 8: Deploy to HuggingFace

**Method 1: Git Push (Recommended)**

```bash
# HuggingFace Spaces uses Git
# Link your space to your repository

# Add HuggingFace as remote (if not already)
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

# Push to HuggingFace
git push huggingface main
```

**Method 2: Web Upload**

1. Go to your HuggingFace Space
2. Click "Files" tab
3. Upload/replace files:
   - `app.py`
   - `xgboost_best.pkl`
   - `requirements.txt`

### Step 9: Verify Deployment

After deployment:

1. Visit your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Wait for build to complete (~2-5 minutes)
3. Test the interface:
   - Set X1 = 1.61 → Should predict ~16 kWh with BLUE SHAP arrow
   - Set X1 = 1.02 → Should predict ~35 kWh with RED SHAP arrow

### Step 10: Rollback Plan (If Issues)

If deployment has issues:

```bash
# Revert to previous version
git revert HEAD
git push huggingface main

# Or force push previous working version
git reset --hard PREVIOUS_COMMIT_HASH
git push --force huggingface main
```

## 🔧 Troubleshooting

### Issue: "Model file not found"

**Solution:**
```python
# Add debug logging in app.py
import os
print(f"Current directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Model file exists: {os.path.exists('xgboost_best.pkl')}")
```

### Issue: "Wrong predictions after deployment"

**Causes:**
1. Old model file deployed (inverted X1)
2. Feature scaling issues
3. Model/code mismatch

**Solution:**
- Verify `xgboost_best.pkl` is the FIXED version
- Check file size (should be same as local)
- Download deployed model and test locally

### Issue: "Model loads but predictions are wrong"

**Debug:**
```python
# Add validation in app.py
def validate_model(model_data):
    """Validate model has correct behavior"""
    import numpy as np
    
    # Test case: Compact building
    test_compact = np.array([[1.61, 637, 318, 147, 5.25, 3, 0.0, 2]])
    pred_compact = model_data['model'].predict(test_compact)[0]
    
    # Test case: Elongated building  
    test_elongated = np.array([[1.02, 637, 318, 147, 5.25, 3, 0.0, 2]])
    pred_elongated = model_data['model'].predict(test_elongated)[0]
    
    print(f"Validation:")
    print(f"  Compact (X1=1.61): {pred_compact:.2f} kWh (should be ~15-20)")
    print(f"  Elongated (X1=1.02): {pred_elongated:.2f} kWh (should be ~35-40)")
    
    if pred_compact < pred_elongated:
        print("  ✅ Model behavior is CORRECT")
        return True
    else:
        print("  ❌ Model behavior is INVERTED!")
        return False

# Call after loading model
model_data = load_model_auto()
validate_model(model_data)
```

## 📝 Quick Reference

### Local Development
```python
model_path = 'models/advanced/xgboost_best.pkl'
```

### HuggingFace Deployment
```python
model_path = 'xgboost_best.pkl'
```

### Auto-Detect (Best Practice)
```python
import os
is_deployment = os.path.exists('xgboost_best.pkl')
model_path = 'xgboost_best.pkl' if is_deployment else 'models/advanced/xgboost_best.pkl'
```

## 🎯 Final Checklist

Before deploying, verify:

- [ ] Model file `xgboost_best.pkl` is the FIXED version (not inverted)
- [ ] Model file copied to project root
- [ ] app.py updated to handle both local and deployment paths
- [ ] requirements.txt is complete and up-to-date
- [ ] Tested locally with root path (`xgboost_best.pkl`)
- [ ] X1 slider range: 1.02 to 1.61
- [ ] X1 label includes "(Higher = More Compact)"
- [ ] Validation: X1=1.61 → ~16 kWh, X1=1.02 → ~35 kWh
- [ ] Git repository is clean and committed
- [ ] README.md is updated with correct info

## 🚀 You're Ready to Deploy!

Once all checklist items are complete, push to HuggingFace and your corrected model will be live!
