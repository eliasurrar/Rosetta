# Guide: Modifying Predictor Variables in the Interactive Script

When you add, remove, or change predictor variables in your model, you need to update **4 key sections** in the script. Here's a comprehensive checklist:

---

## Section 1: Slider Configurations (Lines ~293-310)

**Location:** `slider_configs` dictionary

**What to update:**
- Add/remove/modify slider entries for your predictor variables
- **DO NOT** include `catalyst_start_day`, `catalyst_dose`, or `max_time` here (these are simulation parameters, not predictors)
- **DO NOT** include `cyanide_soluble` here (it's calculated automatically)

**Example:**
```python
slider_configs = {
    'acid_soluble': {'label': 'Acid Soluble %', 'min': 1.0, 'max': 26.0, 'step': 0.5, 'value': 3.5},
    'residual_cpy': {'label': 'Residual Cpy %', 'min': 23.0, 'max': 96.0, 'step': 0.5, 'value': 88.5},
    'material_size_p80': {'label': 'Material Size P80 (in)', 'min': 0.5, 'max': 8.0, 'step': 0.1, 'value': 1.0},
    'copper_sulfides': {'label': 'Copper Sulfides', 'min': 0.20, 'max': 1.3, 'step': 0.05, 'value': 0.55},
    # ... add your new predictors here
    # 'new_predictor': {'label': 'New Predictor Name', 'min': 0.0, 'max': 10.0, 'step': 0.1, 'value': 5.0},
}
```

**Important Notes:**
- The **key names** (e.g., `'acid_soluble'`) will be used as slider IDs in the UI
- The **order matters** - it determines the order of sliders in the callback inputs
- Min/max should match the training data range
- Default `value` should be a typical/representative value

---

## Section 2: Layout - Slider Rows (Lines ~330-390)

**Location:** Inside `app.layout` → `dbc.CardBody`

**What to update:**
- Add/remove `create_slider_row()` calls for each predictor variable
- Group related variables together in cards

**Example:**
```python
dbc.Card([
    dbc.CardHeader(html.H5("Sample Features")),
    dbc.CardBody([
        create_slider_row('acid_soluble', slider_configs['acid_soluble']),
        create_slider_row('residual_cpy', slider_configs['residual_cpy']),
        create_slider_row('material_size_p80', slider_configs['material_size_p80']),
        create_slider_row('copper_sulfides', slider_configs['copper_sulfides']),
        # ... add your new predictor sliders here
        # create_slider_row('new_predictor', slider_configs['new_predictor']),
    ])
], className='mb-3'),
```

**Important Notes:**
- The slider ID (first argument) must match the key in `slider_configs`
- Keep the same order as in `slider_configs` for consistency

---

## Section 3: Callback Function Signature (Lines ~470-481)

**Location:** `@app.callback` → `def update_plot(...)` parameters

**What to update:**
- Add/remove parameters for each predictor variable
- **Order must match** the order in `slider_configs`
- Keep `catalyst_start_day`, `catalyst_dose`, `max_time`, `cyanide_soluble` at the end

**Example:**
```python
@app.callback(
    [Output('prediction-plot', 'figure'),
     Output('parameters-output', 'children')],
    [Input(f'slider-{k}', 'value') for k in slider_configs.keys()] +
    [Input('slider-catalyst_start_day', 'value'),
     Input('slider-catalyst_dose', 'value'),
     Input('slider-max_time', 'value'),
     Input('slider-cyanide_soluble', 'value')]
)
def update_plot(acid_soluble, residual_cpy, material_size_p80, 
                copper_sulfides, secondary_copper, acid_gen_sulfides,
                gangue_silicates, fe_oxides, carbonates,
                # new_predictor,  # ADD NEW PARAMETERS HERE
                catalyst_start_day, catalyst_dose, max_time, cyanide_soluble):
```

**Critical:** The parameter names here can be anything, but their **position** must match the slider order!

---

## Section 4: Sample Features Dictionary (Lines ~484-493)

**Location:** Inside `update_plot()` → `sample_features` dictionary

**What to update:**
- Add/remove entries for each predictor variable
- **Key names must match** the column names in `num_cols` (from your trained model)
- **Values** come from the callback parameters

**Example:**
```python
sample_features = {
    'acid_soluble_%': acid_soluble,
    'residual_cpy_%': residual_cpy,
    'material_size_p80_in': material_size_p80,
    'grouped_copper_sulfides': copper_sulfides,
    'grouped_secondary_copper': secondary_copper,
    'grouped_acid_generating_sulfides': acid_gen_sulfides,
    'grouped_gangue_silicates': gangue_silicates,
    'grouped_fe_oxides': fe_oxides,
    'grouped_carbonates': carbonates,
    # 'new_predictor_column_name': new_predictor,  # ADD NEW FEATURES HERE
}
```

**Critical Notes:**
- The **keys** (left side) must **exactly match** the column names in your model's training data
- Check `num_cols` from your saved model to verify the exact column names
- The **values** (right side) are the parameter names from the callback function

---

## Complete Example: Adding a New Predictor Variable

Let's say you want to add a new predictor called **"pH_level"** with column name **"pH"** in your model.

### Step 1: Add to `slider_configs`
```python
slider_configs = {
    'acid_soluble': {...},
    'residual_cpy': {...},
    # ... existing predictors
    'ph_level': {'label': 'pH Level', 'min': 1.0, 'max': 14.0, 'step': 0.1, 'value': 7.0},  # NEW
}
```

### Step 2: Add to layout
```python
dbc.CardBody([
    create_slider_row('acid_soluble', slider_configs['acid_soluble']),
    create_slider_row('residual_cpy', slider_configs['residual_cpy']),
    # ... existing sliders
    create_slider_row('ph_level', slider_configs['ph_level']),  # NEW
])
```

### Step 3: Add to callback parameters
```python
def update_plot(acid_soluble, residual_cpy, material_size_p80, 
                copper_sulfides, secondary_copper, acid_gen_sulfides,
                gangue_silicates, fe_oxides, carbonates,
                ph_level,  # NEW - add in the same order as slider_configs
                catalyst_start_day, catalyst_dose, max_time, cyanide_soluble):
```

### Step 4: Add to sample_features
```python
sample_features = {
    'acid_soluble_%': acid_soluble,
    'residual_cpy_%': residual_cpy,
    'material_size_p80_in': material_size_p80,
    'grouped_copper_sulfides': copper_sulfides,
    'grouped_secondary_copper': secondary_copper,
    'grouped_acid_generating_sulfides': acid_gen_sulfides,
    'grouped_gangue_silicates': gangue_silicates,
    'grouped_fe_oxides': fe_oxides,
    'grouped_carbonates': carbonates,
    'pH': ph_level,  # NEW - key must match model column name
}
```

---

## Complete Example: Removing a Predictor Variable

Let's say you want to remove **"secondary_copper"** from the model.

### Step 1: Remove from `slider_configs`
```python
slider_configs = {
    'acid_soluble': {...},
    'residual_cpy': {...},
    'copper_sulfides': {...},
    # 'secondary_copper': {...},  # REMOVE THIS LINE
    'acid_gen_sulfides': {...},
    # ... rest
}
```

### Step 2: Remove from layout
```python
dbc.CardBody([
    create_slider_row('acid_soluble', slider_configs['acid_soluble']),
    create_slider_row('copper_sulfides', slider_configs['copper_sulfides']),
    # create_slider_row('secondary_copper', slider_configs['secondary_copper']),  # REMOVE THIS LINE
    create_slider_row('acid_gen_sulfides', slider_configs['acid_gen_sulfides']),
])
```

### Step 3: Remove from callback parameters
```python
def update_plot(acid_soluble, residual_cpy, material_size_p80, 
                copper_sulfides,  # secondary_copper,  # REMOVE THIS PARAMETER
                acid_gen_sulfides, gangue_silicates, fe_oxides, carbonates,
                catalyst_start_day, catalyst_dose, max_time, cyanide_soluble):
```

### Step 4: Remove from sample_features
```python
sample_features = {
    'acid_soluble_%': acid_soluble,
    'residual_cpy_%': residual_cpy,
    'material_size_p80_in': material_size_p80,
    'grouped_copper_sulfides': copper_sulfides,
    # 'grouped_secondary_copper': secondary_copper,  # REMOVE THIS LINE
    'grouped_acid_generating_sulfides': acid_gen_sulfides,
    # ... rest
}
```

---

## Verification Checklist

After making changes, verify:

- [ ] Number of sliders in `slider_configs` matches number of predictors
- [ ] Number of `create_slider_row()` calls matches number of sliders
- [ ] Number of callback parameters (excluding catalyst/time params) matches number of sliders
- [ ] Number of entries in `sample_features` matches number of predictors
- [ ] Column names in `sample_features` match `num_cols` from your model
- [ ] Order is consistent across all 4 sections
- [ ] Script runs without errors: `python Example_usage_NN_v15_interactive.py`
- [ ] Check console output: `✓ Number of features: X` should match your model's feature count

---

## Common Pitfalls

❌ **Don't do this:**
- Adding a predictor to `slider_configs` but forgetting to add it to `sample_features`
- Using different order in callback parameters vs `slider_configs`
- Using wrong column names in `sample_features` (must match model training data)
- Including `cyanide_soluble` in `sample_features` (it's not a model input)
- Including simulation parameters (`catalyst_start_day`, `catalyst_dose`, `max_time`) in `sample_features`

✅ **Do this:**
- Keep the order consistent across all 4 sections
- Use exact column names from `num_cols` in `sample_features`
- Test with a known sample to verify predictions are correct
- Check the model's `num_cols` attribute to verify expected features

---

## Debugging Tips

If you get errors after modifying predictors:

1. **"ValueError: X has Y features, but model expects Z features"**
   - Check that `sample_features` has the correct number of entries
   - Verify column names match `num_cols` exactly

2. **"KeyError: 'some_column'"**
   - The column name in `sample_features` doesn't exist in `num_cols`
   - Check spelling and exact naming (case-sensitive!)

3. **Callback errors or wrong values**
   - Verify parameter order in callback function matches `slider_configs` order
   - Check that you're passing the right parameter to the right key in `sample_features`

4. **To debug, add this after `sample_features`:**
   ```python
   print("Expected columns:", num_cols)
   print("Provided columns:", list(sample_features.keys()))
   print("Sample values:", X_new_df.values)
   ```

---

## Summary Table

| Section | Location | What to Update | Key Requirement |
|---------|----------|----------------|-----------------|
| 1. Slider Configs | Lines ~293-310 | Add/remove slider definitions | Keys become slider IDs |
| 2. Layout | Lines ~330-390 | Add/remove `create_slider_row()` calls | Match slider_configs keys |
| 3. Callback Params | Lines ~470-481 | Add/remove function parameters | **Order must match slider_configs** |
| 4. Sample Features | Lines ~484-493 | Add/remove dictionary entries | **Keys must match num_cols exactly** |

**Golden Rule:** Keep the order consistent in sections 1-3, and use exact column names in section 4!

