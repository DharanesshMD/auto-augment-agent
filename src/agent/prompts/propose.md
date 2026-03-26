## Current Trial: {trial_number} / {max_trials}
## Phase: {phase}

---

### Current Augmentation Config
```yaml
{current_augmentations}
```

### Current Tuning Config
```yaml
{current_tuning}
```

### Recent Trial History (last 5)
```yaml
{trial_history}
```

### Best Trial So Far
```yaml
{best_trial}
```

---

### Allowed Parameter Ranges (Augmentations)
```json
{augmentation_schema}
```

### Allowed Parameter Ranges (Tuning)
```json
{tuning_schema}
```

---

## Your Task
Propose changes to the augmentation pipeline and/or tuning hyperparameters that you believe will improve the validation metric.

Respond with:
1. **Reasoning**: Why you're making these changes (2-3 sentences)
2. **Augmentation patch** (if any): A ```yaml block with the full augmentations config
3. **Tuning patch** (if any): A ```yaml block with the full tuning config

Remember:
- All values MUST be within the allowed ranges
- You can change one or both configs
- Build on what worked, avoid what didn't
