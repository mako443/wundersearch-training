# Wundersearch Training
Model training for the WunderSearch text-to-image search app.

# CoreML Deploy
- Multiple models
- Text-handling ok? Otherwise make the padding-logic external, pad to longest sentence in training
- CARE: Remember to normalize in iOS! TODO: Verify equal outputs
- Maybe .json file for handling?

# TODO
Training
- Don't train the image model? Only lower layers?
- Random-search LR vs. margin

General
- Use a pre-trained model?

- PWR or HRL loss?
- Quantize models?
- Use VSE and detection ensemble?
- Use stronger cross-modal model?
