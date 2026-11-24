# Dataset location

Keep the RBC-QI dataset outside source control. The trainer expects:

```
RBC_QI_DATASET/
  train/<class-name>/*
  test/<class-name>/*
```

You may copy the dataset here or create a symlink named `RBC_QI_DATASET` to its storage location.
