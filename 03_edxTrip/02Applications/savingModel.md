# saved model

```

model = ...
model.compile(...
model.fit(...

export_dir = 'saved_model/1'
tf.saved_model.save(model, export_dir)


```

```
1
├── [4.0K]  assets
├── [  58]  fingerprint.pb
├── [ 38K]  saved_model.pb
└── [4.0K]  variables
    ├── [2.3K]  variables.data-00000-of-00001
    └── [ 621]  variables.index
```


## TFLiteConverted

```
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
```

```
du -h model.tflite
4.0K    model.tflite
```